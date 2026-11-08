"""Importação e validação de resultados históricos a partir de CSV.

Formatos aceitos (cabeçalho obrigatório, separador ";", "," ou tabulação):

    concurso;data;extracao;premio_1;premio_2;premio_3;premio_4;premio_5
    concurso;data;bilhete_1;valor_nominal_1;...;bilhete_5;valor_nominal_5;...

- `extracao` é opcional (vazio = não informada).
- Datas aceitas: DD/MM/AAAA, AAAA-MM-DD ou DD-MM-AAAA, desde 1962 até hoje.
- Números são lidos como texto. Ajustes feitos (sempre contados no relatório):
  * apóstrofo inicial ('032588), usado para o Excel manter zeros, é removido;
  * 6 dígitos começando com 0 (032588) viram 5 dígitos (32588 -> "32588");
    6 dígitos com outro primeiro dígito são recusados;
  * menos de 5 dígitos (1339) recebem zeros à esquerda (01339).
- Colunas extras não são descartadas: as que mudam de linha para linha ficam
  em `dados_originais` de cada concurso; as que têm o mesmo valor no arquivo
  inteiro ficam uma única vez nos metadados da importação.
- Concursos com todos os `valor_nominal_1..5` iguais a zero são marcados
  como `premio_zero` (por exemplo, julho de 2026, sem venda de bilhetes).
  Valor vazio não é tratado como zero.
- O mesmo número pode aparecer em mais de um prêmio do mesmo concurso:
  o Regulamento (item 11) diz que "É válida a repetição de número sorteado
  na mesma extração".
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from src.probability import DIGITOS_BILHETE, EXTRACOES

DATA_MINIMA = date(1962, 1, 1)  # a Loteria Federal começou a ser sorteada em 1962
COLUNAS_PREMIOS = [f"premio_{i}" for i in range(1, 6)]
COLUNAS_VALOR_NOMINAL = [f"valor_nominal_{i}" for i in range(1, 6)]
COLUNAS_OBRIGATORIAS = ["concurso", "data", *COLUNAS_PREMIOS]
COLUNAS_SAIDA = ["concurso", "data", "extracao", *COLUNAS_PREMIOS]
COLUNAS_VALIDOS = [*COLUNAS_SAIDA, "premio_zero", "dados_originais"]

# Nomes alternativos aceitos no cabeçalho (já sem acento e em minúsculas).
ALIASES_COLUNAS = {
    "concurso": "concurso",
    "extracao_numero": "concurso",
    "numero_concurso": "concurso",
    "n_concurso": "concurso",
    "data": "data",
    "data_sorteio": "data",
    "data_do_sorteio": "data",
    "extracao": "extracao",
    "tipo": "extracao",
    "tipo_extracao": "extracao",
    "modalidade": "extracao",
}
for _i in range(1, 6):
    for _nome in (
        f"premio_{_i}", f"premio{_i}", f"{_i}_premio", f"{_i}o_premio", f"{_i}_o_premio",
        f"bilhete_{_i}", f"bilhete{_i}",
    ):
        ALIASES_COLUNAS[_nome] = f"premio_{_i}"

# Textos aceitos para cada extração, após normalização.
ALIASES_EXTRACAO = {
    "quartou": "quartou",
    "quarta": "quartou",
    "quarta_feira": "quartou",
    "domingo": "domingo",
    "enricou": "enricou",
    "milionaria": "enricou",
    "milionaria_federal": "enricou",
    "natal": "natal",
    "especial_de_natal": "natal",
    "especial_natal": "natal",
}

EXTRACAO_NAO_INFORMADA = ""
MAX_TAMANHO_ARQUIVO = 20 * 1024 * 1024  # 20 MB

AJUSTE_APOSTROFO = "apostrofo_removido"
AJUSTE_SEIS_DIGITOS = "zero_inicial_extra_removido"
AJUSTE_ZEROS = "zeros_restaurados"
DESCRICAO_AJUSTES = {
    AJUSTE_APOSTROFO: "apóstrofo inicial removido (ex.: '032588)",
    AJUSTE_SEIS_DIGITOS: "6 dígitos começando com 0 convertidos para 5 (ex.: 032588 -> 32588)",
    AJUSTE_ZEROS: "zeros à esquerda restaurados (ex.: 1339 -> 01339)",
}


class ErroImportacao(ValueError):
    """Erro que impede a leitura do arquivo inteiro (por exemplo, cabeçalho inválido)."""


@dataclass
class ResultadoImportacao:
    validos: pd.DataFrame
    rejeitados: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    linhas_total: int = 0
    duplicados: int = 0
    lacunas: list[int] = field(default_factory=list)
    ajustes: dict[str, int] = field(default_factory=dict)
    metadados_arquivo: dict[str, str] = field(default_factory=dict)
    colunas_extras_por_linha: list[str] = field(default_factory=list)

    @property
    def linhas_validas(self) -> int:
        return len(self.validos)

    @property
    def linhas_rejeitadas(self) -> int:
        return len(self.rejeitados)

    @property
    def concursos_premio_zero(self) -> list[int]:
        if self.validos.empty:
            return []
        return sorted(self.validos.loc[self.validos["premio_zero"], "concurso"].astype(int).tolist())

    def relatorio(self) -> dict:
        """Resumo serializável, gravado junto com a importação."""
        return {
            "linhas_total": self.linhas_total,
            "linhas_validas": self.linhas_validas,
            "linhas_rejeitadas": self.linhas_rejeitadas,
            "duplicados": self.duplicados,
            "lacunas": self.lacunas,
            "ajustes": self.ajustes,
            "concursos_premio_zero": self.concursos_premio_zero,
            "metadados_arquivo": self.metadados_arquivo,
            "colunas_extras_por_linha": self.colunas_extras_por_linha,
            "avisos": self.avisos,
            "rejeitados": self.rejeitados.to_dict("records"),
        }


def _normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    texto = texto.strip().lower()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def _decodificar(conteudo: bytes) -> str:
    for codificacao in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return conteudo.decode(codificacao)
        except UnicodeDecodeError:
            continue
    raise ErroImportacao("Não foi possível ler o arquivo. Salve o CSV em UTF-8.")


def _detectar_separador(texto: str) -> str:
    primeira_linha = texto.splitlines()[0] if texto.strip() else ""
    try:
        return csv.Sniffer().sniff(primeira_linha, delimiters=";,\t").delimiter
    except csv.Error:
        return ";" if primeira_linha.count(";") >= primeira_linha.count(",") else ","


def ler_csv(conteudo: bytes | str) -> pd.DataFrame:
    """Lê o CSV com todas as colunas como texto. Mantém as colunas extras."""
    if isinstance(conteudo, bytes):
        if len(conteudo) > MAX_TAMANHO_ARQUIVO:
            raise ErroImportacao("Arquivo muito grande (limite de 20 MB).")
        texto = _decodificar(conteudo)
    else:
        texto = conteudo
    if not texto.strip():
        raise ErroImportacao("O arquivo está vazio.")

    separador = _detectar_separador(texto)
    try:
        df = pd.read_csv(
            io.StringIO(texto),
            sep=separador,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ErroImportacao(f"O arquivo não está em formato CSV válido: {exc}") from exc

    renomear = {}
    for coluna in df.columns:
        normalizada = _normalizar_texto(coluna)
        renomear[coluna] = ALIASES_COLUNAS.get(normalizada, normalizada)
    df = df.rename(columns=renomear)

    if df.columns.duplicated().any():
        repetidas = sorted(set(df.columns[df.columns.duplicated()]))
        raise ErroImportacao(f"Colunas repetidas no cabeçalho: {', '.join(repetidas)}.")

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ErroImportacao(
            "Colunas obrigatórias ausentes: " + ", ".join(faltando)
            + ". Cabeçalho esperado: " + ";".join(COLUNAS_SAIDA)
            + " (ou bilhete_1 a bilhete_5 no lugar de premio_1 a premio_5)."
        )
    if "extracao" not in df.columns:
        df["extracao"] = ""
    extras = [c for c in df.columns if c not in COLUNAS_SAIDA]
    return df[[*COLUNAS_SAIDA, *extras]].copy()


def normalizar_numero(valor: str) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Valida um número de bilhete.

    Retorna (numero_5_digitos, erro, ajustes_feitos).
    """
    texto = str(valor).strip()
    ajustes: list[str] = []
    if texto.startswith("'"):
        texto = texto[1:].strip()
        ajustes.append(AJUSTE_APOSTROFO)
    if texto == "":
        return None, "número vazio", ()
    if not texto.isdigit():
        return None, f"número com caracteres inválidos ({texto!r})", ()
    if len(texto) == DIGITOS_BILHETE + 1:
        if texto[0] != "0":
            return None, f"número com 6 dígitos que não começa com 0 ({texto!r})", ()
        texto = texto[1:]
        ajustes.append(AJUSTE_SEIS_DIGITOS)
    elif len(texto) > DIGITOS_BILHETE + 1:
        return None, f"número com mais de {DIGITOS_BILHETE} dígitos ({texto!r})", ()
    if len(texto) < DIGITOS_BILHETE:
        texto = texto.zfill(DIGITOS_BILHETE)
        ajustes.append(AJUSTE_ZEROS)
    if int(texto) == 0:
        return None, "00000 não é um número de bilhete válido (Regulamento, item 4)", ()
    return texto, None, tuple(ajustes)


def normalizar_data(valor: str, hoje: date | None = None) -> tuple[str | None, str | None]:
    """Converte a data para AAAA-MM-DD e confere se está entre 1962 e hoje."""
    texto = str(valor).strip()
    if not texto:
        return None, "data vazia"
    data_lida: date | None = None
    for formato in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            data_lida = datetime.strptime(texto, formato).date()
            break
        except ValueError:
            continue
    if data_lida is None:
        return None, f"data inválida ({texto!r}); use DD/MM/AAAA"
    hoje = hoje or date.today()
    if data_lida < DATA_MINIMA:
        return None, f"data anterior a {DATA_MINIMA.year}, início da Loteria Federal ({texto})"
    if data_lida > hoje:
        return None, f"data no futuro ({texto})"
    return data_lida.isoformat(), None


def normalizar_extracao(valor: str) -> tuple[str | None, str | None]:
    texto = _normalizar_texto(valor)
    if texto == "":
        return EXTRACAO_NAO_INFORMADA, None
    chave = ALIASES_EXTRACAO.get(texto)
    if chave is None:
        return None, f"extração desconhecida ({valor!r})"
    return chave, None


def _valor_e_zero(valor: str) -> bool:
    """True só quando o valor está preenchido e é zero (vazio não é zero)."""
    texto = re.sub(r"[R$\s]", "", str(valor))
    return bool(re.fullmatch(r"0+([.,]0+)?", texto))


def _separar_extras(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Divide as colunas extras entre constantes (metadados) e variáveis (por linha)."""
    constantes: dict[str, str] = {}
    variaveis: list[str] = []
    for coluna in df.columns:
        if coluna in COLUNAS_SAIDA:
            continue
        valores = df[coluna].unique()
        if len(df) > 1 and len(valores) == 1:
            constantes[coluna] = str(valores[0])
        else:
            variaveis.append(coluna)
    return constantes, variaveis


def _calcular_lacunas(concursos: set[int]) -> list[int]:
    if not concursos:
        return []
    return sorted(set(range(min(concursos), max(concursos) + 1)) - concursos)


def validar(
    df: pd.DataFrame,
    concursos_existentes: set[int] | None = None,
    hoje: date | None = None,
) -> ResultadoImportacao:
    """Valida cada linha. Os únicos ajustes são os de formato descritos no módulo."""
    concursos_existentes = concursos_existentes or set()
    metadados, extras_variaveis = _separar_extras(df)
    tem_valores = all(c in df.columns for c in COLUNAS_VALOR_NOMINAL)

    validos: list[dict] = []
    rejeitados: list[dict] = []
    avisos: list[str] = []
    vistos: dict[int, tuple] = {}
    duplicados = 0
    ajustes: Counter = Counter()

    for posicao, linha in enumerate(df.to_dict("records")):
        numero_linha = posicao + 2  # +1 do cabeçalho, +1 por começar em 1
        erros: list[str] = []
        ajustes_linha: Counter = Counter()

        concurso_txt = str(linha["concurso"]).strip()
        concurso = int(concurso_txt) if concurso_txt.isdigit() and int(concurso_txt) > 0 else None
        if concurso is None:
            erros.append(f"concurso inválido ({concurso_txt!r})")

        data_iso, erro = normalizar_data(linha["data"], hoje)
        if erro:
            erros.append(erro)

        extracao, erro = normalizar_extracao(linha["extracao"])
        if erro:
            erros.append(erro)

        premios: list[str | None] = []
        for coluna in COLUNAS_PREMIOS:
            numero, erro, feitos = normalizar_numero(linha[coluna])
            if erro:
                erros.append(f"{coluna}: {erro}")
            ajustes_linha.update(feitos)
            premios.append(numero)

        if not erros and extracao:
            maximo = EXTRACOES[extracao].numero_maximo
            acima = [n for n in premios if n and int(n) > maximo]
            if acima:
                avisos.append(
                    f"Linha {numero_linha}: {', '.join(acima)} acima do universo atual da "
                    f"extração {EXTRACOES[extracao].nome}. Planos antigos podem ter tido outra "
                    "quantidade de bilhetes; a linha foi mantida."
                )

        if not erros and concurso is not None:
            chave = (data_iso, extracao, *premios)
            if concurso in vistos:
                duplicados += 1
                if vistos[concurso] == chave:
                    erros.append("linha duplicada no arquivo")
                else:
                    erros.append("concurso repetido no arquivo com dados diferentes")
            elif concurso in concursos_existentes:
                duplicados += 1
                erros.append("concurso já existe no banco de dados")
            else:
                vistos[concurso] = chave

        if erros:
            rejeitados.append({
                "linha": numero_linha,
                "motivo": "; ".join(erros),
                **{c: str(linha[c]) for c in COLUNAS_SAIDA},
            })
            continue

        ajustes.update(ajustes_linha)
        premio_zero = tem_valores and all(_valor_e_zero(linha[c]) for c in COLUNAS_VALOR_NOMINAL)
        originais = {c: linha[c] for c in extras_variaveis}
        validos.append({
            "concurso": concurso,
            "data": data_iso,
            "extracao": extracao or "",
            **dict(zip(COLUNAS_PREMIOS, premios)),
            "premio_zero": bool(premio_zero),
            "dados_originais": json.dumps(originais, ensure_ascii=False) if originais else None,
        })

    for ajuste, quantidade in sorted(ajustes.items()):
        avisos.insert(0, f"{quantidade} número(s): {DESCRICAO_AJUSTES[ajuste]}.")

    validos_df = pd.DataFrame(validos, columns=COLUNAS_VALIDOS)
    if not validos_df.empty:
        validos_df["concurso"] = validos_df["concurso"].astype(int)
        validos_df["premio_zero"] = validos_df["premio_zero"].astype(bool)
    rejeitados_df = pd.DataFrame(rejeitados, columns=["linha", "motivo", *COLUNAS_SAIDA])

    todos = set(vistos) | concursos_existentes
    lacunas = [c for c in _calcular_lacunas(todos) if not vistos or min(vistos) <= c <= max(vistos)]

    return ResultadoImportacao(
        validos=validos_df,
        rejeitados=rejeitados_df,
        avisos=avisos,
        linhas_total=len(df),
        duplicados=duplicados,
        lacunas=lacunas,
        ajustes=dict(ajustes),
        metadados_arquivo=metadados,
        colunas_extras_por_linha=extras_variaveis,
    )


def importar(
    conteudo: bytes | str,
    concursos_existentes: set[int] | None = None,
    hoje: date | None = None,
) -> ResultadoImportacao:
    """Lê e valida um CSV. Não grava nada no banco."""
    return validar(ler_csv(conteudo), concursos_existentes, hoje)


def modelo_csv() -> str:
    """Cabeçalho de exemplo, sem nenhum resultado (não inventa dados)."""
    return ";".join(COLUNAS_SAIDA) + "\n"
