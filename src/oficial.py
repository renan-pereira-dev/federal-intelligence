"""Importação do resultado oficial pelo serviço da CAIXA.

Fluxo em duas etapas, sempre acionado pelo usuário:
1. previsualizar(): consulta o serviço e compara com o banco. Não grava nada.
2. gravar(previa): grava só os concursos NOVOS, numa única transação, pelo
   mesmo caminho da importação de CSV (tabela imports + relatório). Depois
   confere automaticamente as previsões do motor.

Regras de segurança:
- Nada é sobrescrito. Concurso já gravado com os mesmos dados é ignorado; com
  dados diferentes é um CONFLITO, e nesse caso nada é gravado.
- Validação com as mesmas regras do importador de CSV: 5 números de 5
  dígitos (o serviço envia 6, com um zero na frente), 00000 recusado, data
  válida e não futura, número do concurso igual ao pedido.
- Sem internet, a aplicação continua funcionando; só esta consulta falha,
  com mensagem clara.

O serviço (consultado em 08/11/2026) devolve, entre outros campos:
numero, dataApuracao (DD/MM/AAAA), listaDezenas (5 números com 6 dígitos),
listaRateioPremio (valor de cada prêmio), localSorteio, observacao. Ele NÃO
informa o tipo de extração (QUARTOU, Domingo, ENRICOU): o campo fica vazio,
como no restante do histórico. Concurso inexistente devolve erro HTTP 500.
"""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from src import database as db
from src.importer import COLUNAS_VALIDOS, normalizar_data, normalizar_numero

URL_BASE = "https://servicebus2.caixa.gov.br/portaldeloterias/api/federal"
FONTE = "Serviço oficial da CAIXA (portaldeloterias/api/federal)"
MAXIMO_POR_CONSULTA = 20
TIMEOUT = 15
TENTATIVAS = 2

NOVO, IDENTICO, CONFLITO, INDISPONIVEL, INVALIDO = "novo", "identico", "conflito", "indisponivel", "invalido"
ROTULOS = {
    NOVO: "Novo (será gravado)",
    IDENTICO: "Já está no banco, igual",
    CONFLITO: "CONFLITO com o banco",
    INDISPONIVEL: "Não disponível no serviço",
    INVALIDO: "Resposta inválida",
}


class ErroServico(Exception):
    """O serviço não pôde ser consultado (sem internet, fora do ar, resposta não JSON)."""


class ConcursoIndisponivel(ErroServico):
    """O serviço não tem este concurso (ainda não sorteado ou fora do ar para ele)."""


class ErroValidacao(ValueError):
    """A resposta do serviço não passou nas regras de validação."""


@dataclass
class ResultadoOficial:
    concurso: int
    data: str
    premios: list[str]
    premio_zero: bool
    dados_originais: dict
    ajustes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ consulta

def _abrir(url: str, timeout: float):
    requisicao = urllib.request.Request(
        url, headers={"User-Agent": "FederalIntelligence/0.1", "Accept": "application/json"})
    return urllib.request.urlopen(requisicao, timeout=timeout)


def buscar_json(concurso: int | None = None, timeout: float = TIMEOUT, tentativas: int = TENTATIVAS,
                abrir=None) -> dict:
    """JSON do concurso pedido (ou do último, se None). Erros viram exceções claras."""
    abrir = abrir or _abrir
    url = URL_BASE if concurso is None else f"{URL_BASE}/{int(concurso)}"
    ultimo_erro: Exception | None = None
    for _ in range(max(1, tentativas)):
        try:
            with abrir(url, timeout) as resposta:
                bruto = resposta.read()
            try:
                return json.loads(bruto.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ErroServico("O serviço respondeu, mas não em JSON válido.") from exc
        except urllib.error.HTTPError as exc:
            if concurso is not None and exc.code in (404, 500):
                raise ConcursoIndisponivel(f"Concurso {concurso} não disponível no serviço (HTTP {exc.code}).") from exc
            ultimo_erro = exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            ultimo_erro = exc
    raise ErroServico(f"Não foi possível consultar o serviço oficial ({type(ultimo_erro).__name__}). "
                      "Verifique a conexão com a internet.") from ultimo_erro


# ------------------------------------------------------------------ validação

def _valor(item: dict) -> float | None:
    try:
        return float(item.get("valorPremio"))
    except (TypeError, ValueError):
        return None


def interpretar(dados: dict, concurso_esperado: int | None = None, hoje: date | None = None) -> ResultadoOficial:
    """Valida a resposta e converte para o formato do banco. Erro claro se algo não fecha."""
    if not isinstance(dados, dict):
        raise ErroValidacao("Resposta em formato inesperado.")
    numero = dados.get("numero")
    if not isinstance(numero, int) or numero <= 0:
        raise ErroValidacao(f"Número do concurso ausente ou inválido ({numero!r}).")
    if concurso_esperado is not None and numero != concurso_esperado:
        raise ErroValidacao(f"Foi pedido o concurso {concurso_esperado}, mas veio o {numero}.")
    data_iso, erro = normalizar_data(dados.get("dataApuracao") or "", hoje)
    if erro:
        raise ErroValidacao(f"Concurso {numero}: {erro}.")
    lista = dados.get("listaDezenas")
    if not isinstance(lista, list) or len(lista) != 5:
        raise ErroValidacao(f"Concurso {numero}: eram esperados 5 números, vieram {lista!r}.")
    premios, ajustes = [], []
    for i, valor in enumerate(lista, 1):
        texto, erro, aj = normalizar_numero(str(valor))
        if erro:
            raise ErroValidacao(f"Concurso {numero}, {i}º prêmio: {erro}.")
        premios.append(texto)
        ajustes.extend(aj)
    ordem = dados.get("dezenasSorteadasOrdemSorteio")
    if isinstance(ordem, list) and ordem and list(ordem) != list(lista):
        raise ErroValidacao(f"Concurso {numero}: as duas listas de números do serviço não coincidem.")

    rateio = dados.get("listaRateioPremio") or []
    valores = {}
    for item in rateio:
        faixa = item.get("faixa")
        if isinstance(faixa, int) and 1 <= faixa <= 5:
            valores[faixa] = _valor(item)
    # Mesma regra do importador: prêmio zero só quando os 5 valores estão informados e são 0.
    premio_zero = len(valores) == 5 and all(v == 0 for v in valores.values())
    originais = {
        **{f"valor_nominal_{f}": (None if v is None else str(int(v)) if float(v).is_integer() else str(v))
           for f, v in sorted(valores.items())},
        "fonte": FONTE,
        "consultado_em": datetime.now().isoformat(timespec="seconds"),
        "local_sorteio": dados.get("localSorteio"),
        "observacao": dados.get("observacao"),
        "lista_original": list(lista),
    }
    return ResultadoOficial(numero, data_iso, premios, premio_zero, originais, sorted(set(ajustes)))


# ------------------------------------------------------------------ prévia

def _ultimo_local(caminho) -> int:
    with db.conectar(caminho) as conexao:
        linha = conexao.execute("SELECT max(concurso) FROM contests").fetchone()
    return int(linha[0] or 0)


def _situacao(resultado: ResultadoOficial, caminho) -> tuple[str, str]:
    existente = db.buscar_concurso(resultado.concurso, caminho)
    if existente is None:
        return NOVO, ""
    if existente["premios"] == resultado.premios and existente["data"] == resultado.data:
        return IDENTICO, ""
    return CONFLITO, (f"No banco: {existente['data']} {', '.join(existente['premios'])}. "
                      f"No serviço: {resultado.data} {', '.join(resultado.premios)}.")


def previsualizar(caminho=None, maximo: int = MAXIMO_POR_CONSULTA, buscar=None, hoje: date | None = None) -> dict:
    """Consulta o serviço e compara com o banco. NÃO grava nada.

    Confere o último concurso do banco e busca, em ordem, os seguintes até o
    último publicado (no máximo `maximo` por vez).
    """
    buscar = buscar or buscar_json
    caminho = db.resolver_caminho(caminho)
    db.inicializar(caminho)
    previa = {"consultado_em": datetime.now().isoformat(timespec="seconds"), "itens": [], "erro": None,
              "ultimo_local": _ultimo_local(caminho), "ultimo_oficial": None, "proximo_oficial": None,
              "restantes": 0}
    try:
        ultimo = buscar(None)
    except ErroServico as exc:
        previa["erro"] = str(exc)
        return previa
    previa["ultimo_oficial"] = ultimo.get("numero") if isinstance(ultimo.get("numero"), int) else None
    previa["proximo_oficial"] = ultimo.get("numeroConcursoProximo")
    if previa["ultimo_oficial"] is None:
        previa["erro"] = "O serviço não informou o número do último concurso."
        return previa

    local, oficial = previa["ultimo_local"], previa["ultimo_oficial"]
    # Banco vazio: só o último publicado (o histórico completo vem pelo CSV).
    inicio = local if local else oficial
    fim = min(oficial, inicio + maximo)
    previa["restantes"] = max(0, oficial - fim)
    for n in range(inicio, fim + 1):
        item = {"concurso": n, "situacao": None, "mensagem": "", "resultado": None}
        try:
            dados = ultimo if n == oficial else buscar(n)
            resultado = interpretar(dados, concurso_esperado=n, hoje=hoje)
            item["situacao"], item["mensagem"] = _situacao(resultado, caminho)
            item["resultado"] = resultado
        except ConcursoIndisponivel as exc:
            item["situacao"], item["mensagem"] = INDISPONIVEL, str(exc)
        except ErroValidacao as exc:
            item["situacao"], item["mensagem"] = INVALIDO, str(exc)
        except ErroServico as exc:
            previa["erro"] = str(exc)
            break
        previa["itens"].append(item)
    return previa


def resumo_previa(previa: dict) -> dict:
    contagem = {s: 0 for s in ROTULOS}
    for item in previa["itens"]:
        contagem[item["situacao"]] += 1
    pode_gravar = previa["erro"] is None and contagem[NOVO] > 0 and contagem[CONFLITO] == 0
    return {"contagem": contagem, "pode_gravar": pode_gravar}


# ------------------------------------------------------------------ gravação

def gravar(previa: dict, caminho=None) -> dict:
    """Grava os concursos novos da prévia (uma transação) e confere as previsões.

    Recusa se houver conflito ou erro. Antes de gravar, confere de novo cada
    concurso contra o banco (outra sessão pode ter gravado no meio tempo).
    """
    from src.preditivo import armazenamento

    caminho = db.resolver_caminho(caminho)
    if previa.get("erro"):
        raise ErroValidacao(f"A consulta teve erro; nada foi gravado: {previa['erro']}")
    if any(i["situacao"] == CONFLITO for i in previa["itens"]):
        raise ErroValidacao("Há conflito entre o serviço e o banco; nada foi gravado.")
    novos = [i["resultado"] for i in previa["itens"] if i["situacao"] == NOVO]
    if not novos:
        return {"gravados": [], "import_id": None, "previsoes_conferidas": []}
    for r in novos:
        situacao, mensagem = _situacao(r, caminho)
        if situacao == CONFLITO:
            raise ErroValidacao(f"Concurso {r.concurso} mudou no banco desde a prévia: {mensagem} Nada foi gravado.")
    novos = [r for r in novos if _situacao(r, caminho)[0] == NOVO]
    if not novos:
        return {"gravados": [], "import_id": None, "previsoes_conferidas": []}

    validos = pd.DataFrame([{
        "concurso": r.concurso, "data": r.data, "extracao": "",
        **{f"premio_{i}": n for i, n in enumerate(r.premios, 1)},
        "premio_zero": r.premio_zero,
        "dados_originais": json.dumps(r.dados_originais, ensure_ascii=False),
    } for r in novos], columns=COLUNAS_VALIDOS)
    relatorio = {
        "origem": FONTE,
        "url": URL_BASE,
        "consultado_em": previa["consultado_em"],
        "concursos": [r.concurso for r in novos],
        "ignorados_identicos": [i["concurso"] for i in previa["itens"] if i["situacao"] == IDENTICO],
        "indisponiveis": [i["concurso"] for i in previa["itens"] if i["situacao"] == INDISPONIVEL],
        "invalidos": [{"concurso": i["concurso"], "motivo": i["mensagem"]}
                      for i in previa["itens"] if i["situacao"] == INVALIDO],
        "ajustes": sorted({a for r in novos for a in r.ajustes}),
        "impressao": hashlib.sha256(json.dumps([[r.concurso, r.data, r.premios] for r in novos]).encode()).hexdigest()[:16],
    }
    contagem = resumo_previa(previa)["contagem"]
    import_id = db.registrar_importacao(
        validos, FONTE, linhas_total=len(previa["itens"]),
        linhas_rejeitadas=contagem[INVALIDO], linhas_duplicadas=contagem[IDENTICO],
        relatorio=relatorio, caminho=caminho,
    )
    armazenamento.inicializar(caminho)
    conferidas = armazenamento.conferir_pendentes(caminho)
    return {"gravados": [r.concurso for r in novos], "import_id": import_id, "previsoes_conferidas": conferidas}
