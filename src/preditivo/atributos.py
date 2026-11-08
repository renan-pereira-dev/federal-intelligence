"""Indicadores por milhar, calculados sem vazamento de informação.

O motor avalia os 4 últimos algarismos (a "milhar", posições 2 a 5). O 1º
algarismo (dezena de milhar) fica de fora porque depende do universo de cada
época (a quantidade de bilhetes mudou ao longo das décadas): um modelo que o
usasse aprenderia mudanças de regra, não comportamento do sorteio.

Regra central: o indicador de um concurso de índice t usa somente os concursos
de índice 0..t-1 (ordem cronológica pelo número do concurso). Nenhuma função
deste módulo olha para o concurso t nem para os seguintes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JANELAS = (50, 100, 500, 1000)
UNIVERSO_MILHAR = 10_000
NOMES_TERMINACAO = {1: "unidade", 2: "dezena", 3: "centena", 4: "milhar"}


class IndiceHistorico:
    """Histórico em ordem cronológica, indexado para consultas rápidas.

    Cada ocorrência vira uma chave inteira valor * B + índice_do_concurso, em um
    vetor ordenado. Assim, "quantas vezes o valor v saiu nos concursos [a, b)" é
    a diferença de duas buscas binárias, para milhares de valores de uma vez.
    """

    def __init__(self, historico: pd.DataFrame):
        if historico is None or historico.empty:
            raise ValueError("Histórico vazio.")
        dados = historico[["concurso", "data", "premio", "numero"]].copy()
        largura = dados.groupby("concurso")["premio"].count()
        incompletos = largura[largura != 5]
        if not incompletos.empty:
            raise ValueError(f"Concursos sem os cinco prêmios: {list(incompletos.index[:5])}")
        dados = dados.sort_values(["concurso", "premio"])
        self.concursos = dados["concurso"].drop_duplicates().to_numpy(dtype=np.int64)
        self.datas = dados.drop_duplicates("concurso")["data"].to_numpy(dtype=str)
        self.numeros = dados["numero"].astype(int).to_numpy(dtype=np.int64).reshape(-1, 5)
        self.total = len(self.concursos)
        self.base = self.total + 1
        indices = np.repeat(np.arange(self.total, dtype=np.int64), 5)
        planos = self.numeros.ravel()
        self._chaves: dict[str, np.ndarray] = {"exato": np.sort(planos * self.base + indices)}
        for d in (1, 2, 3, 4):
            self._chaves[f"t{d}"] = np.sort((planos % 10**d) * self.base + indices)
        for p in range(1, 5):  # posições 2..5 do bilhete (índice 1..4 da string)
            digito = (planos // 10 ** (4 - p)) % 10
            self._chaves[f"p{p}"] = np.sort(digito * self.base + indices)

    # --------------------------------------------------------------- consultas

    def contagem(self, tipo: str, valores, t, janela: int | None = None) -> np.ndarray:
        """Ocorrências de cada valor nos concursos [t - janela, t) (ou [0, t))."""
        valores = np.asarray(valores, dtype=np.int64)
        t = np.broadcast_to(np.asarray(t, dtype=np.int64), valores.shape)
        inicio = np.zeros_like(t) if janela is None else np.maximum(t - janela, 0)
        chaves = self._chaves[tipo]
        alto = np.searchsorted(chaves, valores * self.base + t, side="left")
        baixo = np.searchsorted(chaves, valores * self.base + inicio, side="left")
        return (alto - baixo).astype(np.float64)

    def atraso(self, tipo: str, valores, t) -> np.ndarray:
        """Concursos desde a última ocorrência antes de t; nunca saiu: t + 1."""
        valores = np.asarray(valores, dtype=np.int64)
        t = np.broadcast_to(np.asarray(t, dtype=np.int64), valores.shape)
        chaves = self._chaves[tipo]
        pos = np.searchsorted(chaves, valores * self.base + t, side="left") - 1
        seguro = np.clip(pos, 0, len(chaves) - 1)
        achou = (pos >= 0) & (chaves[seguro] // self.base == valores)
        ultimo = chaves[seguro] % self.base
        return np.where(achou, t - ultimo, t + 1).astype(np.float64)


def _razao(observado: np.ndarray, esperado: np.ndarray) -> np.ndarray:
    """Observado / esperado; sem dados (esperado 0) vale 1, o valor neutro."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(esperado > 0, observado / np.where(esperado > 0, esperado, 1), 1.0)


JANELA_EXTRA = 250


def calcular_atributos(indice: IndiceHistorico, milhares, t, extras: bool = False) -> pd.DataFrame:
    """Indicadores de cada milhar no momento t (usa só concursos < t).

    `milhares` e `t` podem ser vetores do mesmo tamanho (um t por linha).
    Frequências são razões observado/esperado: 1,0 é o que o acaso produziria.
    `extras=True` (usado só nos experimentos de auditoria) acrescenta a janela de
    250 concursos e indicadores de estabilidade; o motor padrão não os usa.
    """
    milhares = np.asarray(milhares, dtype=np.int64)
    t = np.broadcast_to(np.asarray(t, dtype=np.int64), milhares.shape)
    tf = t.astype(np.float64)
    col: dict[str, np.ndarray] = {}
    for d in (1, 2, 3, 4):
        v = milhares % 10**d
        por_concurso = 5 / 10**d  # ocorrências esperadas por concurso
        col[f"freq_{d}_total"] = _razao(indice.contagem(f"t{d}", v, t), por_concurso * tf)
        janelas = JANELAS if d < 4 else (500, 1000)
        for w in janelas:
            col[f"freq_{d}_{w}"] = _razao(indice.contagem(f"t{d}", v, t, w), por_concurso * np.minimum(w, tf))
        col[f"atraso_{d}"] = np.log1p(indice.atraso(f"t{d}", v, t) * por_concurso)
        if d < 4:
            col[f"tendencia_{d}"] = col[f"freq_{d}_100"] - col[f"freq_{d}_1000"]
        if d >= 2:
            col[f"repetiu_{d}"] = indice.contagem(f"t{d}", v, t, 1)
    if extras:
        for d in (1, 2, 3, 4):
            v = milhares % 10**d
            por_concurso = 5 / 10**d
            col[f"freq_{d}_{JANELA_EXTRA}"] = _razao(
                indice.contagem(f"t{d}", v, t, JANELA_EXTRA), por_concurso * np.minimum(JANELA_EXTRA, tf))
        for d in (1, 2, 3):
            # Estabilidade: desvio entre as razões das janelas (0 = frequência igual em todas).
            razoes = np.stack([col[f"freq_{d}_{w}"] for w in (50, 100, JANELA_EXTRA, 500, 1000)], axis=1)
            col[f"estabilidade_{d}"] = razoes.std(axis=1)
    # Posições 2 a 4. A 5ª posição (unidade) ficou de fora porque seria idêntica a
    # freq_1_500 e freq_1_total (auditoria de 08/11/2026).
    for p in range(1, 4):
        digito = (milhares // 10 ** (4 - p)) % 10
        col[f"posicao_{p + 1}_500"] = _razao(indice.contagem(f"p{p}", digito, t, 500), 0.5 * np.minimum(500, tf))
        col[f"posicao_{p + 1}_total"] = _razao(indice.contagem(f"p{p}", digito, t), 0.5 * tf)
    digitos = np.stack([(milhares // 10**k) % 10 for k in range(4)], axis=1)
    col["soma_algarismos"] = digitos.sum(axis=1).astype(np.float64)
    col["algarismos_pares"] = (digitos % 2 == 0).sum(axis=1).astype(np.float64)
    col["algarismos_distintos"] = np.array([len(set(linha)) for linha in digitos.tolist()], dtype=np.float64)
    return pd.DataFrame(col)


def colunas_atributos() -> list[str]:
    indice_vazio = IndiceHistorico(pd.DataFrame({
        "concurso": [1] * 5, "data": ["2000-01-01"] * 5, "premio": [1, 2, 3, 4, 5],
        "numero": ["00001", "00002", "00003", "00004", "00005"],
    }))
    return list(calcular_atributos(indice_vazio, [0], 1).columns)


DESCRICOES = {
    "freq": "frequência",
    "atraso": "atraso desde a última saída",
    "tendencia": "tendência (últimos 100 contra últimos 1000)",
    "repetiu": "saiu no concurso anterior",
    "posicao": "frequência do algarismo na posição",
}


def descrever_atributo(nome: str) -> str:
    """Nome legível de um indicador (ex.: freq_3_500 -> frequência da centena, últimos 500)."""
    partes = nome.split("_")
    if partes[0] == "freq":
        janela = "todo o histórico" if partes[2] == "total" else f"últimos {partes[2]} concursos"
        return f"Frequência da {NOMES_TERMINACAO[int(partes[1])]} ({janela})"
    if partes[0] == "atraso":
        return f"Atraso da {NOMES_TERMINACAO[int(partes[1])]}"
    if partes[0] == "tendencia":
        return f"Tendência da {NOMES_TERMINACAO[int(partes[1])]} (100 contra 1000)"
    if partes[0] == "estabilidade":
        return f"Estabilidade da frequência da {NOMES_TERMINACAO[int(partes[1])]} entre janelas"
    if partes[0] == "repetiu":
        return f"{NOMES_TERMINACAO[int(partes[1])].capitalize()} saiu no concurso anterior"
    if partes[0] == "posicao":
        janela = "todo o histórico" if partes[2] == "total" else "últimos 500"
        return f"Algarismo da {partes[1]}ª posição ({janela})"
    return {
        "soma_algarismos": "Soma dos algarismos da milhar",
        "algarismos_pares": "Quantidade de algarismos pares",
        "algarismos_distintos": "Quantidade de algarismos distintos",
    }.get(nome, nome)


def resumo_estatistico(indice: IndiceHistorico) -> dict:
    """Indicadores gerais do histórico inteiro (item 3), para exibição."""
    t = indice.total
    resumo: dict = {"concursos": int(t), "ultimo_concurso": int(indice.concursos[-1])}
    for d, nome in ((2, "dezena"), (3, "centena"), (4, "milhar")):
        valores = np.arange(10**d)
        contagem = {w: indice.contagem(f"t{d}", valores, t, w) for w in (None, *JANELAS)}
        tabela = pd.DataFrame({
            "valor": [f"{v:0{d}d}" for v in valores],
            "total": contagem[None].astype(int),
            **{f"ultimos_{w}": contagem[w].astype(int) for w in JANELAS},
            "atraso": indice.atraso(f"t{d}", valores, t).astype(int),
        })
        ordenada = tabela.sort_values(["total", "valor"], ascending=[False, True])
        resumo[nome] = {
            "esperado_total": 5 * t / 10**d,
            "mais_frequentes": ordenada.head(10).to_dict("records"),
            "menos_frequentes": ordenada.tail(10).iloc[::-1].to_dict("records"),
            "maiores_atrasos": tabela.sort_values("atraso", ascending=False).head(10).to_dict("records"),
        }
    resumo["algarismos_por_posicao"] = {
        str(p + 1): {
            "total": indice.contagem(f"p{p}", np.arange(10), t).astype(int).tolist(),
            "ultimos_500": indice.contagem(f"p{p}", np.arange(10), t, 500).astype(int).tolist(),
        }
        for p in range(1, 5)
    }
    numeros = indice.numeros.ravel()
    unicos, vezes = np.unique(numeros, return_counts=True)
    resumo["numeros_repetidos"] = int((vezes > 1).sum())
    resumo["numeros_mais_frequentes"] = [
        {"numero": f"{n:05d}", "vezes": int(c)}
        for n, c in sorted(zip(unicos.tolist(), vezes.tolist()), key=lambda x: (-x[1], x[0]))[:10]
    ]
    return resumo


def historico_do_numero(indice: IndiceHistorico, numero: str) -> dict:
    """Frequência histórica observada de um número e de suas terminações (bloco C)."""
    valor = int(numero)
    t = indice.total
    exato = indice.contagem("exato", [valor], t)[0]
    atraso_exato = indice.atraso("exato", [valor], t)[0]
    saida = {
        "numero": numero,
        "vezes_exato": int(exato),
        "ultima_vez_concurso": None if atraso_exato > t else int(indice.concursos[t - int(atraso_exato)]),
        "terminacoes": {},
    }
    for d in (2, 3, 4):
        v = valor % 10**d
        saida["terminacoes"][NOMES_TERMINACAO[d]] = {
            "valor": f"{v:0{d}d}",
            "vezes": int(indice.contagem(f"t{d}", [v], t)[0]),
            "esperado": round(5 * t / 10**d, 2),
            "ultimos_1000": int(indice.contagem(f"t{d}", [v], t, 1000)[0]),
            "esperado_1000": round(5 * min(1000, t) / 10**d, 2),
        }
    return saida
