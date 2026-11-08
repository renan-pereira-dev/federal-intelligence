"""Monitoramento da justiça do sorteio ao longo do tempo.

Pergunta: os números sorteados continuam se comportando como um sorteio justo
(uniforme e independente)? Este módulo NÃO prevê nada; ele procura defeitos
persistentes no sorteio, a única fonte plausível de alguma vantagem real.

Desenho (fixado antes de olhar os resultados):
- Blocos independentes de 250 concursos consecutivos (1.250 números), contados
  a partir do mais recente. O bloco mais antigo incompleto fica de fora.
- Seis testes por bloco:
  1. unidade: o último algarismo é uniforme (qui-quadrado, 9 g.l.);
  2. dezena: os dois últimos algarismos são uniformes (qui-quadrado, 99 g.l.);
  3. algarismo da 3ª posição uniforme (qui-quadrado, 9 g.l.);
  4. algarismo da 4ª posição uniforme (qui-quadrado, 9 g.l.);
  5. homogeneidade entre prêmios: a unidade tem a mesma distribuição do 1º ao
     5º prêmio (tabela 10 × 5, qui-quadrado, 36 g.l.);
  6. independência: a unidade de cada prêmio repete a do mesmo prêmio no
     concurso anterior em 10% dos casos (binomial bilateral).
- As posições 1 e 2 ficam de fora: dependem da quantidade de bilhetes de cada
  época, e um desvio ali seria mudança de regra, não defeito do sorteio. Nas
  posições 3 a 5 a distribuição é uniforme para qualquer universo múltiplo de
  1.000 (92.000, 96.000, 84.000, 85.000) e quase uniforme nos demais.
- Correção de Holm entre os 6 testes de cada bloco.

Situação:
- normal: nenhum teste significativo após Holm no bloco mais recente;
- atenção: algum teste com p de Holm < 0,05 no bloco mais recente, sem
  repetição (num sorteio justo isso acontece em cerca de 5% dos blocos);
- alerta: o MESMO teste tem p < 0,05 em dois blocos consecutivos e
  independentes, e em pelo menos um deles p de Holm < 0,05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.statistics import p_valor_qui_quadrado

BLOCO = 250
ALFA = 0.05
TESTES = {
    "unidade": "Unidade (último algarismo) uniforme",
    "dezena": "Dezena (dois últimos algarismos) uniforme",
    "posicao_3": "Algarismo da 3ª posição uniforme",
    "posicao_4": "Algarismo da 4ª posição uniforme",
    "homogeneidade_premios": "Mesma unidade do 1º ao 5º prêmio",
    "independencia": "Independência entre concursos seguidos",
}
NORMAL, ATENCAO, ALERTA = "normal", "atencao", "alerta"


def matriz(historico: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(concursos, datas, números T×5) em ordem de concurso; só concursos completos."""
    if historico is None or historico.empty:
        return np.array([], dtype=int), np.array([], dtype=str), np.zeros((0, 5), dtype=int)
    dados = historico[["concurso", "data", "premio", "numero"]].sort_values(["concurso", "premio"])
    completos = dados.groupby("concurso")["premio"].transform("count") == 5
    dados = dados[completos]
    concursos = dados["concurso"].drop_duplicates().to_numpy(dtype=int)
    datas = dados.drop_duplicates("concurso")["data"].to_numpy(dtype=str)
    numeros = dados["numero"].astype(int).to_numpy().reshape(-1, 5)
    return concursos, datas, numeros


def _qui_uniforme(valores: np.ndarray, categorias: int) -> dict:
    observados = np.bincount(valores, minlength=categorias)
    esperado = len(valores) / categorias
    estat = float(((observados - esperado) ** 2 / esperado).sum())
    gl = categorias - 1
    residuos = (observados - esperado) / np.sqrt(esperado)
    maior = int(np.argmax(np.abs(residuos)))
    return {"estatistica": estat, "gl": gl, "p": p_valor_qui_quadrado(estat, gl),
            "observados": observados.tolist(), "esperado_por_categoria": esperado,
            "maior_desvio": {"categoria": maior, "observado": int(observados[maior]),
                             "residuo": float(residuos[maior])}}


def testar_bloco(numeros: np.ndarray, anterior: np.ndarray | None = None) -> dict:
    """Os 6 testes para um bloco (n×5). `anterior` é o concurso logo antes do bloco, se houver."""
    unidades = numeros % 10
    resultado = {
        "unidade": _qui_uniforme(unidades.ravel(), 10),
        "dezena": _qui_uniforme((numeros % 100).ravel(), 100),
        "posicao_3": _qui_uniforme(((numeros // 100) % 10).ravel(), 10),
        "posicao_4": _qui_uniforme(((numeros // 10) % 10).ravel(), 10),
    }
    tabela = np.stack([np.bincount(unidades[:, p], minlength=10) for p in range(5)], axis=1)
    estat, _, gl, _ = stats.chi2_contingency(tabela, correction=False)
    resultado["homogeneidade_premios"] = {"estatistica": float(estat), "gl": int(gl),
                                          "p": p_valor_qui_quadrado(float(estat), int(gl))}
    # Pares (concurso, concurso seguinte) dentro do bloco; o concurso anterior ao bloco
    # entra só como "passado" do primeiro, sem sobrepor blocos de teste.
    sequencia = unidades if anterior is None else np.vstack([anterior % 10, unidades])
    iguais = int((sequencia[1:] == sequencia[:-1]).sum())
    pares = int(sequencia[1:].size)
    teste = stats.binomtest(iguais, pares, 0.1)
    resultado["independencia"] = {"estatistica": iguais / pares, "gl": None, "p": float(teste.pvalue),
                                  "repeticoes": iguais, "pares": pares, "esperado": 0.1 * pares}
    ajustados = holm({k: v["p"] for k, v in resultado.items()})
    for k, v in resultado.items():
        v["p_holm"] = ajustados[k]
    return resultado


def holm(pvalores: dict[str, float]) -> dict[str, float]:
    ordem = sorted(pvalores, key=pvalores.get)
    ajustados, maximo, m = {}, 0.0, len(ordem)
    for i, chave in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - i) * pvalores[chave]))
        ajustados[chave] = maximo
    return ajustados


def classificar(blocos: list[dict]) -> dict:
    """Aplica as regras de atenção e alerta. `blocos` em ordem cronológica, cada um com 'testes'."""
    alertas = []
    for k in range(1, len(blocos)):
        for teste in TESTES:
            atual, anterior = blocos[k]["testes"][teste], blocos[k - 1]["testes"][teste]
            if (atual["p"] < ALFA and anterior["p"] < ALFA
                    and min(atual["p_holm"], anterior["p_holm"]) < ALFA):
                alertas.append({"teste": teste, "blocos": [k - 1, k],
                                "de": blocos[k - 1]["de"], "ate": blocos[k]["ate"],
                                "p": [anterior["p"], atual["p"]], "p_holm": [anterior["p_holm"], atual["p_holm"]]})
    if not blocos:
        return {"situacao": None, "alertas": [], "atencao": []}
    ultimo = len(blocos) - 1
    atencao = [t for t in TESTES if blocos[ultimo]["testes"][t]["p_holm"] < ALFA]
    if any(a["blocos"][1] == ultimo for a in alertas):
        situacao = ALERTA
    elif atencao:
        situacao = ATENCAO
    else:
        situacao = NORMAL
    return {"situacao": situacao, "alertas": alertas, "atencao": atencao}


def monitorar(historico: pd.DataFrame, bloco: int = BLOCO) -> dict:
    """Divide o histórico em blocos, testa cada um e classifica a situação atual."""
    concursos, datas, numeros = matriz(historico)
    total = len(concursos)
    quantidade = total // bloco
    saida = {"bloco": bloco, "concursos": total, "blocos": [], "descartados": total - quantidade * bloco}
    if quantidade == 0:
        saida.update(situacao=None, alertas=[], atencao=[], resumo=None)
        return saida
    inicio = total - quantidade * bloco  # blocos alinhados ao concurso mais recente
    for b in range(quantidade):
        a, z = inicio + b * bloco, inicio + (b + 1) * bloco
        anterior = numeros[a - 1] if a > 0 else None
        saida["blocos"].append({
            "indice": b, "de": int(concursos[a]), "ate": int(concursos[z - 1]),
            "data_de": str(datas[a]), "data_ate": str(datas[z - 1]),
            "testes": testar_bloco(numeros[a:z], anterior),
        })
    saida.update(classificar(saida["blocos"]))
    p_todos = [t["p"] for b in saida["blocos"] for t in b["testes"].values()]
    saida["resumo"] = {
        "testes": len(p_todos),
        "p_menor_005": int(sum(p < ALFA for p in p_todos)),
        "esperado_p_menor_005": ALFA * len(p_todos),
        "holm_menor_005": int(sum(t["p_holm"] < ALFA for b in saida["blocos"] for t in b["testes"].values())),
    }
    return saida
