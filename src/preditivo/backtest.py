"""Backtest walk-forward, em ordem cronológica e sem vazamento.

Para cada bloco de concursos de teste, os modelos são treinados apenas com
concursos ANTERIORES ao início do bloco. Cada concurso do bloco é avaliado com
indicadores calculados só com concursos anteriores a ele. Os 10 candidatos de
cada concurso são comparados com os 5 números sorteados e com uma seleção
aleatória equivalente.

Teste principal (definido antes de rodar, para não escolher o resultado):
a "posição relativa" de cada milhar sorteada no ranking do conjunto. Sem
vantagem, ela se distribui por igual entre 0 e 1 (média 0,5, desvio 1/sqrt(12)).
O conjunto só é considerado validado se, ao mesmo tempo:
1. a média ficar abaixo de 0,5 com p < 0,05 (teste z unilateral); e
2. for melhor que a referência "estrutura do universo" com p < 0,05 (teste
   pareado). Motivo: quando a quantidade de bilhetes não é múltipla de 10.000
   (ex.: 96.000), algumas milhares têm 10 bilhetes válidos e outras 9. Um modelo
   que só aprenda isso fica abaixo de 0,5 sem prever nada: o número inteiro
   continua com chance de 1 em N.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preditivo.atributos import UNIVERSO_MILHAR, IndiceHistorico
from src.preditivo.modelos import Configuracao, ranking_em_lote, treinar

LOTE = 20  # concursos avaliados por chamada aos modelos

ALFA = 0.05
QUANTIDADE = 10
JANELA_LIMITE = 500  # concursos usados para estimar o maior número válido no backtest


def limite_universo(indice: IndiceHistorico, t: int) -> int:
    """Maior número sorteado nos concursos anteriores a t (estimativa do universo da época)."""
    inicio = max(0, t - JANELA_LIMITE)
    return int(indice.numeros[inicio:t].max())


def bilhetes_validos(limite: int) -> np.ndarray:
    """Quantos números de 00001 a `limite` terminam em cada milhar (estrutura do universo)."""
    milhares = np.arange(UNIVERSO_MILHAR)
    numeros = np.arange(10)[:, None] * UNIVERSO_MILHAR + milhares[None, :]
    return ((numeros >= 1) & (numeros <= limite)).sum(axis=0).astype(float)


def posicoes_no_ranking(score: np.ndarray, milhares: np.ndarray) -> np.ndarray:
    """Posição relativa (0 a 1) de cada milhar no ranking `score` (maior = melhor); empates no meio."""
    alvo = score[milhares]
    maiores = (score[None, :] > alvo[:, None]).sum(axis=1)
    iguais = (score[None, :] == alvo[:, None]).sum(axis=1)
    return (maiores + 0.5 * iguais) / len(score)


def teste_pareado(u_a, u_b) -> dict:
    """Diferença média A - B nas mesmas milhares (negativo = A melhor). Bilateral."""
    d = np.asarray(u_a, dtype=float) - np.asarray(u_b, dtype=float)
    n = len(d)
    media = float(d.mean()) if n else 0.0
    erro = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    if erro == 0:
        return {"n": n, "diferenca": media, "ic95": [media, media], "z": 0.0, "p": 1.0}
    z = media / erro
    return {"n": n, "diferenca": media, "ic95": [media - 1.96 * erro, media + 1.96 * erro],
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def escolher_numero(milhar: int, limite: int, escolher: Callable[[list[int]], int]) -> str | None:
    """Completa a milhar com um 1º algarismo válido, escolhido ao acaso (o modelo não o avalia)."""
    primeiros = [p for p in range(10) if 1 <= p * UNIVERSO_MILHAR + milhar <= limite]
    if not primeiros:
        return None
    return f"{escolher(primeiros) * UNIVERSO_MILHAR + milhar:05d}"


def selecionar(tabela: pd.DataFrame, limite: int, escolher: Callable[[list[int]], int],
               quantidade: int = QUANTIDADE) -> list[tuple[str, int]]:
    """Os `quantidade` melhores do ranking, com milhares distintas e dentro do universo."""
    escolhidos, usadas = [], set()
    for posicao, milhar in enumerate(tabela["milhar"].to_numpy()):
        if int(milhar) in usadas:
            continue
        usadas.add(int(milhar))
        numero = escolher_numero(int(milhar), limite, escolher)
        if numero is not None:
            escolhidos.append((numero, posicao))
        if len(escolhidos) == quantidade:
            break
    return escolhidos


def _coincidencias(candidatos: list[str], sorteados: list[str]) -> dict:
    inteiros = {int(c) for c in candidatos}
    saida = {
        "exato": sum(int(s) in inteiros for s in sorteados),
        "principal": int(int(candidatos[0]) in {int(s) for s in sorteados}),
    }
    for d, nome in ((4, "milhar"), (3, "centena"), (2, "dezena")):
        finais = {c[-d:] for c in candidatos}
        saida[nome] = sum(s[-d:] in finais for s in sorteados)
        saida[f"{nome}_esperado"] = len(sorteados) * len(finais) / 10**d
    saida["exato_esperado"] = None  # preenchido com o limite do universo
    return saida


def _teste_posicao(valores: np.ndarray) -> dict:
    n = len(valores)
    if n == 0:
        return {"n": 0, "media": None, "z": None, "p": None}
    media = float(np.mean(valores))
    z = (media - 0.5) / (math.sqrt(1 / 12) / math.sqrt(n))
    return {"n": int(n), "media": media, "z": z, "p": float(stats.norm.cdf(z))}


def _teste_contagem(observado: float, esperado: float) -> float | None:
    """P(contagem >= observado) sob o acaso (aproximação de Poisson)."""
    if esperado <= 0:
        return None
    return float(stats.poisson.sf(observado - 1, esperado))


def executar(indice: IndiceHistorico, config: Configuracao,
             progresso: Callable[[float, str], None] | None = None) -> dict:
    """Roda o backtest e devolve as métricas. Erro claro se o histórico for curto."""
    inicio_relogio = time.perf_counter()
    total = indice.total
    teste = min(config.concursos_teste, total - config.minimo_treino - 1)
    if teste < max(20, config.bloco // 2):
        raise ValueError(
            f"Histórico insuficiente para o backtest: {total} concursos. São necessários pelo "
            f"menos {config.minimo_treino + max(20, config.bloco // 2) + 1}."
        )
    primeiro = total - teste
    rng = np.random.default_rng(config.semente + 1)
    escolher_rng = lambda opcoes: int(rng.choice(opcoes))  # noqa: E731

    posicoes: dict[str, list[float]] = {}
    linhas = []
    blocos = list(range(primeiro, total, config.bloco))
    for b, inicio_bloco in enumerate(blocos):
        fim_bloco = min(inicio_bloco + config.bloco, total)
        if progresso:
            progresso(b / len(blocos), f"Backtest: bloco {b + 1} de {len(blocos)} "
                                       f"(concursos {indice.concursos[inicio_bloco]}–{indice.concursos[fim_bloco - 1]})")
        pacote = treinar(indice, inicio_bloco, config)  # só concursos < inicio_bloco
        ts = list(range(inicio_bloco, fim_bloco))
        tabelas = []
        for k in range(0, len(ts), LOTE):
            tabelas += ranking_em_lote(pacote, indice, ts[k:k + LOTE])  # cada t usa só concursos < t
        for t, tabela in zip(ts, tabelas):
            sorteados = [f"{n:05d}" for n in indice.numeros[t]]
            milhares_sorteadas = indice.numeros[t] % UNIVERSO_MILHAR
            por_milhar = tabela.set_index("milhar")
            n = len(tabela)
            for chave in pacote["modelos"]:
                pct = por_milhar.loc[milhares_sorteadas, chave].to_numpy()
                posicoes.setdefault(chave, []).extend((1 - pct + 0.5 / n).tolist())
            pct = por_milhar.loc[milhares_sorteadas, "pontuacao"].to_numpy() / 100
            posicoes.setdefault("conjunto", []).extend((1 - pct + 0.5 / n).tolist())

            limite = limite_universo(indice, t)
            posicoes.setdefault("estrutura", []).extend(
                posicoes_no_ranking(bilhetes_validos(limite), milhares_sorteadas).tolist())
            candidatos = [c for c, _ in selecionar(tabela, limite, escolher_rng)]
            aleatorios = [f"{v:05d}" for v in rng.choice(np.arange(1, limite + 1), QUANTIDADE, replace=False)]
            modelo = _coincidencias(candidatos, sorteados)
            acaso = _coincidencias(aleatorios, sorteados)
            linha = {"concurso": int(indice.concursos[t]), "bloco": b + 1, "limite": limite}
            linha.update({f"modelo_{k}": v for k, v in modelo.items() if v is not None})
            linha.update({f"acaso_{k}": v for k, v in acaso.items() if v is not None})
            linha["exato_esperado"] = len(sorteados) * QUANTIDADE / limite
            linhas.append(linha)
    if progresso:
        progresso(1.0, "Backtest concluído")

    detalhe = pd.DataFrame(linhas)
    testes = {chave: _teste_posicao(np.array(v)) for chave, v in posicoes.items()}
    conjunto = testes["conjunto"]
    contra_estrutura = teste_pareado(posicoes["conjunto"], posicoes["estrutura"])
    abaixo_do_acaso = bool(conjunto["p"] is not None and conjunto["media"] < 0.5 and conjunto["p"] < ALFA)
    melhor_que_estrutura = bool(contra_estrutura["diferenca"] < 0 and contra_estrutura["p"] < ALFA)
    validado = abaixo_do_acaso and melhor_que_estrutura

    resumo_coinc = {}
    for nome in ("exato", "milhar", "centena", "dezena"):
        esperado = float(detalhe["exato_esperado"].sum() if nome == "exato" else detalhe[f"modelo_{nome}_esperado"].sum())
        observado = float(detalhe[f"modelo_{nome}"].sum())
        resumo_coinc[nome] = {
            "modelo": observado,
            "acaso_simulado": float(detalhe[f"acaso_{nome}"].sum()),
            "esperado_teorico": esperado,
            "esperado_acaso": float(detalhe["exato_esperado"].sum() if nome == "exato"
                                    else detalhe[f"acaso_{nome}_esperado"].sum()),
            "p_mais_que_acaso": _teste_contagem(observado, esperado),
        }
    concursos = len(detalhe)
    acertos = detalhe["modelo_exato"].sum()
    metricas = {
        "concursos_testados": int(concursos),
        "primeiro_concurso": int(detalhe["concurso"].min()),
        "ultimo_concurso": int(detalhe["concurso"].max()),
        "blocos": len(blocos),
        "acertos_principal": int(detalhe["modelo_principal"].sum()),
        "acertos_principal_acaso": int(detalhe["acaso_principal"].sum()),
        "acertos_principal_esperado": float((5 / detalhe["limite"]).sum()),
        "acertos_top10": int(acertos),
        "acertos_top10_acaso": int(detalhe["acaso_exato"].sum()),
        "precision_10": float(acertos / (QUANTIDADE * concursos)),
        "recall_10": float(acertos / (5 * concursos)),
        "precision_10_acaso": float(detalhe["acaso_exato"].sum() / (QUANTIDADE * concursos)),
        "recall_10_acaso": float(detalhe["acaso_exato"].sum() / (5 * concursos)),
        "coincidencias": resumo_coinc,
        "testes_posicao": testes,
        "validado": validado,
        "criterios": {"abaixo_do_acaso": abaixo_do_acaso, "melhor_que_estrutura": melhor_que_estrutura},
        "conjunto_contra_estrutura": contra_estrutura,
        "alfa": ALFA,
    }
    periodos = detalhe.groupby("bloco").agg(
        primeiro=("concurso", "min"), ultimo=("concurso", "max"), concursos=("concurso", "count"),
        milhar_modelo=("modelo_milhar", "sum"), milhar_esperado=("modelo_milhar_esperado", "sum"),
        centena_modelo=("modelo_centena", "sum"), centena_esperado=("modelo_centena_esperado", "sum"),
        exato_modelo=("modelo_exato", "sum"), exato_acaso=("acaso_exato", "sum"),
    ).reset_index()
    pos = np.array(posicoes["conjunto"]).reshape(-1, 5).mean(axis=1)
    periodos["posicao_media"] = pd.Series(pos).groupby(detalhe["bloco"].to_numpy()).mean().to_numpy()
    metricas["por_periodo"] = periodos.to_dict("records")
    metricas["duracao_segundos"] = round(time.perf_counter() - inicio_relogio, 1)
    return metricas
