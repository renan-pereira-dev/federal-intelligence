"""Auditoria técnica e experimentos controlados do motor (não altera o motor padrão).

Divisão cronológica, fixada ANTES de rodar:

    [ ........ treino ........ | ajuste (500) | teste final (1000) ]

- Treino: em cada bloco, os modelos usam só concursos anteriores ao bloco.
- Ajuste: onde as configurações (janelas, atributos, parâmetros, combinações
  e pesos do conjunto) são comparadas e a melhor é escolhida.
- Teste final: só é usado DEPOIS da escolha, para medir fora da amostra a
  configuração escolhida, o padrão atual, cada modelo e os modelos de
  referência (acaso, frequência histórica, frequência recente, ausência).

Métrica: posição relativa de cada milhar sorteada no ranking das 10.000
milhares (0 = primeira, 1 = última). Sem capacidade preditiva, a média é 0,5.
Comparações entre estratégias são pareadas (mesmas milhares sorteadas), com
correção de Holm para comparações múltiplas.

Critério de aprovação (definido antes de rodar com dados reais): uma
configuração só substitui o padrão se, no teste final, após correção de Holm,
(a) ficar abaixo de 0,5 com p < 0,05;
(b) for melhor que o padrão atual com p < 0,05; e
(c) for melhor que a referência "estrutura do universo" com p < 0,05.

Por que (c): quando a quantidade de bilhetes não é múltipla de 10.000 (ex.:
96.000), as milhares 0000–5999 têm 10 bilhetes válidos e as 6000–9999 têm 9.
Um modelo que só aprenda isso fica abaixo de 0,5 na métrica de milhar sem
nenhuma vantagem no número inteiro, que continua com chance de 1 em N.
"""

from __future__ import annotations

import itertools
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preditivo.atributos import UNIVERSO_MILHAR, IndiceHistorico, calcular_atributos, descrever_atributo
from src.preditivo.backtest import bilhetes_validos, limite_universo  # noqa: F401
from src.preditivo.modelos import NOMES_MODELOS, IndiceEstatistico, _xgboost_disponivel

ALFA = 0.05
LOTE = 20
SEMENTE = 20261107
ESTRUTURAIS = ["soma_algarismos", "algarismos_pares", "algarismos_distintos"]

HIPERPARAMETROS = {
    "simples": dict(arvores=40, profundidade=4, folha=80, iteracoes=60, prof_gb=3, C=0.01, camadas=(16,)),
    "atual": dict(arvores=80, profundidade=8, folha=40, iteracoes=150, prof_gb=4, C=0.1, camadas=(32, 16)),
    "complexo": dict(arvores=200, profundidade=14, folha=10, iteracoes=400, prof_gb=6, C=1.0, camadas=(64, 32)),
}

NOMES_REFERENCIAS = {
    "ref_aleatorio": "Seleção aleatória",
    "ref_freq_historica": "Frequência histórica (todo o passado)",
    "ref_freq_recente": "Frequência recente (100/500/1000)",
    "ref_ausencia": "Intervalo de ausência (mais atrasados primeiro)",
    "ref_estrutura": "Estrutura do universo (bilhetes válidos por milhar)",
}


@dataclass(frozen=True)
class Variante:
    nome: str
    janela: int = 2000
    atributos: str = "atual"      # atual | reduzido | ampliado
    hiper: str = "atual"          # simples | atual | complexo
    negativos: int = 15


PADRAO = Variante("padrão do motor")
VARIANTES = [
    PADRAO,
    Variante("janela 500", janela=500),
    Variante("janela 1000", janela=1000),
    Variante("janela 4000", janela=4000),
    Variante("atributos reduzidos", atributos="reduzido"),
    Variante("atributos ampliados (250 + estabilidade)", atributos="ampliado"),
    Variante("parâmetros simples", hiper="simples"),
    Variante("parâmetros complexos", hiper="complexo"),
]


# ------------------------------------------------------------------ estatística

def posicoes(score: np.ndarray, milhares: np.ndarray) -> np.ndarray:
    """Posição relativa (0 a 1) de cada milhar no ranking `score` (maior = melhor).

    Empates ficam no meio do grupo empatado. Sem informação, a média é 0,5.
    """
    alvo = score[milhares]
    maiores = (score[None, :] > alvo[:, None]).sum(axis=1)
    iguais = (score[None, :] == alvo[:, None]).sum(axis=1)
    return (maiores + 0.5 * iguais) / len(score)


def teste_contra_acaso(u) -> dict:
    u = np.asarray(u, dtype=float)
    n = len(u)
    media = float(u.mean())
    z = (media - 0.5) / math.sqrt(1 / 12 / n)
    erro = float(u.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    top10 = int((u < 10 / UNIVERSO_MILHAR).sum())
    esperado_top10 = n * 10 / UNIVERSO_MILHAR
    return {
        "n": n, "media": media, "ic95": [media - 1.96 * erro, media + 1.96 * erro],
        "z": float(z), "p": float(stats.norm.cdf(z)),
        "milhar_no_top10": top10, "milhar_no_top10_esperado": esperado_top10,
        "p_top10": float(stats.poisson.sf(top10 - 1, esperado_top10)),
    }


def teste_pareado(u_a, u_b) -> dict:
    """Diferença média A - B nas mesmas milhares (negativo = A melhor). Bilateral."""
    d = np.asarray(u_a, dtype=float) - np.asarray(u_b, dtype=float)
    n = len(d)
    media = float(d.mean())
    erro = float(d.std(ddof=1) / math.sqrt(n))
    if erro == 0:
        return {"n": n, "diferenca": media, "ic95": [media, media], "z": 0.0, "p": 1.0}
    z = media / erro
    return {"n": n, "diferenca": media, "ic95": [media - 1.96 * erro, media + 1.96 * erro],
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def holm(pvalores: dict[str, float]) -> dict[str, float]:
    """Correção de Holm-Bonferroni (controla o erro de família)."""
    ordem = sorted(pvalores, key=pvalores.get)
    m = len(ordem)
    ajustados, maximo = {}, 0.0
    for i, chave in enumerate(ordem):
        maximo = max(maximo, min(1.0, (m - i) * pvalores[chave]))
        ajustados[chave] = maximo
    return ajustados


# ------------------------------------------------------------------ atributos

def amostra(indice: IndiceHistorico, inicio: int, fim: int, negativos: int, semente: int, extras: bool):
    """Mesma amostragem do motor (mesma sequência aleatória), com atributos opcionais."""
    rng = np.random.default_rng(semente)
    ts, ms, ys = [], [], []
    for t in range(inicio, fim):
        sorteadas = set((indice.numeros[t] % UNIVERSO_MILHAR).tolist())
        negs = [m for m in rng.integers(0, UNIVERSO_MILHAR, negativos).tolist() if m not in sorteadas]
        for m in indice.numeros[t] % UNIVERSO_MILHAR:
            ts.append(t); ms.append(int(m)); ys.append(1)
        for m in negs:
            ts.append(t); ms.append(m); ys.append(0)
    X = calcular_atributos(indice, np.array(ms), np.array(ts), extras=extras)
    return X, np.array(ys)


def reduzir_atributos(X: pd.DataFrame, limite: float = 0.95) -> tuple[list[str], list[dict]]:
    """Remove estruturais, duplicados e correlação |r| > limite (mantém o primeiro)."""
    removidos, mantidos = [], []
    correl = X.corr().abs()
    for coluna in X.columns:
        if coluna in ESTRUTURAIS:
            removidos.append({"atributo": coluna, "motivo": "propriedade fixa da milhar, sem relação com o histórico"})
            continue
        if X[coluna].std() == 0:
            removidos.append({"atributo": coluna, "motivo": "constante"})
            continue
        parecido = next((m for m in mantidos if correl.loc[coluna, m] > limite), None)
        if parecido:
            removidos.append({"atributo": coluna, "motivo": f"|r| = {correl.loc[coluna, parecido]:.3f} com {parecido}"})
            continue
        mantidos.append(coluna)
    return mantidos, removidos


def analisar_atributos(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Redundância, correlação e utilidade individual (AUC) na amostra de treino."""
    correl = X.corr()
    pares = []
    colunas = list(X.columns)
    for a, b in itertools.combinations(colunas, 2):
        r = correl.loc[a, b]
        if pd.notna(r) and abs(r) >= 0.9:
            pares.append({"a": a, "b": b, "r": float(r), "identicos": bool(np.allclose(X[a], X[b]))})
    linhas, pvals = [], {}
    for c in colunas:
        pos, neg = X.loc[y == 1, c], X.loc[y == 0, c]
        if X[c].std() == 0:
            continue
        u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        auc = u / (len(pos) * len(neg))
        linhas.append({"atributo": c, "descricao": descrever_atributo(c), "auc": float(auc), "p": float(p)})
        pvals[c] = float(p)
    ajust = holm(pvals)
    for linha in linhas:
        linha["p_holm"] = ajust[linha["atributo"]]
    return {"pares_correlacionados": sorted(pares, key=lambda x: -abs(x["r"])),
            "utilidade": sorted(linhas, key=lambda x: x["p"]),
            "amostra": {"linhas": int(len(y)), "positivos": int(y.sum())}}


# ------------------------------------------------------------------ modelos

def construtores(hiper: str, semente: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    h = HIPERPARAMETROS[hiper]
    saida = {
        "estatistico": lambda: IndiceEstatistico(),
        "logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=h["C"], max_iter=1000)),
        "floresta": lambda: RandomForestClassifier(
            n_estimators=h["arvores"], max_depth=h["profundidade"], min_samples_leaf=h["folha"],
            n_jobs=-1, random_state=semente),
        "gradiente": lambda: HistGradientBoostingClassifier(
            max_iter=h["iteracoes"], learning_rate=0.05, max_depth=h["prof_gb"], random_state=semente),
        "rede_neural": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=h["camadas"], alpha=1e-3, early_stopping=True, max_iter=200,
            random_state=semente)),
    }
    if _xgboost_disponivel()[0]:
        from xgboost import XGBClassifier
        saida["xgboost"] = lambda: XGBClassifier(
            n_estimators=h["iteracoes"], max_depth=h["prof_gb"], learning_rate=0.05, subsample=0.8,
            n_jobs=-1, verbosity=0, random_state=semente)
    return saida


def treinar_variante(indice, fim: int, variante: Variante, colunas: list[str], semente: int = SEMENTE) -> dict:
    inicio = max(1, fim - variante.janela)
    X, y = amostra(indice, inicio, fim, variante.negativos, semente, variante.atributos == "ampliado")
    modelos = {}
    for chave, construir in construtores(variante.hiper, semente).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelos[chave] = construir().fit(X if chave == "estatistico" else X[colunas], y)
    return {"modelos": modelos, "colunas": colunas, "inicio": inicio, "fim": fim}


def _notas(pacote: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    saida = {}
    for chave, modelo in pacote["modelos"].items():
        if chave == "estatistico":
            saida[chave] = modelo.nota(X)
        else:
            saida[chave] = modelo.predict_proba(X[pacote["colunas"]])[:, 1]
    return saida


def _pct(v: np.ndarray) -> np.ndarray:
    return stats.rankdata(v, method="average") / len(v)


def _referencias(X: pd.DataFrame, limite: int, rng) -> dict[str, np.ndarray]:
    return {
        "ref_aleatorio": rng.random(UNIVERSO_MILHAR),
        "ref_freq_historica": X[["freq_2_total", "freq_3_total", "freq_4_total"]].mean(axis=1).to_numpy(),
        "ref_freq_recente": X[["freq_2_100", "freq_3_500", "freq_4_1000"]].mean(axis=1).to_numpy(),
        "ref_ausencia": X[["atraso_2", "atraso_3", "atraso_4"]].mean(axis=1).to_numpy(),
        "ref_estrutura": bilhetes_validos(limite),
    }



def combinacoes(chaves: list[str]) -> dict[str, dict[str, float]]:
    """Todas as combinações (peso igual) dos modelos: 'conj:a+b+c'."""
    saida = {}
    for k in range(1, len(chaves) + 1):
        for grupo in itertools.combinations(chaves, k):
            saida["conj:" + "+".join(grupo)] = {g: 1.0 for g in grupo}
    return saida


def avaliar(indice: IndiceHistorico, inicio: int, fim: int, variante: Variante, colunas: list[str],
            estrategias: dict[str, dict[str, float]], referencias: bool = True, bloco: int = 100,
            progresso: Callable[[str], None] | None = None) -> dict:
    """Walk-forward em [inicio, fim): treino antes de cada bloco, avaliação concurso a concurso.

    Devolve, por estratégia, a posição relativa de cada milhar sorteada, e o
    índice do concurso de cada posição (para separar períodos).
    """
    rng = np.random.default_rng(SEMENTE + inicio)
    extras = variante.atributos == "ampliado"
    resultado: dict[str, list[float]] = {}
    concursos: list[int] = []
    for b, comeco in enumerate(range(inicio, fim, bloco)):
        final = min(comeco + bloco, fim)
        if progresso:
            progresso(f"{variante.nome}: bloco {b + 1}, concursos {indice.concursos[comeco]}–{indice.concursos[final - 1]}")
        pacote = treinar_variante(indice, comeco, variante, colunas)
        for k in range(comeco, final, LOTE):
            ts = list(range(k, min(k + LOTE, final)))
            X = calcular_atributos(indice, np.tile(np.arange(UNIVERSO_MILHAR), len(ts)),
                                   np.repeat(np.array(ts), UNIVERSO_MILHAR), extras=extras)
            brutas = _notas(pacote, X)
            for i, t in enumerate(ts):
                fatia = slice(i * UNIVERSO_MILHAR, (i + 1) * UNIVERSO_MILHAR)
                pcts = {m: _pct(v[fatia]) for m, v in brutas.items()}
                sorteadas = indice.numeros[t] % UNIVERSO_MILHAR
                for m, v in pcts.items():
                    resultado.setdefault(f"modelo:{m}", []).extend(posicoes(v, sorteadas))
                for nome, pesos in estrategias.items():
                    if not all(m in pcts for m in pesos):
                        continue
                    total = sum(pesos.values())
                    score = sum(pcts[m] * w for m, w in pesos.items()) / total
                    resultado.setdefault(nome, []).extend(posicoes(score, sorteadas))
                if referencias:
                    Xt = X.iloc[fatia].reset_index(drop=True)
                    for nome, score in _referencias(Xt, limite_universo(indice, t), rng).items():
                        resultado.setdefault(nome, []).extend(posicoes(score, sorteadas))
                concursos.extend([int(indice.concursos[t])] * 5)
    return {"posicoes": {k: np.array(v) for k, v in resultado.items()}, "concursos": np.array(concursos)}


# ------------------------------------------------------------------ auditoria completa

def executar_auditoria(indice: IndiceHistorico, variantes: list[Variante] | None = None,
                       teste: int = 1000, ajuste: int = 500, bloco: int = 100, periodos: int = 4,
                       progresso: Callable[[str], None] | None = None) -> dict:
    variantes = variantes or VARIANTES
    padrao = variantes[0]  # a primeira variante é a configuração atual do motor
    avisar = progresso or (lambda texto: None)
    relogio = time.perf_counter()
    T = indice.total
    inicio_teste = T - teste
    inicio_ajuste = inicio_teste - ajuste
    maior_janela = max(v.janela for v in variantes)
    if inicio_ajuste < 200:
        raise ValueError(f"Histórico insuficiente: {T} concursos para ajuste {ajuste} e teste {teste}.")

    # 1. Atributos: analisados só na amostra de treino anterior ao período de ajuste.
    avisar("Analisando atributos na amostra de treino...")
    inicio_amostra = max(1, inicio_ajuste - padrao.janela)
    X, y = amostra(indice, inicio_amostra, inicio_ajuste, padrao.negativos, SEMENTE, extras=True)
    analise = analisar_atributos(X, y)
    colunas_atuais = list(calcular_atributos(indice, [0], 1).columns)
    reduzidas, removidas = reduzir_atributos(X[colunas_atuais])
    colunas = {"atual": colunas_atuais, "reduzido": reduzidas, "ampliado": list(X.columns)}

    # Uso dos atributos pelos modelos padrão (treinados com o mesmo recorte).
    pacote = treinar_variante(indice, inicio_ajuste, padrao, colunas_atuais)
    uso = {}
    for chave, modelo in pacote["modelos"].items():
        if chave == "estatistico":
            uso[chave] = {"atributos": len(IndiceEstatistico.COLUNAS), "zerados": []}
            continue
        final = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
        n = getattr(final, "n_features_in_", None)
        pesos = getattr(final, "feature_importances_", None)
        if pesos is None and hasattr(final, "coef_"):
            pesos = np.abs(final.coef_.ravel())
        zerados = [c for c, p in zip(colunas_atuais, pesos) if p <= 1e-9] if pesos is not None else []
        uso[chave] = {"atributos": int(n) if n else None, "zerados": zerados}

    # 2. Ajuste: todas as variantes e combinações, só no período de ajuste.
    chaves_modelos = list(construtores("atual", SEMENTE))
    combos = combinacoes(chaves_modelos)
    ajuste_res = {}
    for v in variantes:
        r = avaliar(indice, inicio_ajuste, inicio_teste, v, colunas[v.atributos], combos,
                    referencias=(v == padrao), bloco=bloco, progresso=avisar)
        ajuste_res[v.nome] = {k: float(u.mean()) for k, u in r["posicoes"].items()}
        ajuste_res[v.nome]["_n"] = int(len(r["concursos"]))

    candidatos = [(v, k, m) for v in variantes for k, m in ajuste_res[v.nome].items()
                  if k.startswith("conj:")]
    melhor_v, melhor_k, melhor_media = min(candidatos, key=lambda x: x[2])
    medias_modelos = {m: ajuste_res[melhor_v.nome][f"modelo:{m}"] for m in chaves_modelos}
    pesos = {m: max(0.0, 0.5 - u) for m, u in medias_modelos.items()}
    pesos_validos = sum(pesos.values()) > 0
    selecao = {
        "variante": asdict(melhor_v), "combinacao": melhor_k, "media_ajuste": melhor_media,
        "pesos_aprendidos": pesos if pesos_validos else None,
        "obs_pesos": None if pesos_validos else "Nenhum modelo ficou abaixo de 0,5 no ajuste; pesos ficam iguais.",
    }

    # 3. Teste final (fora da amostra): só agora o período final é usado.
    conj_padrao = "conj:" + "+".join(chaves_modelos)
    estr_padrao = {"padrão do motor (conjunto, peso igual)": combos[conj_padrao]}
    estr_sel = {"configuração escolhida no ajuste": combos[melhor_k]}
    if pesos_validos:
        estr_sel["escolhida com pesos aprendidos no ajuste"] = pesos
    if melhor_v == padrao:
        r_final = avaliar(indice, inicio_teste, T, padrao, colunas["atual"], {**estr_padrao, **estr_sel},
                          bloco=bloco, progresso=avisar)
    else:
        r_final = avaliar(indice, inicio_teste, T, padrao, colunas["atual"], estr_padrao, bloco=bloco,
                          progresso=avisar)
        r_sel = avaliar(indice, inicio_teste, T, melhor_v, colunas[melhor_v.atributos], estr_sel,
                        referencias=False, bloco=bloco, progresso=avisar)
        assert np.array_equal(r_sel["concursos"], r_final["concursos"])
        r_final["posicoes"].update({k: v for k, v in r_sel["posicoes"].items() if k in estr_sel})
    pos = r_final["posicoes"]

    contra_acaso = {k: teste_contra_acaso(u) for k, u in pos.items()}
    ajust = holm({k: v["p"] for k, v in contra_acaso.items() if k != "ref_aleatorio"})
    for k, v in contra_acaso.items():
        v["p_holm"] = ajust.get(k)

    principais = ["padrão do motor (conjunto, peso igual)", *estr_sel]
    comparacoes, pvals = {}, {}
    for a in principais:
        for b in ["ref_aleatorio", "ref_freq_historica", "ref_freq_recente", "ref_ausencia", "ref_estrutura",
                  "padrão do motor (conjunto, peso igual)"]:
            if a == b:
                continue
            chave = f"{a} × {NOMES_REFERENCIAS.get(b, b)}"
            comparacoes[chave] = teste_pareado(pos[a], pos[b])
            pvals[chave] = comparacoes[chave]["p"]
    for m in chaves_modelos:
        for b in ("ref_aleatorio", "ref_estrutura"):
            chave = f"{NOMES_MODELOS[m]} × {NOMES_REFERENCIAS[b]}"
            comparacoes[chave] = teste_pareado(pos[f"modelo:{m}"], pos[b])
            pvals[chave] = comparacoes[chave]["p"]
    ajust = holm(pvals)
    for k, v in comparacoes.items():
        v["p_holm"] = ajust[k]

    # Períodos cronológicos separados dentro do teste final.
    fatias = np.array_split(np.unique(r_final["concursos"]), periodos)
    por_periodo = []
    for fatia in fatias:
        mascara = np.isin(r_final["concursos"], fatia)
        linha = {"de": int(fatia.min()), "ate": int(fatia.max())}
        for k, u in pos.items():
            t = teste_contra_acaso(u[mascara])
            linha[k] = {"media": t["media"], "ic95": t["ic95"], "p": t["p"]}
        por_periodo.append(linha)

    aprovacao = {}
    for k in estr_sel:
        vs_padrao = comparacoes[f"{k} × padrão do motor (conjunto, peso igual)"]
        vs_estrutura = comparacoes[f"{k} × {NOMES_REFERENCIAS['ref_estrutura']}"]
        a = contra_acaso[k]
        aprovacao[k] = {
            "abaixo_do_acaso": bool(a["media"] < 0.5 and a["p_holm"] < ALFA),
            "melhor_que_padrao": bool(vs_padrao["diferenca"] < 0 and vs_padrao["p_holm"] < ALFA),
            "melhor_que_estrutura": bool(vs_estrutura["diferenca"] < 0 and vs_estrutura["p_holm"] < ALFA),
        }
        aprovacao[k]["aprovada"] = all(aprovacao[k].values())

    def supera(chave: str) -> bool:
        a = contra_acaso[chave]
        nome = NOMES_MODELOS.get(chave.removeprefix("modelo:"), chave)
        e = comparacoes.get(f"{nome} × {NOMES_REFERENCIAS['ref_estrutura']}") or             comparacoes.get(f"{chave} × {NOMES_REFERENCIAS['ref_estrutura']}")
        return bool(a["media"] < 0.5 and a["p_holm"] < ALFA and e and e["diferenca"] < 0 and e["p_holm"] < ALFA)

    algum_supera = any(supera(k) for k in contra_acaso if not k.startswith("ref_"))
    abaixo_de_05 = [k for k, v in contra_acaso.items()
                    if not k.startswith("ref_") and v["media"] < 0.5 and v["p_holm"] < ALFA]

    return {
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "duracao_segundos": round(time.perf_counter() - relogio, 1),
        "divisao": {
            "concursos_total": int(T),
            "amostra_atributos": [int(indice.concursos[inicio_amostra]), int(indice.concursos[inicio_ajuste - 1])],
            "ajuste": [int(indice.concursos[inicio_ajuste]), int(indice.concursos[inicio_teste - 1])],
            "teste_final": [int(indice.concursos[inicio_teste]), int(indice.concursos[T - 1])],
            "bloco": bloco, "maior_janela": maior_janela,
        },
        "atributos": {**analise, "removidos_no_reduzido": removidas, "colunas": colunas, "uso_pelos_modelos": uso},
        "ajuste": ajuste_res,
        "selecao": selecao,
        "teste_final": {"contra_acaso": contra_acaso, "comparacoes": comparacoes, "por_periodo": por_periodo},
        "aprovacao": aprovacao,
        "algum_modelo_supera_acaso": bool(algum_supera),
        "abaixo_de_05_sem_controle_estrutural": abaixo_de_05,
        "variantes": [asdict(v) for v in variantes],
    }


def salvar(resultado: dict, pasta: Path | str) -> Path:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / f"auditoria_{datetime.now():%Y%m%d-%H%M%S}.json"
    arquivo.write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    return arquivo


def _json(valor):
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if hasattr(valor, "item"):
        return valor.item()
    raise TypeError(type(valor).__name__)


# ------------------------------------------------------------------ relatório

def _nome(chave: str) -> str:
    if chave.startswith("modelo:"):
        return NOMES_MODELOS[chave.split(":", 1)[1]]
    if chave.startswith("conj:"):
        return "Conjunto: " + " + ".join(NOMES_MODELOS[m] for m in chave[5:].split("+"))
    return NOMES_REFERENCIAS.get(chave, chave)


def _f(valor, casas: int = 4) -> str:
    if valor is None:
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def gerar_relatorio(r: dict) -> str:
    """Relatório comparativo em Markdown a partir do resultado de executar_auditoria."""
    d = r["divisao"]
    linhas = [
        "# Auditoria do motor preditivo — relatório comparativo",
        "",
        f"Gerado em {r['criado_em']} ({_f(r['duracao_segundos'] / 60, 1)} min). Histórico: {d['concursos_total']} concursos.",
        "",
        "## Divisão cronológica",
        f"- Amostra para análise de atributos: concursos {d['amostra_atributos'][0]}–{d['amostra_atributos'][1]}",
        f"- Ajuste (escolha da configuração): {d['ajuste'][0]}–{d['ajuste'][1]}",
        f"- Teste final, fora da amostra: {d['teste_final'][0]}–{d['teste_final'][1]}",
        f"- Re-treino a cada {d['bloco']} concursos, sempre só com concursos anteriores.",
        "",
        "Métrica: posição média das milhares sorteadas no ranking (0,5 = acaso; menor = melhor).",
        "",
        "## Resultado do teste final (fora da amostra)",
        "",
        "| Estratégia | Posição média | IC 95% | p (Holm) | Milhar no top 10 (esperado) |",
        "|---|---|---|---|---|",
    ]
    for k, v in sorted(r["teste_final"]["contra_acaso"].items(), key=lambda x: x[1]["media"]):
        linhas.append(f"| {_nome(k)} | {_f(v['media'])} | {_f(v['ic95'][0])} a {_f(v['ic95'][1])} | "
                      f"{_f(v.get('p_holm'), 3)} | {v['milhar_no_top10']} ({_f(v['milhar_no_top10_esperado'], 1)}) |")
    linhas += ["", "## Comparações pareadas (diferença negativa = primeira estratégia melhor)", "",
               "| Comparação | Diferença | IC 95% | p (Holm) |", "|---|---|---|---|"]
    for k, v in r["teste_final"]["comparacoes"].items():
        linhas.append(f"| {k} | {_f(v['diferenca'], 5)} | {_f(v['ic95'][0], 5)} a {_f(v['ic95'][1], 5)} | {_f(v['p_holm'], 3)} |")
    linhas += ["", "## Períodos cronológicos do teste final", ""]
    chaves = ["padrão do motor (conjunto, peso igual)", "configuração escolhida no ajuste",
              "ref_estrutura", "ref_aleatorio"]
    linhas += ["| Período | " + " | ".join(_nome(c) for c in chaves) + " |", "|---" * (len(chaves) + 1) + "|"]
    for p in r["teste_final"]["por_periodo"]:
        linhas.append(f"| {p['de']}–{p['ate']} | " + " | ".join(
            f"{_f(p[c]['media'])} (p={_f(p[c]['p'], 3)})" if c in p else "—" for c in chaves) + " |")
    s = r["selecao"]
    linhas += ["", "## Escolha feita no período de ajuste", "",
               f"- Variante: **{s['variante']['nome']}**; combinação: **{_nome(s['combinacao'])}**; "
               f"posição média no ajuste: {_f(s['media_ajuste'])}.",
               f"- Pesos aprendidos: {s['pesos_aprendidos'] or s['obs_pesos']}", "",
               "| Variante | Melhor combinação | Posição média | Conjunto completo | Pior modelo isolado |",
               "|---|---|---|---|---|"]
    for nome, res in r["ajuste"].items():
        combos = {k: v for k, v in res.items() if k.startswith("conj:")}
        melhor = min(combos, key=combos.get)
        completo = max(combos, key=lambda k: k.count("+"))
        modelos = {k: v for k, v in res.items() if k.startswith("modelo:")}
        pior = max(modelos, key=modelos.get)
        linhas.append(f"| {nome} | {_nome(melhor)} | {_f(combos[melhor])} | {_f(combos[completo])} | "
                      f"{_nome(pior)} ({_f(modelos[pior])}) |")
    linhas += ["", f"Foram comparadas {sum(1 for res in r['ajuste'].values() for k in res if k.startswith('conj:'))} "
               "combinações no ajuste: com tantas tentativas, a melhor parece boa por sorte. Por isso só o teste "
               "final decide.", "", "## Critério de aprovação", ""]
    for k, a in r["aprovacao"].items():
        linhas.append(f"- **{k}**: abaixo de 0,5 = {a['abaixo_do_acaso']}; melhor que o padrão = "
                      f"{a['melhor_que_padrao']}; melhor que a estrutura do universo = {a['melhor_que_estrutura']} "
                      f"→ **{'APROVADA' if a['aprovada'] else 'não aprovada'}**")
    linhas += ["", f"Algum modelo supera o acaso (com controle estrutural): **{'sim' if r['algum_modelo_supera_acaso'] else 'não'}**.",
               "", "## Atributos", ""]
    at = r["atributos"]
    identicos = [p for p in at["pares_correlacionados"] if p["identicos"]]
    linhas.append(f"- Pares idênticos: {len(identicos)}; pares com |r| ≥ 0,9: {len(at['pares_correlacionados'])}.")
    for p in identicos[:12]:
        linhas.append(f"  - {p['a']} = {p['b']}")
    uteis = [u for u in at["utilidade"] if u["p_holm"] < ALFA]
    linhas.append(f"- Atributos com diferença significativa entre sorteadas e não sorteadas (Holm p < 0,05): "
                  f"{len(uteis)} de {len(at['utilidade'])}.")
    linhas += ["", "| Atributo | AUC | p | p (Holm) |", "|---|---|---|---|"]
    for u in at["utilidade"][:12]:
        linhas.append(f"| {u['descricao']} (`{u['atributo']}`) | {_f(u['auc'])} | {_f(u['p'], 4)} | {_f(u['p_holm'], 3)} |")
    linhas += ["", "- Removidos no conjunto reduzido:"]
    linhas += [f"  - `{x['atributo']}`: {x['motivo']}" for x in at["removidos_no_reduzido"]]
    linhas += ["", "- Uso dos atributos pelos modelos padrão:"]
    for k, u in at["uso_pelos_modelos"].items():
        linhas.append(f"  - {NOMES_MODELOS[k]}: {u['atributos']} atributos; sem uso: {', '.join(u['zerados']) or 'nenhum'}")
    return "\n".join(linhas) + "\n"
