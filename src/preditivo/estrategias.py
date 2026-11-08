"""Laboratório de estratégias de seleção dos 10 candidatos (módulo experimental).

Não altera o motor padrão. Uma estratégia é um RANKING das 10.000 milhares
combinado com uma regra de SELEÇÃO dos 10 candidatos:

Rankings: cada modelo isolado; conjunto com peso igual (o motor padrão);
conjunto ponderado (pesos aprendidos só com concursos já avaliados);
frequência histórica; frequência recente; combinação de atributos
estatísticos; intervalo de ausência; estrutura do universo (controle).

Seleções: tradicional (os 10 primeiros); diversificada por unidade, dezena ou
centena (não repete a terminação); e combinação de ranking e diversidade (MMR,
com peso lambda para o ranking).

Referências: seleção aleatória sem reposição; motor padrão (conjunto + seleção
tradicional); controle estrutural.

Protocolo (fixado antes de rodar):
1. Desenvolvimento: todas as estratégias são avaliadas em walk-forward.
2. Pré-registro: com dados do desenvolvimento apenas, escolhem-se até 3
   candidatas por regras fixas (abaixo). Nada do período final é lido antes.
3. Amostra final: avaliada uma única vez. Só as pré-registradas podem ser
   aprovadas. As demais aparecem como resultados exploratórios.

Aprovação (todas as condições, com correção de Holm na família das candidatas):
a) mais coincidências de centena que a seleção aleatória (pareado);
b) excesso de centena acima do esperado pela própria cobertura (previsão,
   não cobertura);
c) posição média melhor que o controle estrutural (pareado);
d) melhor que o motor padrão em posição média OU em coincidências de centena.

Por que centena: o acerto exato e a milhar são raros demais para ter poder
estatístico (cerca de 0,25 e 2,5 esperados em 500 concursos); a dezena é
dominada pela cobertura. Todas as métricas são relatadas mesmo assim.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preditivo.atributos import UNIVERSO_MILHAR, IndiceHistorico, calcular_atributos
from src.preditivo.backtest import bilhetes_validos, limite_universo, posicoes_no_ranking, teste_pareado
from src.preditivo.modelos import CONFIG_PADRAO, NOMES_MODELOS, Configuracao, notas, percentil, treinar

ALFA = 0.05
QUANTIDADE = 10
LOTE = 20
SEMENTE = 20261108
POOL_MMR = 300
LAMBDAS = (0.25, 0.5, 0.75)

NOMES_RANKINGS = {
    **{f"modelo:{k}": v for k, v in NOMES_MODELOS.items()},
    "ens_igual": "Conjunto com peso igual (motor padrão)",
    "ens_ponderado": "Conjunto ponderado (pesos do passado)",
    "freq_historica": "Frequência histórica",
    "freq_recente": "Frequência recente",
    "combo_estatistica": "Combinação de atributos estatísticos",
    "ausencia": "Intervalo de ausência",
    "estrutura": "Controle: estrutura do universo",
    "aleatorio": "Ranking aleatório",
}
NOMES_SELECOES = {
    "tradicional": "tradicional (10 primeiros)",
    "div_unidade": "diversificada por unidade",
    "div_dezena": "diversificada por dezena",
    "div_centena": "diversificada por centena",
    **{f"mmr_{l}": f"ranking + diversidade (λ={l})" for l in LAMBDAS},
}
SELECAO_ALEATORIA = "selecao_aleatoria"
MOTOR_PADRAO = "ens_igual|tradicional"
CONTROLE_ESTRUTURAL = "estrutura|tradicional"
REFERENCIAS = {SELECAO_ALEATORIA, MOTOR_PADRAO, CONTROLE_ESTRUTURAL}
# Rankings usados como referência/controle, nunca candidatos à aprovação.
RANKINGS_CONTROLE = {"estrutura", "aleatorio"}
FREQUENCIAS = [f"freq_{d}_{w}" for d in (1, 2, 3) for w in ("total", 50, 100, 500, 1000)] + \
    ["freq_4_total", "freq_4_500", "freq_4_1000"] + \
    [f"posicao_{p}_{w}" for p in (2, 3, 4) for w in (500, "total")]


def nome_estrategia(chave: str) -> str:
    if chave == SELECAO_ALEATORIA:
        return "Seleção aleatória sem reposição"
    ranking, selecao = chave.split("|")
    return f"{NOMES_RANKINGS[ranking]} · {NOMES_SELECOES[selecao]}"


# ------------------------------------------------------------------ rankings

def rankings_base(X: pd.DataFrame, brutas: dict[str, np.ndarray], pesos: dict[str, float],
                  limite: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Notas das 10.000 milhares para cada ranking (maior = melhor). Só usa dados < t."""
    pcts = {k: percentil(v) for k, v in brutas.items()}
    saida = {f"modelo:{k}": v for k, v in pcts.items()}
    saida["ens_igual"] = np.mean(list(pcts.values()), axis=0) if pcts else np.zeros(UNIVERSO_MILHAR)
    total = sum(pesos.get(k, 0.0) for k in pcts)
    if total > 0:
        saida["ens_ponderado"] = sum(pcts[k] * pesos.get(k, 0.0) for k in pcts) / total
    else:
        saida["ens_ponderado"] = saida["ens_igual"]
    saida["freq_historica"] = X[["freq_2_total", "freq_3_total", "freq_4_total"]].mean(axis=1).to_numpy()
    saida["freq_recente"] = X[["freq_2_100", "freq_3_500", "freq_4_1000"]].mean(axis=1).to_numpy()
    colunas = [c for c in FREQUENCIAS if c in X.columns]
    saida["combo_estatistica"] = X[colunas].rank(pct=True).mean(axis=1).to_numpy()
    saida["ausencia"] = X[["atraso_2", "atraso_3", "atraso_4"]].mean(axis=1).to_numpy()
    saida["estrutura"] = bilhetes_validos(limite)
    saida["aleatorio"] = rng.random(UNIVERSO_MILHAR)
    return saida


def ordem(score: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Milhares da maior para a menor nota; empates em ordem aleatória (reprodutível)."""
    return np.lexsort((rng.random(len(score)), -score))


# ------------------------------------------------------------------ seleções

def _completar(milhares: list[int], limite: int, rng: np.random.Generator) -> list[str]:
    """Põe um 1º algarismo válido, sorteado, em cada milhar (o ranking não o avalia)."""
    numeros = []
    for m in milhares:
        primeiros = [p for p in range(10) if 1 <= p * UNIVERSO_MILHAR + m <= limite]
        numeros.append(f"{primeiros[int(rng.integers(len(primeiros)))] * UNIVERSO_MILHAR + m:05d}")
    return numeros


def selecionar(regra: str, ordem_milhares: np.ndarray, validas: np.ndarray, limite: int,
               rng: np.random.Generator, quantidade: int = QUANTIDADE) -> list[str]:
    """Aplica a regra de seleção ao ranking e devolve `quantidade` números distintos."""
    candidatos = [int(m) for m in ordem_milhares if validas[m]]
    if regra == "tradicional":
        escolhidas = candidatos[:quantidade]
    elif regra.startswith("div_"):
        d = {"div_unidade": 1, "div_dezena": 2, "div_centena": 3}[regra]
        usados, escolhidas = set(), []
        for m in candidatos:
            if m % 10**d not in usados:
                usados.add(m % 10**d)
                escolhidas.append(m)
                if len(escolhidas) == quantidade:
                    break
    elif regra.startswith("mmr_"):
        lam = float(regra.split("_")[1])
        pool = np.array(candidatos[:POOL_MMR])
        nota = 1 - np.arange(len(pool)) / len(pool)
        digitos = np.stack([(pool // 10**k) % 10 for k in range(4)], axis=1)
        semelhanca = np.zeros(len(pool))
        livre = np.ones(len(pool), dtype=bool)
        escolhidas = []
        for _ in range(quantidade):
            valor = np.where(livre, lam * nota - (1 - lam) * semelhanca, -np.inf)
            i = int(np.argmax(valor))
            escolhidas.append(int(pool[i]))
            livre[i] = False
            semelhanca = np.maximum(semelhanca, (digitos == digitos[i]).mean(axis=1))
    else:
        raise ValueError(f"Regra de seleção desconhecida: {regra}")
    if len(escolhidas) < quantidade:
        raise ValueError("Não há milhares suficientes para a seleção.")
    return _completar(escolhidas, limite, rng)


def selecao_aleatoria(limite: int, rng: np.random.Generator, quantidade: int = QUANTIDADE) -> list[str]:
    return [f"{v:05d}" for v in rng.choice(np.arange(1, limite + 1), quantidade, replace=False)]


# ------------------------------------------------------------------ métricas

def contagem_terminacoes(terminacoes, d: int, limite: int) -> int:
    """Quantos números de 1 a `limite` terminam em algum dos valores de d dígitos dados."""
    total = 0
    for r in set(int(x) for x in terminacoes):
        if r == 0:
            total += limite // 10**d
        elif r <= limite:
            total += (limite - r) // 10**d + 1
    return total


def metricas_selecao(candidatos: list[str], sorteados: list[str], limite: int) -> dict:
    """Métricas de um concurso. Acerto exato separado das coincidências parciais.

    - exato: quantos dos 5 prêmios são iguais a algum candidato.
    - milhar/centena/dezena: quantos prêmios NÃO acertados exatamente têm os
      últimos 4/3/2 algarismos iguais aos de algum candidato (inclusivas entre si).
    - esperado_*: o mesmo número sob sorteio uniforme de 1 a `limite`, dada a
      cobertura de terminações destes 10 candidatos.
    - Precision@10 = candidatos iguais a algum prêmio / 10.
    - Recall@10 = prêmios cujo número está entre os candidatos / 5.
    """
    conjunto = set(candidatos)
    exatos = [s in conjunto for s in sorteados]
    saida = {
        "exato": sum(exatos),
        "principal": int(candidatos[0] in set(sorteados)),
        "precision": sum(c in set(sorteados) for c in candidatos) / len(candidatos),
        "recall": sum(exatos) / len(sorteados),
        "esperado_exato": len(sorteados) * len(conjunto) / limite,
        "var_exato": len(sorteados) * (len(conjunto) / limite) * (1 - len(conjunto) / limite),
    }
    for d, nome in ((4, "milhar"), (3, "centena"), (2, "dezena")):
        finais = {c[-d:] for c in candidatos}
        saida[nome] = sum(1 for s, e in zip(sorteados, exatos) if not e and s[-d:] in finais)
        p = max(contagem_terminacoes(finais, d, limite) - len(conjunto), 0) / limite
        saida[f"esperado_{nome}"] = len(sorteados) * p
        saida[f"var_{nome}"] = len(sorteados) * p * (1 - p)
    return saida


def diversidade(candidatos: list[str]) -> dict:
    """Quanto os 10 candidatos cobrem de algarismos e terminações."""
    return {
        "algarismos_por_posicao": float(np.mean([len({c[i] for c in candidatos}) for i in range(5)])),
        "unidades": len({c[-1:] for c in candidatos}),
        "dezenas": len({c[-2:] for c in candidatos}),
        "centenas": len({c[-3:] for c in candidatos}),
        "milhares": len({c[-4:] for c in candidatos}),
        "semelhanca_media": float(np.mean([
            sum(a[i] == b[i] for i in range(1, 5)) / 4
            for k, a in enumerate(candidatos) for b in candidatos[k + 1:]
        ])),
    }


# ------------------------------------------------------------------ avaliação

def _pesos(somas: dict[str, float], contagem: int) -> dict[str, float]:
    if contagem == 0:
        return {}
    pesos = {k: max(0.0, 0.5 - s / contagem) for k, s in somas.items()}
    return pesos if sum(pesos.values()) > 0 else {}


def avaliar(indice: IndiceHistorico, inicio: int, fim: int, config: Configuracao = CONFIG_PADRAO,
            estado: dict | None = None, progresso: Callable[[str], None] | None = None) -> dict:
    """Walk-forward em [inicio, fim) para TODAS as estratégias, nas mesmas condições.

    Os modelos são treinados antes de cada bloco só com concursos anteriores. Os
    pesos do conjunto ponderado vêm só de concursos já avaliados (`estado`).
    """
    estado = estado if estado is not None else {"somas": {}, "contagem": 0}
    rng = np.random.default_rng(SEMENTE + inicio)
    registros: dict[str, list[dict]] = {}
    posicoes: dict[str, list[float]] = {}
    concursos: list[int] = []
    for comeco in range(inicio, fim, config.bloco):
        final = min(comeco + config.bloco, fim)
        if progresso:
            progresso(f"concursos {indice.concursos[comeco]}–{indice.concursos[final - 1]}")
        pacote = treinar(indice, comeco, config)
        pesos = _pesos(estado["somas"], estado["contagem"])
        novas_somas: dict[str, float] = {}
        for k in range(comeco, final, LOTE):
            ts = list(range(k, min(k + LOTE, final)))
            X = calcular_atributos(indice, np.tile(np.arange(UNIVERSO_MILHAR), len(ts)),
                                   np.repeat(np.array(ts), UNIVERSO_MILHAR))
            brutas = notas(pacote, X)
            for i, t in enumerate(ts):
                fatia = slice(i * UNIVERSO_MILHAR, (i + 1) * UNIVERSO_MILHAR)
                Xt = X.iloc[fatia].reset_index(drop=True)
                limite = limite_universo(indice, t)
                validas = bilhetes_validos(limite) > 0
                sorteados = [f"{n:05d}" for n in indice.numeros[t]]
                milhares = indice.numeros[t] % UNIVERSO_MILHAR
                notas_t = rankings_base(Xt, {m: v[fatia] for m, v in brutas.items()}, pesos, limite, rng)
                for nome, score in notas_t.items():
                    u = posicoes_no_ranking(score, milhares)
                    posicoes.setdefault(nome, []).extend(u.tolist())
                    if nome.startswith("modelo:"):
                        novas_somas[nome[7:]] = novas_somas.get(nome[7:], 0.0) + float(u.sum())
                    if nome == "aleatorio":
                        continue
                    ordenadas = ordem(score, rng)
                    for regra in NOMES_SELECOES:
                        cand = selecionar(regra, ordenadas, validas, limite, rng)
                        registros.setdefault(f"{nome}|{regra}", []).append(
                            {**metricas_selecao(cand, sorteados, limite), **diversidade(cand)})
                cand = selecao_aleatoria(limite, rng)
                registros.setdefault(SELECAO_ALEATORIA, []).append(
                    {**metricas_selecao(cand, sorteados, limite), **diversidade(cand)})
                concursos.append(int(indice.concursos[t]))
        for m, s in novas_somas.items():  # pesos atualizados só depois do bloco
            estado["somas"][m] = estado["somas"].get(m, 0.0) + s
        estado["contagem"] += 5 * (final - comeco)
    return {"registros": {k: pd.DataFrame(v) for k, v in registros.items()},
            "posicoes": {k: np.array(v) for k, v in posicoes.items()},
            "concursos": np.array(concursos), "estado": estado}


# ------------------------------------------------------------------ estatística

def resumo(df: pd.DataFrame, u: np.ndarray | None) -> dict:
    """Métricas observadas com IC 95% (Poisson exato para contagens; normal para médias)."""
    n = len(df)
    saida = {"concursos": n}
    for nome in ("exato", "milhar", "centena", "dezena"):
        obs = int(df[nome].sum())
        esp = float(df[f"esperado_{nome}"].sum())
        var = float(df[f"var_{nome}"].sum())
        baixo = stats.chi2.ppf(0.025, 2 * obs) / 2 if obs > 0 else 0.0
        alto = stats.chi2.ppf(0.975, 2 * obs + 2) / 2
        z = (obs - esp) / math.sqrt(var) if var > 0 else 0.0
        saida[nome] = {"observado": obs, "ic95": [float(baixo), float(alto)], "esperado_cobertura": esp,
                       "excesso": obs - esp, "z_excesso": z, "p_excesso": float(2 * stats.norm.sf(abs(z)))}
    saida["principal"] = int(df["principal"].sum())
    saida["precision_10"] = float(df["precision"].mean())
    saida["recall_10"] = float(df["recall"].mean())
    saida["diversidade"] = {k: float(df[k].mean()) for k in
                            ("algarismos_por_posicao", "unidades", "dezenas", "centenas", "milhares", "semelhanca_media")}
    if u is not None:
        media = float(u.mean())
        erro = float(u.std(ddof=1) / math.sqrt(len(u)))
        z = (media - 0.5) / math.sqrt(1 / 12 / len(u))
        saida["posicao"] = {"media": media, "ic95": [media - 1.96 * erro, media + 1.96 * erro],
                            "p_contra_05": float(2 * stats.norm.sf(abs(z)))}
    return saida


def holm(pvalores: dict[str, float]) -> dict[str, float]:
    ordem_p = sorted(pvalores, key=pvalores.get)
    ajustados, maximo, m = {}, 0.0, len(ordem_p)
    for i, chave in enumerate(ordem_p):
        maximo = max(maximo, min(1.0, (m - i) * pvalores[chave]))
        ajustados[chave] = maximo
    return ajustados


def _posicoes_da(chave: str, posicoes: dict) -> np.ndarray | None:
    if chave == SELECAO_ALEATORIA:
        return posicoes.get("aleatorio")
    return posicoes.get(chave.split("|")[0])


def comparar(a: str, b: str, r: dict) -> dict:
    """Comparações pareadas (mesmos concursos) entre as estratégias a e b."""
    ra, rb = r["registros"][a], r["registros"][b]
    saida = {"centena": teste_pareado(ra["centena"], rb["centena"]),
             "dezena": teste_pareado(ra["dezena"], rb["dezena"]),
             "milhar": teste_pareado(ra["milhar"], rb["milhar"])}
    ua, ub = _posicoes_da(a, r["posicoes"]), _posicoes_da(b, r["posicoes"])
    if ua is not None and ub is not None:
        saida["posicao"] = teste_pareado(ua, ub)
    return saida


def pre_registrar(dev: dict) -> dict:
    """Escolhe as candidatas SÓ com o desenvolvimento, por regras fixas."""
    elegiveis = [k for k in dev["registros"] if k not in REFERENCIAS
                 and k.split("|")[0] not in RANKINGS_CONTROLE]
    resumos = {k: resumo(dev["registros"][k], _posicoes_da(k, dev["posicoes"])) for k in elegiveis}
    c1 = max(elegiveis, key=lambda k: (resumos[k]["centena"]["z_excesso"], k))
    rankings = {k.split("|")[0] for k in elegiveis}
    melhor_ranking = min(rankings, key=lambda n: (float(dev["posicoes"][n].mean()), n))
    c2 = f"{melhor_ranking}|tradicional"
    diversificadas = [k for k in elegiveis if k.startswith("ens_igual|") and k != MOTOR_PADRAO]
    c3 = max(diversificadas, key=lambda k: (resumos[k]["centena"]["z_excesso"], k))
    candidatas = list(dict.fromkeys([c1, c2, c3]))
    return {
        "candidatas": candidatas,
        "regras": {
            c1: "maior excesso de centena sobre a cobertura no desenvolvimento",
            c2: "melhor posição média no desenvolvimento (seleção tradicional)",
            c3: "melhor seleção diversificada do motor padrão no desenvolvimento",
        },
    }


def executar(indice: IndiceHistorico, final: int = 1000, desenvolvimento: int = 1000,
             periodos: int = 4, config: Configuracao = CONFIG_PADRAO,
             progresso: Callable[[str], None] | None = None) -> dict:
    """Desenvolvimento → pré-registro → amostra final (uma vez). Devolve o relatório."""
    relogio = time.perf_counter()
    avisar = progresso or (lambda texto: None)
    T = indice.total
    inicio_final = T - final
    inicio_dev = inicio_final - desenvolvimento
    if inicio_dev < config.minimo_treino + 1:
        raise ValueError(f"Histórico insuficiente: {T} concursos.")

    avisar("Desenvolvimento")
    dev = avaliar(indice, inicio_dev, inicio_final, config, progresso=lambda t: avisar(f"Desenvolvimento: {t}"))
    registro = pre_registrar(dev)
    registro["criado_em"] = datetime.now().isoformat(timespec="seconds")
    registro["impressao"] = hashlib.sha256(json.dumps(registro["candidatas"]).encode()).hexdigest()[:12]
    avisar(f"Pré-registro: {registro['candidatas']}")

    avisar("Amostra final")
    fin = avaliar(indice, inicio_final, T, config, estado=dev["estado"],
                  progresso=lambda t: avisar(f"Final: {t}"))
    divisao = {"desenvolvimento": [int(indice.concursos[inicio_dev]), int(indice.concursos[inicio_final - 1])],
               "final": [int(indice.concursos[inicio_final]), int(indice.concursos[T - 1])],
               "bloco": config.bloco, "janela_treino": config.janela_treino}
    relatorio = montar_relatorio(dev, fin, registro, divisao, periodos, config)
    relatorio["duracao_segundos"] = round(time.perf_counter() - relogio, 1)
    # Dados brutos por concurso: permitem recalcular os critérios sem rodar tudo de novo.
    relatorio["_dados"] = {"desenvolvimento": dev, "final": fin, "pre_registro": registro}
    return relatorio


def montar_relatorio(dev: dict, fin: dict, registro: dict, divisao: dict, periodos: int,
                     config: Configuracao) -> dict:
    """Critérios, comparações e tabelas a partir dos dados brutos (sem treinar nada)."""

    def tabela(r: dict) -> dict:
        return {k: resumo(df, _posicoes_da(k, r["posicoes"])) for k, df in r["registros"].items()}

    tab_dev, tab_fin = tabela(dev), tabela(fin)

    # Aprovação: família = candidatas pré-registradas × 4 testes, Holm.
    testes, pvals = {}, {}
    for c in registro["candidatas"]:
        vs_acaso = comparar(c, SELECAO_ALEATORIA, fin)["centena"]
        vs_estrutura = comparar(c, CONTROLE_ESTRUTURAL, fin).get("posicao")
        vs_motor = comparar(c, MOTOR_PADRAO, fin)
        excesso = tab_fin[c]["centena"]
        testes[c] = {"a_vs_aleatoria_centena": vs_acaso, "b_excesso_centena": excesso,
                     "c_vs_estrutura_posicao": vs_estrutura, "d_vs_motor": vs_motor}
        pvals[f"{c}|a"] = vs_acaso["p"]
        pvals[f"{c}|b"] = excesso["p_excesso"]
        pvals[f"{c}|c"] = vs_estrutura["p"] if vs_estrutura else 1.0
        # d) só conta a direção favorável: posição menor OU mais centenas que o motor.
        favoraveis = [1.0]
        if vs_motor.get("posicao") and vs_motor["posicao"]["diferenca"] < 0:
            favoraveis.append(vs_motor["posicao"]["p"])
        if vs_motor["centena"]["diferenca"] > 0:
            favoraveis.append(vs_motor["centena"]["p"])
        pvals[f"{c}|d"] = 1.0 if c == MOTOR_PADRAO else min(favoraveis)
    ajust = holm(pvals)
    aprovacao = {}
    for c in registro["candidatas"]:
        t = testes[c]
        mot = t["d_vs_motor"]
        melhor_motor = c != MOTOR_PADRAO and ajust[f"{c}|d"] < ALFA
        criterios = {
            "supera_aleatoria": bool(t["a_vs_aleatoria_centena"]["diferenca"] > 0 and ajust[f"{c}|a"] < ALFA),
            "excesso_sobre_cobertura": bool(t["b_excesso_centena"]["excesso"] > 0 and ajust[f"{c}|b"] < ALFA),
            "supera_estrutura": bool(t["c_vs_estrutura_posicao"] is not None
                                     and t["c_vs_estrutura_posicao"]["diferenca"] < 0 and ajust[f"{c}|c"] < ALFA),
            "supera_motor_padrao": bool(melhor_motor),
        }
        aprovacao[c] = {"criterios": criterios, "aprovada": all(criterios.values()),
                        "p_holm": {k: ajust[f"{c}|{k}"] for k in "abcd"}}

    # Exploratório: todas as estratégias contra a seleção aleatória (centena), Holm em todas.
    exploratorio = {}
    for k in fin["registros"]:
        if k == SELECAO_ALEATORIA:
            continue
        exploratorio[k] = comparar(k, SELECAO_ALEATORIA, fin)["centena"]
    ajust_exp = holm({k: v["p"] for k, v in exploratorio.items()})
    for k, v in exploratorio.items():
        v["p_holm"] = ajust_exp[k]

    # Pergunta da diversidade: motor tradicional × diversificado × aleatória.
    diversidade_q = {}
    for k in [x for x in fin["registros"] if x.startswith("ens_igual|") and x != MOTOR_PADRAO]:
        diversidade_q[k] = {"bruto_vs_motor": comparar(k, MOTOR_PADRAO, fin),
                            "bruto_vs_aleatoria": comparar(k, SELECAO_ALEATORIA, fin)}

    # Períodos cronológicos da amostra final.
    fatias = np.array_split(np.arange(len(fin["concursos"])), periodos)
    acompanhar = list(dict.fromkeys([*registro["candidatas"], MOTOR_PADRAO, CONTROLE_ESTRUTURAL, SELECAO_ALEATORIA]))
    por_periodo = []
    for fatia in fatias:
        linha = {"de": int(fin["concursos"][fatia[0]]), "ate": int(fin["concursos"][fatia[-1]])}
        for k in acompanhar:
            df = fin["registros"][k].iloc[fatia]
            u = _posicoes_da(k, fin["posicoes"])
            linha[k] = resumo(df, None if u is None else u[fatia[0] * 5:(fatia[-1] + 1) * 5])
        por_periodo.append(linha)

    pesos_finais = _pesos(fin["estado"]["somas"], fin["estado"]["contagem"])
    return {
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "divisao": divisao,
        "estrategias_testadas": sorted(fin["registros"]),
        "quantidade_estrategias": len(fin["registros"]),
        "pre_registro": registro,
        "desenvolvimento": tab_dev,
        "final": tab_fin,
        "aprovacao": aprovacao,
        "alguma_aprovada": any(a["aprovada"] for a in aprovacao.values()),
        "exploratorio": exploratorio,
        "diversidade": diversidade_q,
        "por_periodo": por_periodo,
        "pesos_ponderado": pesos_finais,
        "config": {"versao_atributos": config.versao_atributos, "hash": config.hash()},
    }


# ------------------------------------------------------------------ gravação e registro

def pasta(caminho_banco) -> Path:
    p = Path(caminho_banco).resolve().parent / "experimentos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    raise TypeError(type(v).__name__)


def salvar(relatorio: dict, caminho_banco) -> Path:
    """Grava o relatório (JSON) e, se houver, os dados brutos (joblib) ao lado."""
    import joblib

    arquivo = pasta(caminho_banco) / f"estrategias_{datetime.now():%Y%m%d-%H%M%S}.json"
    publico = {k: v for k, v in relatorio.items() if k != "_dados"}
    arquivo.write_text(json.dumps(publico, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    if "_dados" in relatorio:
        joblib.dump(relatorio["_dados"], arquivo.with_suffix(".dados.joblib"), compress=3)
    return arquivo


def ultimo_relatorio(caminho_banco) -> tuple[Path, dict] | None:
    arquivos = sorted(pasta(caminho_banco).glob("estrategias_*.json"))
    if not arquivos:
        return None
    return arquivos[-1], json.loads(arquivos[-1].read_text(encoding="utf-8"))


# Estratégia ativa do motor: arquivo com histórico, para poder voltar atrás.
PADRAO_ATIVO = {"estrategia": MOTOR_PADRAO, "relatorio": None}


def _arquivo_ativo(caminho_banco) -> Path:
    return pasta(caminho_banco) / "estrategia_ativa.json"


def estrategia_ativa(caminho_banco) -> dict:
    arquivo = _arquivo_ativo(caminho_banco)
    if not arquivo.exists():
        return {**PADRAO_ATIVO, "historico": []}
    return json.loads(arquivo.read_text(encoding="utf-8"))


def ativar(estrategia: str, relatorio: dict, nome_relatorio: str, caminho_banco) -> dict:
    """Ativa uma estratégia SÓ se ela foi aprovada no relatório. Guarda a anterior."""
    if estrategia != MOTOR_PADRAO and not relatorio.get("aprovacao", {}).get(estrategia, {}).get("aprovada"):
        raise ValueError("Só uma estratégia aprovada no backtest pode ser ativada.")
    atual = estrategia_ativa(caminho_banco)
    novo = {"estrategia": estrategia, "relatorio": nome_relatorio,
            "pesos": relatorio.get("pesos_ponderado"),
            "historico": atual["historico"] + [{k: atual.get(k) for k in ("estrategia", "relatorio", "pesos")}
                                                | {"trocado_em": datetime.now().isoformat(timespec="seconds")}]}
    _arquivo_ativo(caminho_banco).write_text(json.dumps(novo, ensure_ascii=False, indent=1), encoding="utf-8")
    return novo


def reverter(caminho_banco) -> dict:
    """Volta para a configuração anterior (ou para o motor padrão)."""
    atual = estrategia_ativa(caminho_banco)
    if not atual["historico"]:
        anterior = dict(PADRAO_ATIVO)
        historico = []
    else:
        anterior = {k: v for k, v in atual["historico"][-1].items() if k != "trocado_em"}
        historico = atual["historico"][:-1]
    novo = {**PADRAO_ATIVO, **anterior, "historico": historico}
    _arquivo_ativo(caminho_banco).write_text(json.dumps(novo, ensure_ascii=False, indent=1), encoding="utf-8")
    return novo


# ------------------------------------------------------------------ próximo concurso

def candidatos_proximo(indice: IndiceHistorico, pacote: dict, estrategia: str, limite: int,
                       pesos: dict | None = None, semente: int | None = None) -> dict:
    """Os 10 candidatos de uma estratégia para o próximo concurso (dados até o último)."""
    rng = np.random.default_rng(semente)
    t = indice.total
    X = calcular_atributos(indice, np.arange(UNIVERSO_MILHAR), t)
    notas_t = rankings_base(X, notas(pacote, X), pesos or {}, limite, rng)
    if estrategia == SELECAO_ALEATORIA:
        cand = selecao_aleatoria(limite, rng)
        pontos = [None] * QUANTIDADE
    else:
        ranking, regra = estrategia.split("|")
        score = notas_t[ranking]
        cand = selecionar(regra, ordem(score, rng), bilhetes_validos(limite) > 0, limite, rng)
        pct = 100 * percentil(score)
        pontos = [round(float(pct[int(c) % UNIVERSO_MILHAR]), 2) for c in cand]
    return {"estrategia": estrategia, "candidatos": cand, "pontuacao": pontos, "diversidade": diversidade(cand)}



# ------------------------------------------------------------------ relatório

def _n(v, casas=4):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


def gerar_relatorio(r: dict) -> str:
    """Relatório comparativo em Markdown."""
    d = r["divisao"]
    L = [
        "# Laboratório de estratégias — relatório comparativo", "",
        f"Gerado em {r['criado_em']} ({_n(r['duracao_segundos'] / 60, 1)} min). "
        f"{r['quantidade_estrategias']} estratégias, todas nas mesmas condições.", "",
        f"- Desenvolvimento (escolhas e pré-registro): concursos {d['desenvolvimento'][0]}–{d['desenvolvimento'][1]}",
        f"- Amostra final (avaliada uma vez): {d['final'][0]}–{d['final'][1]}",
        f"- Walk-forward: re-treino a cada {d['bloco']} concursos, janela de {d['janela_treino']} concursos anteriores.",
        "", "## Pré-registro", "",
        f"Impressão `{r['pre_registro']['impressao']}`, criado em {r['pre_registro']['criado_em']}.", "",
    ]
    L += [f"- **{nome_estrategia(c)}**: {r['pre_registro']['regras'][c]}" for c in r["pre_registro"]["candidatas"]]
    L += ["", "## Aprovação (amostra final, Holm nas candidatas × 4 critérios)", "",
          "| Candidata | a) > aleatória | b) excesso | c) > estrutura | d) > motor | Aprovada |", "|---|---|---|---|---|---|"]
    for c, a in r["aprovacao"].items():
        cr, p = a["criterios"], a["p_holm"]
        L.append(f"| {nome_estrategia(c)} | {'sim' if cr['supera_aleatoria'] else 'não'} (p={_n(p['a'], 3)}) | "
                 f"{'sim' if cr['excesso_sobre_cobertura'] else 'não'} (p={_n(p['b'], 3)}) | "
                 f"{'sim' if cr['supera_estrutura'] else 'não'} (p={_n(p['c'], 3)}) | "
                 f"{'sim' if cr['supera_motor_padrao'] else 'não'} (p={_n(p['d'], 3)}) | "
                 f"**{'sim' if a['aprovada'] else 'não'}** |")
    L += ["", "Valores-p de a, b e c são bilaterais; o critério também exige a direção favorável (um p pequeno "
          "na direção contrária significa resultado PIOR). Em d, o p já considera só a direção favorável.",
          "", f"Alguma estratégia aprovada: **{'sim' if r['alguma_aprovada'] else 'não'}**.", ""]

    def tabela(tab: dict, titulo: str, chaves: list[str]) -> None:
        L.extend(["", f"## {titulo}", "",
                  "| Estratégia | Exatos (esp.) | Milhar (esp.) | Centena [IC95%] (esp.) | z excesso | Dezena (esp.) | "
                  "P@10 | R@10 | Posição média [IC95%] | Centenas distintas |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        for k in chaves:
            t = tab[k]
            pos = t.get("posicao")
            c = t["centena"]
            L.append(
                f"| {nome_estrategia(k)} | {t['exato']['observado']} ({_n(t['exato']['esperado_cobertura'], 2)}) | "
                f"{t['milhar']['observado']} ({_n(t['milhar']['esperado_cobertura'], 2)}) | "
                f"{c['observado']} [{_n(c['ic95'][0], 1)}–{_n(c['ic95'][1], 1)}] ({_n(c['esperado_cobertura'], 2)}) | "
                f"{_n(c['z_excesso'], 2)} | {t['dezena']['observado']} ({_n(t['dezena']['esperado_cobertura'], 1)}) | "
                f"{_n(t['precision_10'], 5)} | {_n(t['recall_10'], 5)} | "
                + ("—" if not pos else f"{_n(pos['media'])} [{_n(pos['ic95'][0])}–{_n(pos['ic95'][1])}]")
                + f" | {_n(t['diversidade']['centenas'], 2)} |")

    principais = list(dict.fromkeys([*r["pre_registro"]["candidatas"], MOTOR_PADRAO, CONTROLE_ESTRUTURAL,
                                     SELECAO_ALEATORIA]))
    tabela(r["final"], "Amostra final (fora da amostra): candidatas e referências", principais)
    tabela(r["desenvolvimento"], "Desenvolvimento (dentro da amostra): candidatas e referências", principais)

    L += ["", "## Diversidade (motor padrão com seleções diferentes, amostra final)", "",
          "| Seleção | Centenas distintas | Centena obs. | Esperada (cobertura) | z excesso | vs motor (dif./concurso, p) | vs aleatória (p) |",
          "|---|---|---|---|---|---|---|"]
    for k in [MOTOR_PADRAO, *r["diversidade"], SELECAO_ALEATORIA]:
        t = r["final"][k]
        comp = r["diversidade"].get(k, {})
        vm = comp.get("bruto_vs_motor", {}).get("centena")
        va = comp.get("bruto_vs_aleatoria", {}).get("centena")
        L.append(f"| {nome_estrategia(k)} | {_n(t['diversidade']['centenas'], 2)} | {t['centena']['observado']} | "
                 f"{_n(t['centena']['esperado_cobertura'], 2)} | {_n(t['centena']['z_excesso'], 2)} | "
                 + ("—" if not vm else f"{_n(vm['diferenca'], 4)}, p={_n(vm['p'], 3)}") + " | "
                 + ("—" if not va else f"p={_n(va['p'], 3)}") + " |")

    L += ["", "## Por período (amostra final): centenas observadas / esperadas · posição média", "",
          "| Período | " + " | ".join(nome_estrategia(k) for k in principais) + " |",
          "|---" * (len(principais) + 1) + "|"]
    for p in r["por_periodo"]:
        celulas = []
        for k in principais:
            t = p[k]
            pos = t.get("posicao")
            celulas.append(f"{t['centena']['observado']}/{_n(t['centena']['esperado_cobertura'], 1)}"
                           + ("" if not pos else f" · {_n(pos['media'])}"))
        L.append(f"| {p['de']}–{p['ate']} | " + " | ".join(celulas) + " |")

    exp = r["exploratorio"]
    signif = [k for k, v in exp.items() if v["p_holm"] < ALFA]
    menores = sorted(exp, key=lambda k: exp[k]["p"])[:10]
    L += ["", "## Exploratório: todas as estratégias contra a seleção aleatória (centena, amostra final)", "",
          f"{len(exp)} comparações; significativas após Holm: {len(signif)}.", "",
          "| Estratégia (10 menores p) | Diferença por concurso | p | p Holm |", "|---|---|---|---|"]
    L += [f"| {nome_estrategia(k)} | {_n(exp[k]['diferenca'], 4)} | {_n(exp[k]['p'], 4)} | {_n(exp[k]['p_holm'], 3)} |"
          for k in menores]
    L += ["", "## Todas as estratégias testadas", "", ", ".join(nome_estrategia(k) for k in r["estrategias_testadas"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":  # .venv\Scripts\python.exe -m src.preditivo.estrategias
    from src import database as db

    inicio_execucao = time.perf_counter()
    relatorio = executar(IndiceHistorico(db.carregar_historico()),
                         progresso=lambda t: print(f"[{time.perf_counter() - inicio_execucao:6.0f}s] {t}", flush=True))
    arquivo = salvar(relatorio, db.resolver_caminho())
    arquivo.with_suffix(".md").write_text(gerar_relatorio(relatorio), encoding="utf-8")
    print("Relatório gravado em", arquivo)
