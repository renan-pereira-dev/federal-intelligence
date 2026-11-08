"""Auditoria final: o motor prevê melhor que o acaso? (não altera o motor)

Backtest walk-forward do motor PADRÃO no maior trecho possível do histórico:
antes de cada bloco de 100 concursos, os modelos são treinados só com
concursos anteriores (em memória; os modelos gravados não são tocados), e cada
concurso é avaliado com indicadores calculados só com o passado.

Estratégias avaliadas nas mesmas condições:
- motor padrão (conjunto com peso igual, 10 primeiros);
- motor diversificado por centena;
- controle estrutural (milhares com mais bilhetes válidos);
- seleção aleatória sem reposição.

Métricas, por concurso, para os 10 candidatos (definições fixas):
- "ao menos um" no nível X: algum dos 5 prêmios tem os últimos k algarismos
  iguais aos de algum candidato (exato = número inteiro; milhar k=4; centena
  k=3; dezena k=2; unidade k=1). Níveis inclusivos: acertar a milhar também é
  acertar a centena. O acerto exato é sempre relatado à parte.
- por prêmio: proporção dos 5 prêmios com coincidência no nível X.
- esperado pelo acaso: calculado com a cobertura de terminações dos próprios
  10 candidatos e o universo estimado da época (maior número dos 500
  concursos anteriores).

Análise principal (declarada antes de rodar): concursos ANTERIORES a 2578,
trecho nunca usado em nenhuma avaliação ou ajuste anterior. O restante é
secundário (já foi olhado em auditorias e no laboratório).

Classificação:
- "vantagem comprovada": supera esperado, aleatória e estrutura após Holm;
- "sinal promissor": o trecho inédito é dividido em duas janelas cronológicas
  (primeira e segunda metade). O sinal precisa ter p < 0,05, na direção
  favorável, na primeira janela E na segunda, cada uma analisada sozinha;
- "sem vantagem": nos demais casos.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preditivo.atributos import UNIVERSO_MILHAR, IndiceHistorico, calcular_atributos
from src.preditivo.backtest import bilhetes_validos, limite_universo, posicoes_no_ranking, teste_pareado
from src.preditivo.estrategias import (
    contagem_terminacoes, ordem, rankings_base, selecao_aleatoria, selecionar,
)
from src.preditivo.modelos import CONFIG_PADRAO, Configuracao, notas, treinar

ALFA = 0.05
META = 0.70
LOTE = 20
SEMENTE = 20261109
CORTE_INEDITO = 2578  # concursos < 2578 nunca foram usados em avaliações anteriores
NIVEIS = {"exato": 5, "milhar": 4, "centena": 3, "dezena": 2, "unidade": 1}
ESTRATEGIAS = {
    "motor": "Motor padrão (10 primeiros do conjunto)",
    "motor_diversificado": "Motor diversificado por centena",
    "estrutura": "Controle: estrutura do universo",
    "aleatoria": "Seleção aleatória sem reposição",
}


# ------------------------------------------------------------------ métricas

def linha_metricas(candidatos: list[str], sorteados: list[str], limite: int) -> dict:
    """Coincidências (inclusivas) e probabilidade por prêmio sob o acaso, por nível."""
    saida = {}
    distintos = len(set(candidatos))
    for nivel, k in NIVEIS.items():
        finais = {c[-k:] for c in candidatos}
        saida[f"{nivel}_n"] = sum(s[-k:] in finais for s in sorteados)
        if nivel == "exato":
            p = distintos / limite
        else:
            p = contagem_terminacoes(finais, k, limite) / limite
        saida[f"{nivel}_p"] = min(p, 1.0)
    return saida


def wilson(sucessos: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z = 1.959963984540054
    centro = (sucessos + z * z / 2) / (n + z * z)
    meia = z * math.sqrt(sucessos * (n - sucessos) / n + z * z / 4) / (n + z * z)
    return [max(0.0, centro - meia), min(1.0, centro + meia)]


def taxa_maxima_cobertura(nivel: str, candidatos: int = 10, limite: int = 96_000) -> float:
    """Maior taxa 'ao menos um' que o ACASO dá com `candidatos` números de terminações distintas."""
    k = NIVEIS[nivel]
    cobertos = min(candidatos, 10**k) if nivel != "exato" else candidatos
    p = cobertos / (limite if nivel == "exato" else 10**k)
    return 1 - (1 - min(p, 1.0)) ** 5


def meta_70(nivel: str, n: int, candidatos: int = 10, limite: int = 96_000) -> dict:
    """O que a meta de 70% exige em cada nível, e o que dá para detectar com n concursos."""
    acaso = taxa_maxima_cobertura(nivel, candidatos, limite)
    k = NIVEIS[nivel]
    universo = limite if nivel == "exato" else 10**k
    # Quantos números com terminações distintas seriam necessários só pelo acaso.
    p_necessario = 1 - (1 - META) ** (1 / 5)
    necessarios = math.ceil(p_necessario * universo)
    efeito_detectavel = (1.959963984540054 + 0.8416212335729143) * math.sqrt(acaso * (1 - acaso) / n) if n else None
    return {
        "taxa_acaso_10": acaso,
        "multiplicador_necessario": META / acaso if acaso > 0 else None,
        "numeros_necessarios_pelo_acaso": necessarios,
        "atingivel_por_cobertura": acaso >= META,
        "menor_diferenca_detectavel": efeito_detectavel,
    }


# ------------------------------------------------------------------ backtest

def executar(indice: IndiceHistorico, inicio: int = 1000, fim: int | None = None,
             config: Configuracao = CONFIG_PADRAO,
             progresso: Callable[[str], None] | None = None) -> dict:
    """Walk-forward do motor padrão em [inicio, fim). Nada é gravado em disco."""
    fim = indice.total if fim is None else fim
    if inicio < config.minimo_treino + 1:
        raise ValueError("Início cedo demais: não há passado suficiente para treinar.")
    relogio = time.perf_counter()
    rng = np.random.default_rng(SEMENTE)
    linhas, posicoes = [], {"motor": [], "estrutura": [], "aleatoria": []}
    for comeco in range(inicio, fim, config.bloco):
        final = min(comeco + config.bloco, fim)
        if progresso:
            progresso(f"concursos {indice.concursos[comeco]}–{indice.concursos[final - 1]}")
        pacote = treinar(indice, comeco, config)  # só concursos < comeco
        for k in range(comeco, final, LOTE):
            ts = list(range(k, min(k + LOTE, final)))
            X = calcular_atributos(indice, np.tile(np.arange(UNIVERSO_MILHAR), len(ts)),
                                   np.repeat(np.array(ts), UNIVERSO_MILHAR))
            brutas = notas(pacote, X)
            for i, t in enumerate(ts):  # cada t usa só concursos < t
                fatia = slice(i * UNIVERSO_MILHAR, (i + 1) * UNIVERSO_MILHAR)
                Xt = X.iloc[fatia].reset_index(drop=True)
                limite = limite_universo(indice, t)
                validas = bilhetes_validos(limite) > 0
                sorteados = [f"{n:05d}" for n in indice.numeros[t]]
                milhares = indice.numeros[t] % UNIVERSO_MILHAR
                r = rankings_base(Xt, {m: v[fatia] for m, v in brutas.items()}, {}, limite, rng)
                ordem_motor = ordem(r["ens_igual"], rng)
                selecoes = {
                    "motor": selecionar("tradicional", ordem_motor, validas, limite, rng),
                    "motor_diversificado": selecionar("div_centena", ordem_motor, validas, limite, rng),
                    "estrutura": selecionar("tradicional", ordem(r["estrutura"], rng), validas, limite, rng),
                    "aleatoria": selecao_aleatoria(limite, rng),
                }
                for nome, cand in selecoes.items():
                    linhas.append({"concurso": int(indice.concursos[t]), "estrategia": nome, "limite": limite,
                                   **linha_metricas(cand, sorteados, limite)})
                posicoes["motor"].extend(posicoes_no_ranking(r["ens_igual"], milhares).tolist())
                posicoes["estrutura"].extend(posicoes_no_ranking(r["estrutura"], milhares).tolist())
                posicoes["aleatoria"].extend(posicoes_no_ranking(r["aleatorio"], milhares).tolist())
    return {"linhas": pd.DataFrame(linhas), "posicoes": {k: np.array(v) for k, v in posicoes.items()},
            "duracao_segundos": round(time.perf_counter() - relogio, 1),
            "concursos": [int(indice.concursos[inicio]), int(indice.concursos[fim - 1])]}


# ------------------------------------------------------------------ análise

def holm(pvalores: dict[str, float]) -> dict[str, float]:
    ordem_p = sorted(pvalores, key=pvalores.get)
    ajustados, maximo, m = {}, 0.0, len(ordem_p)
    for i, chave in enumerate(ordem_p):
        maximo = max(maximo, min(1.0, (m - i) * pvalores[chave]))
        ajustados[chave] = maximo
    return ajustados


def analisar_trecho(df: pd.DataFrame, posicoes: dict, mascara_concursos: np.ndarray) -> dict:
    """Taxas, IC, comparação com o esperado, com a aleatória e com a estrutura."""
    concursos = np.sort(df["concurso"].unique())
    dentro = set(concursos[mascara_concursos(concursos)].tolist())
    d = df[df["concurso"].isin(dentro)]
    largo = {e: d[d["estrategia"] == e].set_index("concurso").sort_index() for e in ESTRATEGIAS}
    n = len(largo["motor"])
    saida = {"concursos": n, "intervalo": [min(dentro), max(dentro)] if dentro else None, "estrategias": {}}
    pvals, favoravel = {}, {}
    for e, t in largo.items():
        res = {}
        for nivel in NIVEIS:
            acerto = (t[f"{nivel}_n"] > 0).astype(int).to_numpy()
            p = t[f"{nivel}_p"].to_numpy()
            esperado_concurso = 1 - (1 - p) ** 5
            obs, esp = int(acerto.sum()), float(esperado_concurso.sum())
            var = float((esperado_concurso * (1 - esperado_concurso)).sum())
            z = (obs - esp) / math.sqrt(var) if var > 0 else 0.0
            por_premio = int(t[f"{nivel}_n"].sum())
            item = {
                "taxa_concurso": obs / n if n else None, "ic95": wilson(obs, n),
                "acertos_concursos": obs, "taxa_esperada": esp / n if n else None,
                "z_vs_esperado": z, "p_vs_esperado": float(2 * stats.norm.sf(abs(z))),
                "taxa_por_premio": por_premio / (5 * n) if n else None,
                "taxa_por_premio_esperada": float(p.mean()) if n else None,
            }
            if e != "aleatoria":
                item["vs_aleatoria"] = teste_pareado(acerto, (largo["aleatoria"][f"{nivel}_n"] > 0).astype(int))
                pvals[f"{e}|{nivel}|esperado"] = item["p_vs_esperado"]
                pvals[f"{e}|{nivel}|aleatoria"] = item["vs_aleatoria"]["p"]
                favoravel[f"{e}|{nivel}|esperado"] = z > 0
                favoravel[f"{e}|{nivel}|aleatoria"] = item["vs_aleatoria"]["diferenca"] > 0
            if e not in ("estrutura", "aleatoria"):
                item["vs_estrutura"] = teste_pareado(acerto, (largo["estrutura"][f"{nivel}_n"] > 0).astype(int))
                pvals[f"{e}|{nivel}|estrutura"] = item["vs_estrutura"]["p"]
                favoravel[f"{e}|{nivel}|estrutura"] = item["vs_estrutura"]["diferenca"] > 0
            res[nivel] = item
        saida["estrategias"][e] = res
    # Posição média das milhares sorteadas (5 por concurso, mesma ordem dos concursos).
    todos = np.sort(df["concurso"].unique())
    sel = np.repeat(np.isin(todos, list(dentro)), 5)
    pos = {}
    for k, u in posicoes.items():
        v = u[sel]
        media = float(v.mean())
        erro = float(v.std(ddof=1) / math.sqrt(len(v)))
        z = (media - 0.5) / math.sqrt(1 / 12 / len(v))
        pos[k] = {"media": media, "ic95": [media - 1.96 * erro, media + 1.96 * erro], "n": int(len(v)),
                  "p_vs_05": float(2 * stats.norm.sf(abs(z)))}
    pos["motor_vs_estrutura"] = teste_pareado(posicoes["motor"][sel], posicoes["estrutura"][sel])
    pvals["posicao|vs_05"] = pos["motor"]["p_vs_05"]
    pvals["posicao|vs_estrutura"] = pos["motor_vs_estrutura"]["p"]
    favoravel["posicao|vs_05"] = pos["motor"]["media"] < 0.5
    favoravel["posicao|vs_estrutura"] = pos["motor_vs_estrutura"]["diferenca"] < 0
    saida["posicao"] = pos
    ajust = holm(pvals)
    saida["p_holm"] = ajust
    # Vantagem = melhor que esperado, que a aleatória E que a estrutura, na direção favorável, após Holm.
    vantagens = []
    for e in ("motor", "motor_diversificado"):
        for nivel in NIVEIS:
            it = saida["estrategias"][e][nivel]
            if (it["z_vs_esperado"] > 0 and ajust[f"{e}|{nivel}|esperado"] < ALFA
                    and it["vs_aleatoria"]["diferenca"] > 0 and ajust[f"{e}|{nivel}|aleatoria"] < ALFA
                    and it["vs_estrutura"]["diferenca"] > 0 and ajust[f"{e}|{nivel}|estrutura"] < ALFA):
                vantagens.append(f"{e}|{nivel}")
    if (pos["motor"]["media"] < 0.5 and ajust["posicao|vs_05"] < ALFA
            and pos["motor_vs_estrutura"]["diferenca"] < 0 and ajust["posicao|vs_estrutura"] < ALFA):
        vantagens.append("motor|posicao")
    saida["vantagens_comprovadas"] = vantagens
    # Sinais nominais: p < 0,05 SEM correção e na direção favorável (não valem como prova).
    saida["sinais_nominais"] = sorted(k for k, p in pvals.items() if p < ALFA and favoravel[k])
    saida["desfavoraveis_nominais"] = sorted(k for k, p in pvals.items() if p < ALFA and not favoravel[k])
    saida["testes"] = len(pvals)
    return saida


def analisar(resultado: dict, periodos: int = 4) -> dict:
    df, pos = resultado["linhas"], resultado["posicoes"]
    trechos = {
        "principal_inedito": lambda c: c < CORTE_INEDITO,
        "secundario_ja_usado": lambda c: c >= CORTE_INEDITO,
        "completo": lambda c: np.ones(len(c), dtype=bool),
    }
    saida = {"criado_em": datetime.now().isoformat(timespec="seconds"),
             "concursos_avaliados": resultado["concursos"], "duracao_segundos": resultado["duracao_segundos"],
             "trechos": {k: analisar_trecho(df, pos, f) for k, f in trechos.items()}}
    todos = np.sort(df["concurso"].unique())
    cortes = np.array_split(todos, periodos)
    saida["periodos"] = [
        {"de": int(c.min()), "ate": int(c.max()),
         **analisar_trecho(df, pos, lambda x, c=c: np.isin(x, c))}
        for c in cortes
    ]
    n = saida["trechos"]["principal_inedito"]["concursos"]
    saida["meta_70"] = {nivel: meta_70(nivel, n) for nivel in NIVEIS}

    # Replicação: o trecho inédito é dividido em duas janelas cronológicas
    # independentes (primeira e segunda metade), analisadas separadamente.
    ineditos = todos[todos < CORTE_INEDITO]
    janela_1, janela_2 = np.array_split(ineditos, 2) if len(ineditos) >= 2 else (ineditos, ineditos[:0])
    saida["replicacao"] = {
        nome: {"de": int(j.min()) if len(j) else None, "ate": int(j.max()) if len(j) else None,
               **analisar_trecho(df, pos, lambda x, j=j: np.isin(x, j))}
        for nome, j in (("janela_1", janela_1), ("janela_2", janela_2))
    }
    classificacao = classificar(saida["trechos"]["principal_inedito"],
                                saida["replicacao"]["janela_1"], saida["replicacao"]["janela_2"])
    saida["sinais_replicados"] = classificacao["sinais_replicados"]
    saida["conclusao"] = classificacao["conclusao"]
    return saida


def sinais_replicados(janela_1: dict, janela_2: dict) -> list[str]:
    """Sinais com p < 0,05, na direção favorável, nas DUAS janelas, cada uma por si.

    `sinais_nominais` de cada janela já contém só os testes com p < 0,05 (sem
    correção) e direção favorável, calculados apenas com os concursos daquela
    janela. Um sinal que aparece em só uma delas não conta.
    """
    return sorted(set(janela_1.get("sinais_nominais", [])) & set(janela_2.get("sinais_nominais", [])))


def classificar(principal: dict, janela_1: dict, janela_2: dict) -> dict:
    """Conclusão da auditoria.

    - vantagem_comprovada: algum teste no trecho inédito inteiro supera esperado,
      aleatória e estrutura após Holm (critério anterior, inalterado);
    - sinal_promissor_nao_confirmado: algum sinal favorável com p < 0,05 na
      primeira janela E, de forma independente, p < 0,05 na segunda;
    - sem_vantagem: nos demais casos (inclusive sinais que aparecem em uma só janela).
    """
    replicados = sinais_replicados(janela_1, janela_2)
    if principal["vantagens_comprovadas"]:
        conclusao = "vantagem_comprovada"
    elif replicados:
        conclusao = "sinal_promissor_nao_confirmado"
    else:
        conclusao = "sem_vantagem"
    return {"conclusao": conclusao, "sinais_replicados": replicados}


# ------------------------------------------------------------------ relatório

CONCLUSOES = {
    "vantagem_comprovada": "EXISTE VANTAGEM PREDITIVA COMPROVADA",
    "sinal_promissor_nao_confirmado": "SINAL PROMISSOR, NÃO CONFIRMADO",
    "sem_vantagem": "NÃO HÁ VANTAGEM SOBRE O ACASO",
}


def _pc(v, casas=2):
    return "—" if v is None else f"{100 * v:.{casas}f}%".replace(".", ",")


def _n(v, casas=4):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


def gerar_relatorio(a: dict) -> str:
    L = ["# Auditoria final — o Federal Intelligence prevê melhor que o acaso?", "",
         f"Gerado em {a['criado_em']}. Concursos avaliados: {a['concursos_avaliados'][0]}–{a['concursos_avaliados'][1]} "
         f"({_n(a['duracao_segundos'] / 60, 1)} min).", "",
         f"## Conclusão: **{CONCLUSOES[a['conclusao']]}**", ""]
    for nome, titulo in (("principal_inedito", f"Análise principal: concursos inéditos (< {CORTE_INEDITO})"),
                         ("secundario_ja_usado", f"Secundária: concursos já usados antes (≥ {CORTE_INEDITO})"),
                         ("completo", "Histórico avaliado completo")):
        t = a["trechos"][nome]
        L += ["", f"## {titulo}", "", f"{t['concursos']} concursos ({t['intervalo'][0]}–{t['intervalo'][1]}); "
              f"{t['testes']} testes com correção de Holm.", "",
              "Taxa = concursos em que ao menos um dos 5 prêmios coincide com algum dos 10 candidatos no nível.", "",
              "| Estratégia | Nível | Taxa observada [IC95%] | Esperada pelo acaso | p vs esperado | Dif. vs aleatória (p) | Dif. vs estrutura (p) |",
              "|---|---|---|---|---|---|---|"]
        for e, nome_e in ESTRATEGIAS.items():
            for nivel, it in t["estrategias"][e].items():
                va, ve = it.get("vs_aleatoria"), it.get("vs_estrutura")
                L.append(f"| {nome_e} | {nivel} | {_pc(it['taxa_concurso'])} [{_pc(it['ic95'][0])}–{_pc(it['ic95'][1])}] | "
                         f"{_pc(it['taxa_esperada'])} | {_n(it['p_vs_esperado'], 3)} | "
                         + ("—" if not va else f"{_pc(va['diferenca'])} ({_n(va['p'], 3)})") + " | "
                         + ("—" if not ve else f"{_pc(ve['diferenca'])} ({_n(ve['p'], 3)})") + " |")
        p = t["posicao"]
        L += ["", f"Posição média das milhares sorteadas (0,5 = acaso): motor {_n(p['motor']['media'])} "
              f"[{_n(p['motor']['ic95'][0])}–{_n(p['motor']['ic95'][1])}], p = {_n(p['motor']['p_vs_05'], 3)}; "
              f"estrutura {_n(p['estrutura']['media'])}; aleatório {_n(p['aleatoria']['media'])}; "
              f"motor − estrutura = {_n(p['motor_vs_estrutura']['diferenca'], 5)} (p = {_n(p['motor_vs_estrutura']['p'], 3)}).",
              "", f"- Vantagens comprovadas (Holm, direção favorável, contra esperado, aleatória e estrutura): "
              f"{', '.join(t['vantagens_comprovadas']) or 'nenhuma'}",
              f"- Sinais nominais favoráveis (p < 0,05 sem correção; NÃO são prova): {', '.join(t['sinais_nominais']) or 'nenhum'}",
              f"- Resultados nominais desfavoráveis: {', '.join(t['desfavoraveis_nominais']) or 'nenhum'}"]
    rep = a.get("replicacao")
    if rep:
        j1, j2 = rep["janela_1"], rep["janela_2"]
        L += ["", "## Replicação no trecho inédito (regra do sinal promissor)", "",
              f"- Janela 1: concursos {j1['de']}–{j1['ate']} ({j1['concursos']} concursos). "
              f"Sinais com p < 0,05: {', '.join(j1['sinais_nominais']) or 'nenhum'}",
              f"- Janela 2: concursos {j2['de']}–{j2['ate']} ({j2['concursos']} concursos). "
              f"Sinais com p < 0,05: {', '.join(j2['sinais_nominais']) or 'nenhum'}",
              f"- Replicados nas duas janelas: **{', '.join(a['sinais_replicados']) or 'nenhum'}**"]
    L += ["", "## Por período (motor padrão)", "",
          "| Período | Concursos | Exato | Milhar | Centena | Dezena | Posição média |", "|---|---|---|---|---|---|---|"]
    for per in a["periodos"]:
        m = per["estrategias"]["motor"]
        L.append(f"| {per['de']}–{per['ate']} | {per['concursos']} | " + " | ".join(
            f"{_pc(m[n]['taxa_concurso'])} (esp. {_pc(m[n]['taxa_esperada'])})" for n in ("exato", "milhar", "centena", "dezena"))
            + f" | {_n(per['posicao']['motor']['media'])} |")
    L += ["", "## A meta de 70%", "",
          "| Nível | Taxa do acaso com 10 números | Multiplicador necessário | Números necessários só pelo acaso | 70% atingível só por cobertura? | Menor diferença detectável (80% de poder) |",
          "|---|---|---|---|---|---|"]
    for nivel, m in a["meta_70"].items():
        L.append(f"| {nivel} | {_pc(m['taxa_acaso_10'], 3)} | {_n(m['multiplicador_necessario'], 1)}× | "
                 f"{m['numeros_necessarios_pelo_acaso']} | {'sim' if m['atingivel_por_cobertura'] else 'não'} | "
                 f"{_pc(m['menor_diferenca_detectavel'])} |")
    return "\n".join(L) + "\n"
