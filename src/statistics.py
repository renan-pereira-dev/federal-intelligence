"""Análises descritivas do histórico importado.

Tudo aqui DESCREVE o passado. Nenhuma função produz probabilidade de sorteio
nem altera o gerador. Os valores "esperados" mostram o que um sorteio uniforme
produziria em média, dado o universo de cada extração, para comparação.

Observação importante: o universo vai de 00001 até a quantidade de bilhetes
emitidos (ex.: 92.000). Por isso o primeiro algarismo NÃO é uniforme: com
92.000 bilhetes, o algarismo 9 só aparece de 90000 a 92000. O esperado é
calculado exatamente para cada universo, e não suposto como 10%.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from src.probability import EXTRACOES, probabilidade_algum_premio
from fractions import Fraction

Z_95 = 1.959963984540054  # quantil 97,5% da normal padrão (intervalo de 95%)

POSICOES = ["Dezena de milhar", "Unidade de milhar", "Centena", "Dezena", "Unidade"]
COLUNAS_HISTORICO = ["concurso", "data", "extracao", "premio", "numero"]


def historico_vazio(historico: pd.DataFrame | None) -> bool:
    return historico is None or historico.empty


@lru_cache(maxsize=16)
def _proporcao_por_posicao(maximo: int, minimo: int = 1) -> np.ndarray:
    """Matriz 5x10: proporção de cada algarismo em cada posição num universo uniforme."""
    numeros = np.arange(minimo, maximo + 1)
    matriz = np.zeros((5, 10))
    for posicao in range(5):
        algarismos = (numeros // 10 ** (4 - posicao)) % 10
        matriz[posicao] = np.bincount(algarismos, minlength=10) / len(numeros)
    return matriz


def _contagem_por_extracao(historico: pd.DataFrame) -> dict[str, int]:
    return historico["extracao"].fillna("").value_counts().to_dict()


def esperado_disponivel(historico: pd.DataFrame) -> bool:
    """Só há valor esperado exato quando todas as linhas têm extração conhecida."""
    contagem = _contagem_por_extracao(historico)
    return bool(contagem) and all(chave in EXTRACOES for chave in contagem)


def matriz_digitos(numeros: pd.Series) -> np.ndarray:
    """Converte números de 5 dígitos (texto) numa matriz n x 5 de algarismos, sem laço Python."""
    if len(numeros) == 0:
        return np.zeros((0, 5), dtype=int)
    bruto = np.frombuffer("".join(numeros.tolist()).encode("ascii"), dtype=np.uint8)
    return (bruto.reshape(-1, 5) - ord("0")).astype(int)


def frequencia_por_posicao(historico: pd.DataFrame) -> pd.DataFrame:
    """Contagem observada de cada algarismo (colunas 0-9) em cada posição (linhas)."""
    digitos = matriz_digitos(historico["numero"])
    tabela = np.zeros((5, 10), dtype=int)
    for posicao in range(5):
        tabela[posicao] = np.bincount(digitos[:, posicao], minlength=10)
    return pd.DataFrame(tabela, index=POSICOES, columns=[str(d) for d in range(10)])


def esperado_por_posicao(historico: pd.DataFrame) -> pd.DataFrame | None:
    """Contagem esperada por posição sob sorteio uniforme no universo de cada extração."""
    if not esperado_disponivel(historico):
        return None
    total = np.zeros((5, 10))
    for chave, quantidade in _contagem_por_extracao(historico).items():
        total += quantidade * _proporcao_por_posicao(EXTRACOES[chave].numero_maximo)
    return pd.DataFrame(total, index=POSICOES, columns=[str(d) for d in range(10)])


def frequencia_algarismos(historico: pd.DataFrame) -> pd.DataFrame:
    """Frequência de cada algarismo somando todas as posições, com o esperado quando possível."""
    observado = frequencia_por_posicao(historico).sum(axis=0)
    df = pd.DataFrame({"algarismo": observado.index, "observado": observado.values})
    total = int(df["observado"].sum())
    df["percentual_observado"] = df["observado"] / total * 100 if total else 0.0
    esperado = esperado_por_posicao(historico)
    if esperado is not None:
        df["esperado"] = esperado.sum(axis=0).values
        df["percentual_esperado"] = df["esperado"] / total * 100 if total else 0.0
    return df


def terminacoes(historico: pd.DataFrame, digitos: int) -> pd.DataFrame:
    """Frequência das terminações com `digitos` algarismos finais (1 a 4)."""
    if digitos not in (1, 2, 3, 4):
        raise ValueError("digitos deve ser 1, 2, 3 ou 4")
    serie = historico["numero"].str[-digitos:]
    contagem = serie.value_counts().rename_axis("terminacao").reset_index(name="observado")
    total = len(serie)
    # As terminações finais são praticamente uniformes em qualquer universo
    # de 00001 a N; o esperado aproximado é total / 10^digitos.
    contagem["esperado_aprox"] = total / 10**digitos
    return contagem.sort_values(["observado", "terminacao"], ascending=[False, True]).reset_index(drop=True)


def repeticoes(historico: pd.DataFrame) -> dict:
    """Números que saíram mais de uma vez, no histórico todo e dentro do mesmo concurso."""
    contagem = historico["numero"].value_counts()
    repetidos = (
        contagem[contagem > 1].rename_axis("numero").reset_index(name="vezes")
        .sort_values(["vezes", "numero"], ascending=[False, True]).reset_index(drop=True)
    )
    # Só os concursos com número repetido entram no agrupamento (são poucos).
    duplicados = historico[historico.duplicated(["concurso", "numero"], keep=False)]
    no_mesmo = (
        duplicados.groupby("concurso")["numero"]
        .agg(lambda s: sorted(s.unique()))
        .reset_index(name="numeros_repetidos")
    )
    return {
        "numeros_repetidos": repetidos,
        "concursos_com_repeticao_interna": no_mesmo,
        "total_numeros": int(len(historico)),
        "numeros_distintos": int(historico["numero"].nunique()),
        "pares_esperados_aprox": _pares_esperados(historico),
    }


def _pares_esperados(historico: pd.DataFrame) -> float | None:
    """Pares iguais esperados ao acaso (efeito 'aniversário'), se o universo for único."""
    contagem = _contagem_por_extracao(historico)
    if len(contagem) != 1 or next(iter(contagem)) not in EXTRACOES:
        return None
    n = len(historico)
    universo = EXTRACOES[next(iter(contagem))].tamanho_universo
    return n * (n - 1) / 2 / universo


def distribuicao_faixas(historico: pd.DataFrame, largura: int = 10_000) -> pd.DataFrame:
    valores = historico["numero"].astype(int)
    inicio = (valores // largura) * largura
    df = inicio.value_counts().sort_index().rename_axis("inicio").reset_index(name="observado")
    df["faixa"] = df["inicio"].map(lambda v: f"{v:05d}-{v + largura - 1:05d}")
    return df[["faixa", "observado"]]


def distribuicao_soma_algarismos(historico: pd.DataFrame) -> pd.DataFrame:
    somas = pd.Series(matriz_digitos(historico["numero"]).sum(axis=1))
    return somas.value_counts().sort_index().rename_axis("soma").reset_index(name="observado")


def distribuicao_paridade(historico: pd.DataFrame) -> pd.DataFrame:
    par = historico["numero"].str[-1].astype(int) % 2 == 0
    return pd.DataFrame({
        "tipo": ["Par", "Ímpar"],
        "observado": [int(par.sum()), int((~par).sum())],
    })


def concursos_por_ano(historico: pd.DataFrame) -> pd.DataFrame:
    unicos = historico.drop_duplicates("concurso")
    anos = pd.to_datetime(unicos["data"]).dt.year
    return anos.value_counts().sort_index().rename_axis("ano").reset_index(name="concursos")


def frequencia_candidato(numero: str, historico: pd.DataFrame | None) -> dict | None:
    """Quantas vezes o número e suas terminações já saíram. Não é probabilidade."""
    if historico_vazio(historico):
        return None
    serie = historico["numero"]
    return {
        "exato": int((serie == numero).sum()),
        "milhar": int((serie.str[-4:] == numero[-4:]).sum()),
        "centena": int((serie.str[-3:] == numero[-3:]).sum()),
        "dezena": int((serie.str[-2:] == numero[-2:]).sum()),
        "total_numeros": int(len(serie)),
    }


# =========================================================== filtros e eras

def filtrar_historico(
    historico: pd.DataFrame,
    incluir_premio_zero: bool = True,
    ano_inicio: int | None = None,
    ano_fim: int | None = None,
) -> pd.DataFrame:
    """Aplica os filtros das telas: concursos com prêmio zero e período em anos."""
    dados = historico
    if not incluir_premio_zero and "premio_zero" in dados:
        dados = dados[~dados["premio_zero"].astype(bool)]
    anos = pd.to_datetime(dados["data"]).dt.year
    if ano_inicio is not None:
        dados = dados[anos >= ano_inicio]
        anos = anos[anos >= ano_inicio]
    if ano_fim is not None:
        dados = dados[anos <= ano_fim]
    return dados


def maior_numero_por_ano(historico: pd.DataFrame) -> pd.DataFrame:
    """Maior número sorteado em cada ano. Ajuda a ver mudanças de plano entre épocas.

    É só uma observação dos dados: a quantidade oficial de bilhetes de cada época
    antiga não está documentada no aplicativo.
    """
    df = historico.assign(ano=pd.to_datetime(historico["data"]).dt.year, valor=historico["numero"].astype(int))
    return (
        df.groupby("ano")
        .agg(maior_numero=("valor", "max"), concursos=("concurso", "nunique"))
        .reset_index()
    )


# ================================================== testes e taxas históricas

def _gamma_q(a: float, x: float) -> float:
    """Função gama incompleta regularizada superior Q(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("parâmetros inválidos")
    if x == 0:
        return 1.0
    log_prefixo = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # série
        termo = soma = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1
            termo *= x / ap
            soma += termo
            if abs(termo) < abs(soma) * 1e-15:
                break
        return max(0.0, 1.0 - soma * math.exp(log_prefixo))
    # fração contínua (método de Lentz)
    minimo = 1e-300
    b = x + 1 - a
    c = 1 / minimo
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > minimo else minimo
        c = b + an / c
        c = c if abs(c) > minimo else minimo
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(log_prefixo) * h)


def p_valor_qui_quadrado(estatistica: float, graus_liberdade: int) -> float:
    """P(X >= estatística) para X com distribuição qui-quadrado."""
    return _gamma_q(graus_liberdade / 2, estatistica / 2)


def teste_terminacoes(historico: pd.DataFrame, digitos: int, apenas_primeiro: bool = False) -> dict:
    """Compara a frequência de cada terminação com a taxa teórica 1/10^digitos.

    A taxa teórica é exata quando a quantidade de bilhetes é múltiplo de
    10^digitos (todos os planos atuais, para 1 a 3 dígitos). Para épocas
    antigas, sem quantidade oficial documentada, ela é aproximada.
    """
    dados = historico[historico["premio"] == 1] if apenas_primeiro else historico
    modulo = 10**digitos
    finais = dados["numero"].str[-digitos:].astype(int).to_numpy()
    n = len(finais)
    observado = np.bincount(finais, minlength=modulo)
    taxa_teorica = 1 / modulo
    esperado = n * taxa_teorica
    margem = Z_95 * math.sqrt(taxa_teorica * (1 - taxa_teorica) / n) if n else float("nan")
    tabela = pd.DataFrame({
        "terminacao": [f"{t:0{digitos}d}" for t in range(modulo)],
        "observado": observado,
        "taxa_observada": observado / n if n else 0.0,
        "taxa_teorica": taxa_teorica,
        "faixa_95_min": max(0.0, taxa_teorica - margem),
        "faixa_95_max": taxa_teorica + margem,
    })
    tabela["dentro_da_faixa"] = tabela["taxa_observada"].between(tabela["faixa_95_min"], tabela["faixa_95_max"])
    qui = float(((observado - esperado) ** 2 / esperado).sum()) if n else float("nan")
    gl = modulo - 1
    return {
        "digitos": digitos,
        "apenas_primeiro": apenas_primeiro,
        "n": n,
        "tabela": tabela,
        "qui_quadrado": qui,
        "graus_liberdade": gl,
        "p_valor": p_valor_qui_quadrado(qui, gl) if n else float("nan"),
        "esperado_por_terminacao": esperado,
        "teste_valido": esperado >= 5,
        "fora_da_faixa": int((~tabela["dentro_da_faixa"]).sum()),
        "fora_da_faixa_esperado": 0.05 * modulo,
    }


def taxa_coincidencia(historico: pd.DataFrame, digitos: int) -> dict:
    """Taxa de acerto histórica de uma terminação contra os cinco prêmios.

    Para cada concurso, conta quantas terminações distintas saíram entre os
    cinco prêmios. A média disso dividida por 10^digitos é a fração de
    terminações que "teriam coincidido" com pelo menos um prêmio: a taxa
    histórica média para um bilhete qualquer. A teórica é 1 - (1 - 1/10^d)^5.
    """
    modulo = 10**digitos
    distintos = historico.assign(fim=historico["numero"].str[-digitos:]).groupby("concurso")["fim"].nunique()
    taxas = distintos / modulo
    concursos = len(taxas)
    observada = float(taxas.mean()) if concursos else float("nan")
    erro_padrao = float(taxas.std(ddof=1) / math.sqrt(concursos)) if concursos > 1 else float("nan")
    teorica = float(probabilidade_algum_premio(Fraction(1, modulo)))
    z = (observada - teorica) / erro_padrao if erro_padrao and not math.isnan(erro_padrao) else float("nan")
    return {
        "digitos": digitos,
        "concursos": concursos,
        "taxa_observada": observada,
        "taxa_teorica": teorica,
        "erro_padrao": erro_padrao,
        "faixa_95_min": observada - Z_95 * erro_padrao if concursos > 1 else float("nan"),
        "faixa_95_max": observada + Z_95 * erro_padrao if concursos > 1 else float("nan"),
        "z": z,
        "compativel": bool(abs(z) <= Z_95) if not math.isnan(z) else None,
    }


def taxa_candidatos(conferencias: pd.DataFrame) -> dict | None:
    """Taxa de acerto real dos candidatos gerados pelo app e já conferidos."""
    if conferencias is None or conferencias.empty:
        return None
    resultados = [json.loads(c) for c in conferencias["conferencia"]]
    n = len(resultados)
    exatos = sum(1 for r in resultados if r.get("premio_principal"))
    unidade = sum(1 for r in resultados if r.get("unidade_primeiro"))
    algum = sum(
        1 for r in resultados
        if r.get("premio_principal") or r.get("terminacoes") or r.get("aproximacao")
        or r.get("dezena_final_vizinha") or r.get("unidade_primeiro")
    )
    # Até 5 prêmios distintos por concurso: 5/N é o limite superior por candidato.
    esperado_exato_max = sum(
        5 / EXTRACOES[r["extracao"]].numero_maximo for r in resultados if r.get("extracao") in EXTRACOES
    )
    return {
        "conferidos": n,
        "exatos": exatos,
        "esperado_exatos_max": esperado_exato_max,
        "unidade_primeiro": unidade,
        "esperado_unidade_primeiro": n * 0.1,
        "com_alguma_coincidencia": algum,
    }


# =========================================================== décadas e épocas

def decada(datas: pd.Series) -> pd.Series:
    return (pd.to_datetime(datas).dt.year // 10 * 10).astype(int)


def rotulo_decada(inicio: int) -> str:
    return f"{inicio}–{inicio + 9}"


def resumo_decadas(historico: pd.DataFrame) -> pd.DataFrame:
    """Por década: volume, faixa observada, taxa histórica da unidade e testes de uniformidade."""
    dados = historico.assign(decada=decada(historico["data"]))
    linhas = []
    for inicio, grupo in dados.groupby("decada"):
        valores = grupo["numero"].astype(int)
        taxa = taxa_coincidencia(grupo, 1)
        testes = {d: teste_terminacoes(grupo, d) for d in (1, 2, 3)}
        linhas.append({
            "decada": rotulo_decada(int(inicio)),
            "inicio": int(inicio),
            "concursos": int(grupo["concurso"].nunique()),
            "numeros": int(len(grupo)),
            "menor_numero": int(valores.min()),
            "maior_numero": int(valores.max()),
            "taxa_unidade": taxa["taxa_observada"],
            "taxa_unidade_min": taxa["faixa_95_min"],
            "taxa_unidade_max": taxa["faixa_95_max"],
            "taxa_unidade_teorica": taxa["taxa_teorica"],
            **{
                f"p_{d}_digito{'s' if d > 1 else ''}": (t["p_valor"] if t["teste_valido"] else float("nan"))
                for d, t in testes.items()
            },
        })
    return pd.DataFrame(linhas)


def primeiro_algarismo_por_decada(historico: pd.DataFrame) -> pd.DataFrame:
    """Percentual de cada 1º algarismo (dezena de milhar) por década.

    Mostra as mudanças de plano: em épocas com menos bilhetes, os algarismos
    altos do início simplesmente não aparecem.
    """
    dados = historico.assign(decada=decada(historico["data"]), primeiro=historico["numero"].str[0])
    tabela = pd.crosstab(dados["decada"], dados["primeiro"], normalize="index") * 100
    tabela = tabela.reindex(columns=[str(d) for d in range(10)], fill_value=0.0)
    tabela.index = [rotulo_decada(int(i)) for i in tabela.index]
    return tabela


def _binomial_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    log = (
        math.lgamma(n + 1) - np.array([math.lgamma(x + 1) + math.lgamma(n - x + 1) for x in k])
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )
    return np.exp(log)


def distribuicao_contagens(historico: pd.DataFrame, digitos: int = 3) -> pd.DataFrame:
    """Distribuição de frequências das terminações.

    Para cada k, quantas terminações (de 10^digitos possíveis) saíram exatamente k
    vezes, ao lado do esperado ao acaso. Se cada terminação tem chance 1/10^d em
    cada um dos n números, a contagem de uma terminação segue uma Binomial(n, 1/10^d);
    o esperado de terminações com k ocorrências é 10^d × P(Binomial = k).
    """
    modulo = 10**digitos
    finais = historico["numero"].str[-digitos:].astype(int).to_numpy()
    n = len(finais)
    contagens = np.bincount(finais, minlength=modulo)
    observado = np.bincount(contagens)
    k_max = max(len(observado) - 1, int(n / modulo + 4 * math.sqrt(max(n / modulo, 1))))
    k = np.arange(0, k_max + 1)
    esperado = modulo * _binomial_pmf(k, n, 1 / modulo) if n else np.zeros(len(k))
    obs = np.zeros(len(k), dtype=int)
    obs[: len(observado)] = observado
    return pd.DataFrame({"vezes": k, "terminacoes_observadas": obs, "terminacoes_esperadas": esperado})


def valor_critico_qui_quadrado(graus_liberdade: int, alfa: float = 0.05) -> float:
    """Valor x tal que P(qui-quadrado >= x) = alfa (busca por bissecção)."""
    baixo, alto = 0.0, graus_liberdade + 20 * math.sqrt(2 * graus_liberdade) + 50
    for _ in range(200):
        meio = (baixo + alto) / 2
        if p_valor_qui_quadrado(meio, graus_liberdade) > alfa:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def resumo_testes_qui_quadrado(historico: pd.DataFrame) -> pd.DataFrame:
    """Todos os testes de uniformidade usados no painel, prontos para tabela e gráfico."""
    nomes = {1: "Unidade", 2: "Dezena", 3: "Centena", 4: "Milhar"}
    testes = [(f"{nomes[d]} (5 prêmios)", teste_terminacoes(historico, d)) for d in (1, 2, 3, 4)]
    testes.append(("Unidade do 1º prêmio", teste_terminacoes(historico, 1, apenas_primeiro=True)))
    linhas = []
    for nome, t in testes:
        critico = valor_critico_qui_quadrado(t["graus_liberdade"])
        linhas.append({
            "teste": nome,
            "n": t["n"],
            "qui_quadrado": t["qui_quadrado"],
            "graus_liberdade": t["graus_liberdade"],
            "valor_critico_95": critico,
            "razao": t["qui_quadrado"] / critico if critico else float("nan"),
            "p_valor": t["p_valor"],
            "valido": t["teste_valido"],
        })
    return pd.DataFrame(linhas)
