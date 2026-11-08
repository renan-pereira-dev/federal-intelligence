"""Auditoria e experimentos do motor. Somente dados ARTIFICIAIS."""

import numpy as np
import pytest

from src.preditivo import experimentos as ex
from src.preditivo.atributos import IndiceHistorico, calcular_atributos
from tests.test_preditivo import historico_artificial

VARIANTES_TESTE = [
    ex.Variante("padrão do motor", janela=100, hiper="simples"),
    ex.Variante("reduzida", janela=100, atributos="reduzido", hiper="simples"),
]


# ------------------------------------------------------------------ estatística

def test_posicoes_primeiro_ultimo_e_empates():
    score = np.arange(10_000, dtype=float)  # milhar 9999 tem a maior nota
    u = ex.posicoes(score, np.array([9999, 0]))
    assert u[0] == pytest.approx(0.5 / 10_000)
    assert u[1] == pytest.approx(1 - 0.5 / 10_000)
    empatado = np.zeros(10_000)
    assert ex.posicoes(empatado, np.array([5]))[0] == pytest.approx(0.5)


def test_posicao_media_sem_informacao_e_meio():
    rng = np.random.default_rng(0)
    u = np.concatenate([ex.posicoes(rng.random(10_000), rng.integers(0, 10_000, 5)) for _ in range(400)])
    t = ex.teste_contra_acaso(u)
    assert 0.47 < t["media"] < 0.53 and t["p"] > 0.01


def test_teste_pareado_e_holm():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    assert ex.teste_pareado(a, a)["p"] == 1.0
    assert ex.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_estrutura_do_universo_96000():
    validos = ex.bilhetes_validos(96_000)
    assert validos.sum() == 96_000
    assert validos[0] == 9          # 00000 não vale
    assert validos[6000] == 10 and validos[6001] == 9  # 96000 existe; 96001 não


def test_estrutura_do_universo_desloca_a_metrica_sem_prever_nada():
    """Sorteio uniforme de 00001 a 96000: a referência estrutural fica perto de 0,4875."""
    rng = np.random.default_rng(1)
    numeros = rng.integers(1, 96_001, 40_000)
    u = ex.posicoes(ex.bilhetes_validos(96_000), numeros % 10_000)
    assert u.mean() == pytest.approx(0.4875, abs=0.005)


# ------------------------------------------------------------------ atributos

def test_atributos_extras_sao_opcionais():
    indice = IndiceHistorico(historico_artificial(120))
    padrao = calcular_atributos(indice, np.arange(50), 100)
    ampliado = calcular_atributos(indice, np.arange(50), 100, extras=True)
    assert len(padrao.columns) == 37 and "freq_1_250" not in padrao
    novos = set(ampliado.columns) - set(padrao.columns)
    assert novos == {"freq_1_250", "freq_2_250", "freq_3_250", "freq_4_250",
                     "estabilidade_1", "estabilidade_2", "estabilidade_3"}
    assert (ampliado[padrao.columns] == padrao).all().all()


def test_analise_aponta_duplicados_e_reducao_remove():
    indice = IndiceHistorico(historico_artificial(1200, semente=4))  # > 1000 concursos
    X, y = ex.amostra(indice, 1100, 1200, 15, 1, extras=False)
    analise = ex.analisar_atributos(X, y)
    # Os duplicados encontrados na auditoria (posicao_5_* = freq_1_*) já foram removidos do motor.
    assert not [p for p in analise["pares_correlacionados"] if p["identicos"]]
    mantidos, removidos = ex.reduzir_atributos(X)
    assert "soma_algarismos" not in mantidos
    assert {r["atributo"] for r in removidos} >= set(ex.ESTRUTURAIS)
    assert all(0 <= linha["auc"] <= 1 for linha in analise["utilidade"])


def test_amostra_reproduz_a_do_motor():
    from src.preditivo import modelos
    indice = IndiceHistorico(historico_artificial(150))
    config = modelos.Configuracao(negativos=15, semente=7)
    X1, y1 = modelos.amostra_treino(indice, 50, 150, config, np.random.default_rng(7))
    X2, y2 = ex.amostra(indice, 50, 150, 15, 7, extras=False)
    assert (y1 == y2).all() and X1.equals(X2)


# ------------------------------------------------------------------ sem vazamento

def test_treino_da_variante_nao_ve_o_futuro():
    df = historico_artificial(300, semente=8)
    alterado = df.copy()
    alterado.loc[alterado["concurso"] > 200, "numero"] = "00007"
    colunas = list(calcular_atributos(IndiceHistorico(df), [0], 1).columns)
    notas = []
    for dados in (df, alterado):
        indice = IndiceHistorico(dados)
        pacote = ex.treinar_variante(indice, 200, VARIANTES_TESTE[0], colunas)
        X = calcular_atributos(indice, np.arange(10_000), 200)
        notas.append(ex._notas(pacote, X))
    for chave in notas[0]:
        np.testing.assert_allclose(notas[0][chave], notas[1][chave])


@pytest.fixture(scope="module")
def auditorias():
    """Mesma auditoria com e sem alteração do período de teste final."""
    df = historico_artificial(320, semente=9)
    alterado = df.copy()
    alterado.loc[alterado["concurso"] > 280, "numero"] = "12345"  # só o teste final muda
    kwargs = dict(variantes=VARIANTES_TESTE, teste=40, ajuste=40, bloco=40, periodos=2)
    return ex.executar_auditoria(IndiceHistorico(df), **kwargs), \
        ex.executar_auditoria(IndiceHistorico(alterado), **kwargs)


def test_escolha_nao_consulta_o_teste_final(auditorias):
    original, alterado = auditorias
    assert original["divisao"]["teste_final"] == [281, 320]
    assert original["ajuste"] == alterado["ajuste"]
    assert original["selecao"] == alterado["selecao"]
    assert original["atributos"]["utilidade"] == alterado["atributos"]["utilidade"]
    assert original["teste_final"]["contra_acaso"] != alterado["teste_final"]["contra_acaso"]


def test_relatorio_tem_referencias_comparacoes_e_periodos(auditorias):
    r = auditorias[0]
    contra = r["teste_final"]["contra_acaso"]
    for chave in ex.NOMES_REFERENCIAS:
        assert chave in contra
    assert "padrão do motor (conjunto, peso igual)" in contra
    assert all(0 <= v["p_holm"] <= 1 for k, v in r["teste_final"]["comparacoes"].items())
    assert len(r["teste_final"]["por_periodo"]) == 2
    assert set(next(iter(r["aprovacao"].values()))) == {
        "abaixo_do_acaso", "melhor_que_padrao", "melhor_que_estrutura", "aprovada"}
    assert r["selecao"]["combinacao"].startswith("conj:")


def test_relatorio_markdown(auditorias, tmp_path):
    md = ex.gerar_relatorio(auditorias[0])
    for trecho in ("Divisão cronológica", "Teste final, fora da amostra", "Comparações pareadas",
                   "Critério de aprovação", "Estrutura do universo", "Seleção aleatória",
                   "Frequência histórica", "Frequência recente", "Intervalo de ausência"):
        assert trecho in md
    arquivo = ex.salvar(auditorias[0], tmp_path)
    assert arquivo.exists() and arquivo.suffix == ".json"
