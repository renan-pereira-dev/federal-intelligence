"""Testes das análises descritivas. Dados ARTIFICIAIS."""

import pandas as pd
import pytest

from src import statistics as est


def _hist(numeros, extracao="quartou"):
    return pd.DataFrame({
        "concurso": [i // 5 + 1 for i in range(len(numeros))],
        "data": ["2020-01-01"] * len(numeros),
        "extracao": [extracao] * len(numeros),
        "premio": [i % 5 + 1 for i in range(len(numeros))],
        "numero": numeros,
    })


def test_nao_sombreia_modulo_statistics_do_python():
    import statistics

    assert hasattr(statistics, "mean")
    assert statistics.mean([1, 2, 3]) == 2


def test_frequencia_por_posicao():
    tabela = est.frequencia_por_posicao(_hist(["01234", "01299", "90000", "00001", "11111"]))
    assert tabela.loc["Dezena de milhar", "0"] == 3
    assert tabela.loc["Unidade", "1"] == 2
    assert int(tabela.values.sum()) == 25


def test_esperado_do_primeiro_algarismo_nao_e_uniforme():
    prop = est._proporcao_por_posicao(92_000)
    # 00001..92000: algarismo 9 na 1ª posição só de 90000 a 92000 (2001 números).
    assert prop[0][9] == pytest.approx(2001 / 92_000)
    assert prop[0][1] == pytest.approx(10_000 / 92_000)
    assert prop[4].sum() == pytest.approx(1.0)


def test_esperado_indisponivel_sem_extracao():
    h = _hist(["01234"] * 5, extracao="")
    assert est.esperado_por_posicao(h) is None
    assert "esperado" not in est.frequencia_algarismos(h).columns


def test_terminacoes_e_repeticoes():
    h = _hist(["01234", "11234", "01234", "55555", "00034"])
    t2 = est.terminacoes(h, 2)
    assert t2.iloc[0]["terminacao"] == "34" and t2.iloc[0]["observado"] == 4
    rep = est.repeticoes(h)
    assert rep["numeros_repetidos"].iloc[0]["numero"] == "01234"
    assert len(rep["concursos_com_repeticao_interna"]) == 1


def test_frequencia_candidato():
    h = _hist(["01234", "11234", "01234", "55555", "00034"])
    f = est.frequencia_candidato("01234", h)
    assert f == {"exato": 2, "milhar": 3, "centena": 3, "dezena": 4, "total_numeros": 5}
    assert est.frequencia_candidato("01234", None) is None
