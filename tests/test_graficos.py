"""Testes das análises dos gráficos (décadas, épocas, distribuição de frequências,
qui-quadrado) e da versão do banco usada pelo cache. Dados ARTIFICIAIS."""

import pandas as pd
import pytest

from src import database as db
from src import statistics as est
from src.generator import gerar_candidatos, parametros_geracao
from src.importer import importar


def _hist(linhas):
    """linhas: lista de (concurso, data, [5 números])."""
    registros = []
    for concurso, data, numeros in linhas:
        for premio, numero in enumerate(numeros, 1):
            registros.append({"concurso": concurso, "data": data, "extracao": "", "origem": "importacao",
                              "premio_zero": False, "premio": premio, "numero": numero})
    return pd.DataFrame(registros)


def test_matriz_digitos():
    m = est.matriz_digitos(pd.Series(["01234", "98765"]))
    assert m.tolist() == [[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]]
    assert est.matriz_digitos(pd.Series([], dtype=str)).shape == (0, 5)


def test_soma_algarismos_vetorizada():
    h = _hist([(1, "2000-01-01", ["00001", "00011", "99999", "12345", "00000"[:4] + "1"])])
    soma = est.distribuicao_soma_algarismos(h).set_index("soma")["observado"].to_dict()
    assert soma == {1: 2, 2: 1, 45: 1, 15: 1}


def test_resumo_decadas_separa_por_decada():
    h = _hist([
        (1, "1962-09-15", ["10000", "20000", "30000", "40000", "50001"]),
        (2, "1975-01-01", ["60000", "70000", "80000", "90000", "00002"]),
        (3, "1979-12-31", ["00003", "00004", "00005", "00006", "00007"]),
    ])
    dec = est.resumo_decadas(h)
    assert list(dec["decada"]) == ["1960–1969", "1970–1979"]
    assert list(dec["concursos"]) == [1, 2]
    assert list(dec["maior_numero"]) == [50001, 90000]
    assert dec["taxa_unidade_teorica"].iloc[0] == pytest.approx(1 - 0.9**5)


def test_primeiro_algarismo_por_decada_em_percentual():
    h = _hist([
        (1, "1962-01-01", ["10000", "10001", "20000", "20001", "30000"]),
        (2, "2001-01-01", ["90000", "90001", "90002", "90003", "90004"]),
    ])
    t = est.primeiro_algarismo_por_decada(h)
    assert list(t.index) == ["1960–1969", "2000–2009"]
    assert t.loc["1960–1969", "1"] == pytest.approx(40.0)
    assert t.loc["2000–2009", "9"] == pytest.approx(100.0)
    assert t.loc["1960–1969"].sum() == pytest.approx(100.0)


def test_distribuicao_contagens_binomial():
    # 200 números com dezenas finais 00..99 duas vezes cada: todas as dezenas saem 2 vezes.
    numeros = [f"{i % 100:05d}" if i % 100 else "00100" for i in range(200)]
    h = _hist([(c + 1, "2000-01-01", numeros[c * 5:(c + 1) * 5]) for c in range(40)])
    d = est.distribuicao_contagens(h, 2)
    assert d.loc[d["vezes"] == 2, "terminacoes_observadas"].item() == 100
    assert d["terminacoes_observadas"].sum() == 100
    assert d["terminacoes_esperadas"].sum() == pytest.approx(100, abs=0.5)
    # Binomial(200, 1/100): P(k = 2) ≈ 0,2720
    assert d.loc[d["vezes"] == 2, "terminacoes_esperadas"].item() == pytest.approx(27.20, abs=0.05)


@pytest.mark.parametrize("gl,critico", [(1, 3.841), (9, 16.919), (99, 123.225), (999, 1073.643)])
def test_valor_critico_qui_quadrado(gl, critico):
    assert est.valor_critico_qui_quadrado(gl) == pytest.approx(critico, abs=1e-3)


def test_resumo_testes_qui_quadrado():
    h = _hist([(c, "2000-01-01", [f"{(c * 5 + i) % 99999 + 1:05d}" for i in range(5)]) for c in range(1, 201)])
    r = est.resumo_testes_qui_quadrado(h)
    assert list(r["teste"]) == ["Unidade (5 prêmios)", "Dezena (5 prêmios)", "Centena (5 prêmios)",
                                "Milhar (5 prêmios)", "Unidade do 1º prêmio"]
    assert r.loc[0, "valido"] and not r.loc[3, "valido"]
    assert (r["razao"] == r["qui_quadrado"] / r["valor_critico_95"]).all()


def test_versao_muda_com_importacao_mas_nao_com_candidatos(tmp_path):
    banco = tmp_path / "v.db"
    db.inicializar(banco)
    v0 = db.versao_historico(banco)
    db.salvar_candidatos("quartou", gerar_candidatos("quartou", 5), parametros_geracao("quartou", 5), banco)
    assert db.versao_historico(banco) == v0
    r = importar("concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5\n1;01/01/2000;1;2;3;4;5\n")
    db.registrar_importacao(r.validos, "a.csv", 1, 0, 0, r.relatorio(), banco)
    v1 = db.versao_historico(banco)
    assert v1 != v0
    db.apagar_historico(banco)
    assert db.versao_historico(banco) not in (v0, v1)
