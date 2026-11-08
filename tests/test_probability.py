from fractions import Fraction

import pytest

from src.probability import (
    CHANCE_DERIVADOS_OFICIAL,
    EXTRACOES,
    explicacao_probabilidade,
    formatar_milhar,
    formatar_numero,
    numero_no_universo,
    obter_extracao,
)

OFICIAIS = {
    "quartou": (92_000, 1, False),
    "domingo": (96_000, 1, False),
    "enricou": (84_000, 1, False),
    "natal": (85_000, 2, True),
}


def test_quatro_extracoes_configuradas():
    assert set(EXTRACOES) == set(OFICIAIS)


@pytest.mark.parametrize("chave,esperado", OFICIAIS.items())
def test_probabilidades_oficiais(chave, esperado):
    chance, series, por_serie = esperado
    e = EXTRACOES[chave]
    assert e.chance_um_em == chance
    assert e.probabilidade == Fraction(1, chance)
    assert e.series == series
    assert e.chance_por_serie is por_serie
    assert e.habilitada


@pytest.mark.parametrize("chave", OFICIAIS)
def test_universo_confere_com_quantidade_de_bilhetes(chave):
    e = EXTRACOES[chave]
    # Regulamento item 4: 00000 é ressorteado; acima da quantidade também.
    assert e.numero_minimo == 1
    assert e.numero_maximo == e.bilhetes_por_serie
    assert e.tamanho_universo == e.bilhetes_por_serie
    # A chance oficial publicada é coerente com o universo documentado.
    assert e.tamanho_universo == e.chance_um_em


def test_universos_sao_diferentes():
    assert len({e.numero_maximo for e in EXTRACOES.values()}) == 4


def test_textos_de_chance():
    assert EXTRACOES["quartou"].chance_texto == "1 em 92.000"
    assert EXTRACOES["natal"].chance_texto == "1 em 85.000 (por série)"
    assert CHANCE_DERIVADOS_OFICIAL == "1 em 4,78"


def test_formatacao():
    assert formatar_numero(7) == "00007"
    assert formatar_numero(92000) == "92000"
    assert formatar_milhar(96000) == "96.000"
    with pytest.raises(ValueError):
        formatar_numero(100000)


@pytest.mark.parametrize("numero,chave,esperado", [
    ("00001", "quartou", True),
    ("92000", "quartou", True),
    ("92001", "quartou", False),
    ("00000", "quartou", False),
    ("96000", "domingo", True),
    ("84001", "enricou", False),
    ("85000", "natal", True),
    ("1234", "quartou", False),
    ("12a45", "quartou", False),
])
def test_numero_no_universo(numero, chave, esperado):
    assert numero_no_universo(numero, chave) is esperado


def test_extracao_invalida():
    with pytest.raises(ValueError):
        obter_extracao("lotofacil")


def test_explicacao_menciona_igualdade_de_chance():
    texto = explicacao_probabilidade("natal")
    assert "mesma chance" in texto
    assert "série" in texto
