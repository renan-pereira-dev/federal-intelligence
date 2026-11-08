import random

import pytest

from src.generator import (
    QUANTIDADES_PERMITIDAS,
    ExtracaoNaoHabilitada,
    gerar_candidatos,
    parametros_geracao,
)
from src.probability import EXTRACOES


@pytest.mark.parametrize("chave", list(EXTRACOES))
@pytest.mark.parametrize("quantidade", QUANTIDADES_PERMITIDAS)
def test_gera_dentro_do_universo_sem_duplicidade(chave, quantidade):
    extracao = EXTRACOES[chave]
    for _ in range(20):
        numeros = gerar_candidatos(chave, quantidade)
        assert len(numeros) == quantidade
        assert len(set(numeros)) == quantidade
        for n in numeros:
            assert isinstance(n, str) and len(n) == 5 and n.isdigit()
            assert extracao.numero_minimo <= int(n) <= extracao.numero_maximo


def test_zeros_a_esquerda_preservados():
    # Com uma fonte determinística, força números pequenos no resultado.
    class FonteBaixa(random.Random):
        def sample(self, populacao, k):
            return list(populacao)[:k]

    numeros = gerar_candidatos("quartou", 5, rng=FonteBaixa())
    assert numeros == ["00001", "00002", "00003", "00004", "00005"]


def test_nunca_gera_00000_nem_acima_do_maximo():
    for chave, extracao in EXTRACOES.items():
        vistos = set()
        rng = random.Random(1234)
        for _ in range(500):
            vistos.update(gerar_candidatos(chave, 20, rng=rng))
        assert "00000" not in vistos
        assert max(int(n) for n in vistos) <= extracao.numero_maximo


def test_extremos_do_universo_sao_alcancaveis():
    class FonteAlta(random.Random):
        def sample(self, populacao, k):
            return list(populacao)[-k:]

    numeros = gerar_candidatos("enricou", 5, rng=FonteAlta())
    assert numeros[-1] == "84000"


def test_distribuicao_aproximadamente_uniforme():
    # Teste estatístico simples: a média das amostras fica perto do centro.
    rng = random.Random(42)
    valores = [int(n) for _ in range(2000) for n in gerar_candidatos("domingo", 20, rng=rng)]
    media = sum(valores) / len(valores)
    centro = (1 + 96_000) / 2
    erro_padrao = (96_000 / (12 ** 0.5)) / (len(valores) ** 0.5)
    assert abs(media - centro) < 5 * erro_padrao


@pytest.mark.parametrize("quantidade", [0, 1, 3, 15, 21, 100])
def test_quantidade_invalida(quantidade):
    with pytest.raises(ValueError):
        gerar_candidatos("quartou", quantidade)


def test_extracao_desconhecida():
    with pytest.raises(ValueError):
        gerar_candidatos("mega", 5)


def test_extracao_desabilitada(monkeypatch):
    from dataclasses import replace

    import src.probability as prob

    monkeypatch.setitem(prob.EXTRACOES, "quartou", replace(prob.EXTRACOES["quartou"], habilitada=False))
    with pytest.raises(ExtracaoNaoHabilitada):
        gerar_candidatos("quartou", 5)


def test_parametros_registram_que_nao_usa_historico():
    p = parametros_geracao("natal", 10)
    assert p["usa_historico"] is False
    assert p["universo_min"] == "00001"
    assert p["universo_max"] == "85000"
