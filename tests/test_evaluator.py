"""Testes da conferência. Os números premiados abaixo são ARTIFICIAIS ou vêm do
exemplo publicado na página oficial da Loteria Federal (95655, 39888, 62547,
01339, 74262), usado lá apenas para ilustrar as regras."""

import pytest

from src.evaluator import conferir, conferir_lista

PREMIOS = ["95655", "39888", "62547", "01339", "74262"]


@pytest.mark.parametrize("posicao", range(1, 6))
def test_acerto_exato_em_cada_um_dos_cinco_premios(posicao):
    candidato = PREMIOS[posicao - 1]
    r = conferir(candidato, PREMIOS, "domingo")
    assert r.premio_principal == [posicao]
    assert r.acertou_principal


def test_repeticao_no_mesmo_concurso_reconhece_dois_premios():
    premios = ["12345", "12345", "22222", "33333", "44444"]
    r = conferir("12345", premios, "domingo")
    assert r.premio_principal == [1, 2]


def test_milhar_centena_dezena_sao_coincidencias_nao_premio_principal():
    r = conferir("05655", PREMIOS, "domingo")  # milhar do 1º
    assert not r.acertou_principal
    assert r.terminacoes == {1: "milhar"}
    assert "não é prêmio principal" in r.resumo()

    assert conferir("00888", PREMIOS, "domingo").terminacoes == {2: "centena"}
    assert conferir("00047", PREMIOS, "domingo").terminacoes == {3: "dezena"}


def test_guarda_apenas_a_terminacao_mais_longa():
    r = conferir("11339", PREMIOS, "domingo")  # milhar do 4º
    assert r.terminacoes == {4: "milhar"}


def test_aproximacoes_do_primeiro_premio():
    assert conferir("95654", PREMIOS, "domingo").aproximacao.startswith("anterior")
    assert conferir("95656", PREMIOS, "domingo").aproximacao.startswith("posterior")


def test_dezenas_finais_exemplo_oficial():
    # Exemplo da página oficial: 52, 53, 54, 56, 57, 58.
    for dezena in ["52", "53", "54", "56", "57", "58"]:
        r = conferir("10" + "0" + dezena, PREMIOS, "domingo")
        assert r.dezena_final_vizinha, dezena
    for dezena in ["51", "55", "59"]:
        assert not conferir("100" + dezena, PREMIOS, "domingo").dezena_final_vizinha


def test_dezena_final_exclui_quem_ganhou_aproximacao():
    r = conferir("95656", PREMIOS, "domingo")
    assert r.aproximacao is not None
    assert r.dezena_final_vizinha is False


def test_unidade_do_primeiro_premio():
    assert conferir("00005", PREMIOS, "domingo").unidade_primeiro
    assert not conferir("00006", PREMIOS, "domingo").unidade_primeiro


def test_nenhuma_coincidencia():
    r = conferir("12340", PREMIOS, "domingo")
    assert r.resumo() == "Nenhuma coincidência"
    assert r.itens() == []


def test_casos_nao_definidos_nao_sao_avaliados():
    r = conferir("00002", ["00001", "11111", "22222", "33333", "44444"], "quartou")
    assert any("Aproximação anterior" in t for t in r.nao_avaliado)
    assert any("Dezenas finais" in t for t in r.nao_avaliado)
    assert r.aproximacao == "posterior, 00002"

    r = conferir("91999", ["92000", "11111", "22222", "33333", "44444"], "quartou")
    assert any("Aproximação posterior" in t for t in r.nao_avaliado)


def test_observacao_de_serie_no_natal_e_fora_do_universo():
    r = conferir("90000", PREMIOS, "natal")
    textos = " ".join(r.observacoes)
    assert "série" in textos
    assert "fora do universo" in textos


@pytest.mark.parametrize("premios", [
    ["1234", "11111", "22222", "33333", "44444"],
    ["00000", "11111", "22222", "33333", "44444"],
    ["11111", "22222", "33333", "44444"],
])
def test_entradas_invalidas(premios):
    with pytest.raises(ValueError):
        conferir("12345", premios, "domingo")


def test_conferir_lista_e_serializacao():
    resultados = conferir_lista(["95655", "12340"], PREMIOS, "domingo")
    d = resultados[0].para_dict()
    assert d["premio_principal"] == [1]
    assert d["itens"]
