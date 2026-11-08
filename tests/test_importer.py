"""Testes do importador. Todos os dados aqui são ARTIFICIAIS, só para teste."""

import pytest

from src.importer import ErroImportacao, importar, ler_csv, normalizar_numero

CABECALHO = "concurso;data;extracao;premio_1;premio_2;premio_3;premio_4;premio_5\n"


def test_importa_csv_valido_preservando_zeros():
    csv = CABECALHO + "1;01/02/2020;quartou;00123;01234;12345;00001;90000\n"
    r = importar(csv)
    assert r.linhas_validas == 1 and r.linhas_rejeitadas == 0
    linha = r.validos.iloc[0]
    assert linha["premio_1"] == "00123"
    assert linha["premio_4"] == "00001"
    assert linha["data"] == "2020-02-01"
    assert linha["extracao"] == "quartou"


def test_restaura_zeros_perdidos_com_aviso():
    csv = CABECALHO + "2;2021-03-04;;123;1234;12345;1;90000\n"
    r = importar(csv)
    assert r.linhas_validas == 1
    assert list(r.validos.iloc[0][["premio_1", "premio_2", "premio_4"]]) == ["00123", "01234", "00001"]
    assert any("zeros" in a for a in r.avisos)


def test_aceita_separador_virgula_e_bytes_latin1():
    texto = "Concurso,Data,Extração,1º Prêmio,2º Prêmio,3º Prêmio,4º Prêmio,5º Prêmio\n"
    texto += "3,10/12/2025,Natal,11111,22222,33333,44444,55555\n"
    r = importar(texto.encode("cp1252"))
    assert r.linhas_validas == 1
    assert r.validos.iloc[0]["extracao"] == "natal"


@pytest.mark.parametrize("linha,motivo", [
    ("4;01/01/2020;;12a45;11111;22222;33333;44444", "caracteres inválidos"),
    ("5;01/01/2020;;123456;11111;22222;33333;44444", "não começa com 0"),
    ("6;01/01/2020;;00000;11111;22222;33333;44444", "00000"),
    ("7;01/01/2020;;;11111;22222;33333;44444", "vazio"),
    ("8;31/02/2020;;12345;11111;22222;33333;44444", "data inválida"),
    ("9;31/12/1961;;12345;11111;22222;33333;44444", "anterior a 1962"),
    ("10;01/01/2999;;12345;11111;22222;33333;44444", "no futuro"),
    ("abc;01/01/2020;;12345;11111;22222;33333;44444", "concurso inválido"),
    ("0;01/01/2020;;12345;11111;22222;33333;44444", "concurso inválido"),
    ("11;01/01/2020;megasena;12345;11111;22222;33333;44444", "extração desconhecida"),
])
def test_rejeita_campos_invalidos(linha, motivo):
    r = importar(CABECALHO + linha + "\n")
    assert r.linhas_validas == 0
    assert r.linhas_rejeitadas == 1
    assert motivo in r.rejeitados.iloc[0]["motivo"]


def test_duplicidades_no_arquivo():
    linha = "20;01/01/2020;;12345;11111;22222;33333;44444\n"
    conflito = "20;01/01/2020;;99999;11111;22222;33333;44444\n"
    r = importar(CABECALHO + linha + linha + conflito)
    assert r.linhas_validas == 1
    assert r.duplicados == 2
    motivos = list(r.rejeitados["motivo"])
    assert "linha duplicada no arquivo" in motivos
    assert "concurso repetido no arquivo com dados diferentes" in motivos


def test_concurso_ja_existente_no_banco():
    r = importar(CABECALHO + "30;01/01/2020;;12345;11111;22222;33333;44444\n", concursos_existentes={30})
    assert r.linhas_validas == 0
    assert "já existe" in r.rejeitados.iloc[0]["motivo"]


def test_repeticao_de_numero_no_mesmo_concurso_e_permitida():
    # Regulamento, item 11: é válida a repetição de número sorteado na mesma extração.
    r = importar(CABECALHO + "40;01/01/2020;;12345;12345;22222;33333;44444\n")
    assert r.linhas_validas == 1


def test_aviso_numero_acima_do_universo_atual():
    r = importar(CABECALHO + "41;01/01/2020;quartou;95000;11111;22222;33333;44444\n")
    assert r.linhas_validas == 1
    assert any("acima do universo" in a for a in r.avisos)


def test_sem_coluna_obrigatoria():
    with pytest.raises(ErroImportacao, match="premio_5"):
        ler_csv("concurso;data;premio_1;premio_2;premio_3;premio_4\n1;01/01/2020;1;2;3;4\n")


def test_arquivo_vazio():
    with pytest.raises(ErroImportacao):
        importar(b"")


def test_extracao_opcional():
    csv = "concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5\n50;01/01/2015;1;2;3;4;5\n"
    r = importar(csv)
    assert r.linhas_validas == 1
    assert r.validos.iloc[0]["extracao"] == ""


def test_normalizar_numero():
    assert normalizar_numero(" 01234 ") == ("01234", None, ())
    assert normalizar_numero("7") == ("00007", None, ("zeros_restaurados",))
    assert normalizar_numero("1234567")[0] is None
    assert normalizar_numero("-12")[0] is None
    assert normalizar_numero("1.5")[0] is None
