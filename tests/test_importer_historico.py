"""Testes do formato de histórico completo (bilhete_N, apóstrofo, 6 dígitos,
valores, metadados, prêmio zero e lacunas). Todos os dados são ARTIFICIAIS."""

import json
from datetime import date

import pytest

from src.importer import (
    AJUSTE_APOSTROFO,
    AJUSTE_SEIS_DIGITOS,
    importar,
    normalizar_numero,
)

CAB = (
    '"concurso";"data";"bilhete_1";"valor_nominal_1";"valor_original_1";'
    '"bilhete_2";"valor_nominal_2";"bilhete_3";"valor_nominal_3";'
    '"bilhete_4";"valor_nominal_4";"bilhete_5";"valor_nominal_5";"fonte";"consultado_em"\n'
)


def _linha(concurso, data, numeros, valores=("100", "50", "40", "30", "20"), consultado="2026-01-01"):
    b = numeros
    v = valores
    return (
        f'"{concurso}";"{data}";"\'{b[0]}";"{v[0]}";"R${v[0]},00";"\'{b[1]}";"{v[1]}";'
        f'"\'{b[2]}";"{v[2]}";"\'{b[3]}";"{v[3]}";"\'{b[4]}";"{v[4]}";"https://exemplo.invalido";'
        f'"{consultado}"\n'
    )


NUMS = ("012345", "000001", "099999", "050000", "001000")


@pytest.mark.parametrize("entrada,saida,ajustes", [
    ("'012345", "12345", (AJUSTE_APOSTROFO, AJUSTE_SEIS_DIGITOS)),
    ("012345", "12345", (AJUSTE_SEIS_DIGITOS,)),
    ("'01234", "01234", (AJUSTE_APOSTROFO,)),
    ("000001", "00001", (AJUSTE_SEIS_DIGITOS,)),
])
def test_normaliza_formatos_do_historico(entrada, saida, ajustes):
    numero, erro, feitos = normalizar_numero(entrada)
    assert erro is None
    assert numero == saida
    assert feitos == ajustes


@pytest.mark.parametrize("entrada", ["'112345", "112345", "'000000", "''01234", "'"])
def test_recusa_formatos_invalidos(entrada):
    numero, erro, _ = normalizar_numero(entrada)
    assert numero is None and erro


def test_importa_formato_bilhete_e_preserva_zeros():
    r = importar(CAB + _linha(1, "1962-09-15", NUMS))
    assert r.linhas_validas == 1 and r.linhas_rejeitadas == 0
    linha = r.validos.iloc[0]
    assert [linha[f"premio_{i}"] for i in range(1, 6)] == ["12345", "00001", "99999", "50000", "01000"]
    assert r.ajustes == {AJUSTE_APOSTROFO: 5, AJUSTE_SEIS_DIGITOS: 5}


def test_aceita_datas_desde_1962_e_ate_hoje():
    csv = CAB + _linha(1, "1962-09-15", NUMS) + _linha(2, "2026-11-04", NUMS)
    r = importar(csv, hoje=date(2026, 11, 7))
    assert r.linhas_validas == 2
    r = importar(CAB + _linha(3, "2026-11-08", NUMS), hoje=date(2026, 11, 7))
    assert r.linhas_validas == 0 and "futuro" in r.rejeitados.iloc[0]["motivo"]


def test_guarda_dados_extras_sem_perder_informacao():
    csv = CAB + _linha(1, "2020-01-01", NUMS, consultado="2026-01-01") + _linha(
        2, "2020-01-04", NUMS, valores=("200", "50", "40", "30", "20"), consultado="2026-01-02")
    r = importar(csv)
    # Constantes no arquivo inteiro: guardadas uma vez nos metadados.
    assert r.metadados_arquivo == {
        "valor_nominal_2": "50", "valor_nominal_3": "40", "valor_nominal_4": "30",
        "valor_nominal_5": "20", "fonte": "https://exemplo.invalido",
    }
    # Variáveis: guardadas em cada concurso.
    originais = json.loads(r.validos.iloc[1]["dados_originais"])
    assert originais == {"valor_nominal_1": "200", "valor_original_1": "R$200,00", "consultado_em": "2026-01-02"}
    assert set(r.colunas_extras_por_linha) == {"valor_nominal_1", "valor_original_1", "consultado_em"}


def test_marca_concursos_com_premio_zero():
    zero = ("0", "0", "0", "0", "0")
    csv = (
        CAB
        + _linha(10, "2026-07-01", NUMS, valores=zero)
        + _linha(11, "2026-07-04", NUMS, valores=("0", "0", "0", "0", "5"))  # não é tudo zero
        + _linha(12, "2026-08-01", NUMS)
    )
    r = importar(csv)
    assert r.linhas_validas == 3
    assert r.concursos_premio_zero == [10]
    assert list(r.validos["premio_zero"]) == [True, False, False]


def test_valor_vazio_nao_e_zero():
    r = importar(CAB + _linha(10, "2026-07-01", NUMS, valores=("", "", "", "", "")))
    assert r.concursos_premio_zero == []


def test_sem_colunas_de_valor_nao_marca_premio_zero():
    csv = "concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5\n1;01/01/2020;1;2;3;4;5\n"
    r = importar(csv)
    assert r.concursos_premio_zero == []


def test_lacunas_sao_informadas_e_nao_preenchidas():
    csv = CAB + _linha(1, "1962-09-15", NUMS) + _linha(2, "1962-09-22", NUMS) + _linha(6, "1962-10-20", NUMS)
    r = importar(csv)
    assert r.linhas_validas == 3
    assert r.lacunas == [3, 4, 5]


def test_lacunas_consideram_concursos_ja_gravados():
    r = importar(CAB + _linha(1, "1962-09-15", NUMS) + _linha(4, "1962-10-06", NUMS), concursos_existentes={2})
    assert r.lacunas == [3]


def test_relatorio_serializavel():
    r = importar(CAB + _linha(1, "1962-09-15", NUMS))
    texto = json.dumps(r.relatorio(), ensure_ascii=False)
    assert "lacunas" in texto and "metadados_arquivo" in texto
