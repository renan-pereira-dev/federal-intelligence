"""Testes das probabilidades teóricas, justificativas e taxas históricas.
Todos os históricos aqui são ARTIFICIAIS."""

import json
import math
from fractions import Fraction

import pandas as pd
import pytest

from src import statistics as est
from src.probability import (
    EXTRACOES,
    contagem_terminacao,
    justificativa_terminacao,
    probabilidade_algum_premio,
    probabilidade_terminacao,
)


# ------------------------------------------------------------- teoria exata

@pytest.mark.parametrize("chave", list(EXTRACOES))
@pytest.mark.parametrize("digitos", [1, 2, 3])
def test_terminacoes_exatas_nos_planos_atuais(chave, digitos):
    maximo = EXTRACOES[chave].numero_maximo
    assert probabilidade_terminacao(maximo, digitos) == Fraction(1, 10**digitos)
    # Conferência por força bruta numa terminação qualquer.
    alvo = 7 % 10**digitos
    contagem = sum(1 for n in range(1, maximo + 1) if n % 10**digitos == alvo)
    assert contagem == contagem_terminacao(maximo, digitos, alvo)


def test_milhar_varia_quando_nao_ha_multiplo_de_10000():
    assert probabilidade_terminacao(92_000, 4) is None
    # 1..92000: final 1999 aparece 10 vezes (01999 ... 91999); final 5000 aparece 9.
    assert probabilidade_terminacao(92_000, 4, 1999) == Fraction(10, 92_000)
    assert probabilidade_terminacao(92_000, 4, 5000) == Fraction(9, 92_000)
    assert probabilidade_terminacao(92_000, 4, 0) == Fraction(9, 92_000)
    soma = sum(contagem_terminacao(92_000, 4, t) for t in range(10_000))
    assert soma == 92_000


def test_probabilidade_algum_dos_cinco():
    assert probabilidade_algum_premio(Fraction(1, 10)) == 1 - Fraction(9, 10) ** 5
    assert float(probabilidade_algum_premio(Fraction(1, 100))) == pytest.approx(0.04900995)


def test_justificativas_mencionam_a_conta():
    assert "múltiplo" in justificativa_terminacao(92_000, 3)
    assert "1 em 1.000" in justificativa_terminacao(92_000, 3)
    assert "varia" in justificativa_terminacao(92_000, 4)


# ------------------------------------------------------------- qui-quadrado

@pytest.mark.parametrize("x,gl,p", [
    (3.841459, 1, 0.05),
    (16.918978, 9, 0.05),
    (21.665994, 9, 0.01),
    (123.225221, 99, 0.05),
    (9.0, 9, 0.43727),
    (0.0, 5, 1.0),
])
def test_p_valor_qui_quadrado(x, gl, p):
    assert est.p_valor_qui_quadrado(x, gl) == pytest.approx(p, abs=1e-4)


# ---------------------------------------------------------- históricos fake

def _hist(numeros_por_concurso, zero=None, anos=None):
    linhas = []
    for c, numeros in enumerate(numeros_por_concurso, start=1):
        ano = anos[c - 1] if anos else 2000
        for premio, numero in enumerate(numeros, start=1):
            linhas.append({
                "concurso": c, "data": f"{ano}-01-01", "extracao": "", "origem": "importacao",
                "premio_zero": bool(zero and c in zero), "premio": premio, "numero": numero,
            })
    return pd.DataFrame(linhas)


def test_teste_terminacoes_com_dados_perfeitamente_uniformes():
    # 100 concursos; os finais dos 500 números percorrem 0..9 igualmente.
    concursos = [[f"{(c * 5 + i) % 100000:05d}" for i in range(5)] for c in range(1, 101)]
    t = est.teste_terminacoes(_hist(concursos), 1)
    assert t["n"] == 500
    assert t["qui_quadrado"] == pytest.approx(0.0)
    assert t["p_valor"] == pytest.approx(1.0)
    assert t["fora_da_faixa"] == 0
    assert t["teste_valido"]


def test_teste_terminacoes_detecta_desvio_grande():
    concursos = [["00001", "00011", "00021", "00031", "00041"]] * 100  # tudo termina em 1
    t = est.teste_terminacoes(_hist(concursos), 1)
    assert t["p_valor"] < 1e-10


def test_teste_apenas_primeiro_premio():
    concursos = [["00003", "00001", "00001", "00001", "00001"]] * 20
    t = est.teste_terminacoes(_hist(concursos), 1, apenas_primeiro=True)
    assert t["n"] == 20
    assert t["tabela"].loc[3, "observado"] == 20


def test_milhar_sem_amostra_suficiente_marca_teste_invalido():
    t = est.teste_terminacoes(_hist([["12345"] * 5] * 10), 4)
    assert not t["teste_valido"]


def test_taxa_coincidencia():
    # Cada concurso tem 5 unidades distintas: taxa histórica = 5/10 = 50%.
    concursos = [["00000", "00001", "00002", "00003", "00004"][::-1]] * 10
    concursos = [[n if n != "00000" else "00010" for n in c] for c in concursos]
    t = est.taxa_coincidencia(_hist(concursos), 1)
    assert t["taxa_observada"] == pytest.approx(0.5)
    assert t["taxa_teorica"] == pytest.approx(1 - 0.9**5)
    # Com todos os concursos iguais o erro padrão é 0; o resultado não pode quebrar.
    assert t["concursos"] == 10


def test_taxa_coincidencia_conta_repeticao_uma_vez():
    t = est.taxa_coincidencia(_hist([["00001", "00011", "00021", "00031", "00041"]] * 3), 1)
    assert t["taxa_observada"] == pytest.approx(0.1)


def test_filtro_premio_zero_e_periodo():
    h = _hist([["00001"] * 5, ["00002"] * 5, ["00003"] * 5], zero={2}, anos=[1962, 2000, 2026])
    assert est.filtrar_historico(h, incluir_premio_zero=True)["concurso"].nunique() == 3
    assert set(est.filtrar_historico(h, incluir_premio_zero=False)["concurso"]) == {1, 3}
    assert set(est.filtrar_historico(h, ano_inicio=1990, ano_fim=2010)["concurso"]) == {2}


def test_maior_numero_por_ano():
    h = _hist([["00001", "99999", "00003", "00004", "00005"], ["00010"] * 5], anos=[1970, 2026])
    eras = est.maior_numero_por_ano(h)
    assert list(eras["maior_numero"]) == [99999, 10]


def test_taxa_candidatos():
    conf = pd.DataFrame({"conferencia": [
        json.dumps({"extracao": "quartou", "premio_principal": [1], "unidade_primeiro": False}),
        json.dumps({"extracao": "quartou", "premio_principal": [], "unidade_primeiro": True}),
        json.dumps({"extracao": "quartou", "premio_principal": [], "terminacoes": {}}),
    ]})
    r = est.taxa_candidatos(conf)
    assert r["conferidos"] == 3 and r["exatos"] == 1 and r["unidade_primeiro"] == 1
    assert r["com_alguma_coincidencia"] == 2
    assert r["esperado_exatos_max"] == pytest.approx(3 * 5 / 92_000)
    assert est.taxa_candidatos(pd.DataFrame()) is None
    assert not math.isnan(r["esperado_unidade_primeiro"])
