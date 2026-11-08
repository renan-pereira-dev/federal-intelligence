"""Monitoramento da justiça do sorteio. Somente dados ARTIFICIAIS."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import monitoramento as mo
from tests.test_preditivo import gravar_historico, historico_artificial

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "monitoramento.db"))


def justo(rng, n=mo.BLOCO):
    return rng.integers(1, 96_001, (n, 5))


# ------------------------------------------------------------------ testes por bloco

def test_bloco_justo_tem_os_seis_testes():
    r = mo.testar_bloco(justo(np.random.default_rng(1)))
    assert set(r) == set(mo.TESTES)
    assert [r[t]["gl"] for t in ("unidade", "dezena", "posicao_3", "posicao_4", "homogeneidade_premios")] == \
        [9, 99, 9, 9, 36]
    assert all(0 <= v["p"] <= 1 and v["p"] <= v["p_holm"] <= 1 for v in r.values())
    assert r["independencia"]["pares"] == (mo.BLOCO - 1) * 5
    assert sum(r["unidade"]["observados"]) == 5 * mo.BLOCO


def test_detecta_unidade_viciada():
    rng = np.random.default_rng(2)
    nums = justo(rng)
    vicio = rng.random(nums.shape) < 0.08
    nums[vicio] = nums[vicio] // 10 * 10 + 7          # 8% a mais de finais 7
    nums[nums == 0] = 7
    r = mo.testar_bloco(nums)
    assert r["unidade"]["p_holm"] < 1e-4
    assert r["unidade"]["maior_desvio"]["categoria"] == 7


def test_detecta_algarismo_da_terceira_posicao():
    rng = np.random.default_rng(3)
    nums = justo(rng)
    vicio = rng.random(nums.shape) < 0.1
    nums[vicio] = nums[vicio] // 1000 * 1000 + 300 + nums[vicio] % 100  # força o algarismo 3 na 3ª posição
    assert mo.testar_bloco(nums)["posicao_3"]["p_holm"] < 1e-4


def test_detecta_premio_com_comportamento_diferente():
    rng = np.random.default_rng(4)
    nums = justo(rng)
    nums[:, 0] = nums[:, 0] // 2 * 2 + 1  # o 1º prêmio só termina em número ímpar
    nums[nums > 96_000] -= 2
    assert mo.testar_bloco(nums)["homogeneidade_premios"]["p_holm"] < 1e-4


def test_detecta_dependencia_entre_concursos():
    rng = np.random.default_rng(5)
    nums = justo(rng)
    for i in range(1, len(nums)):  # metade das unidades copia o concurso anterior
        copia = rng.random(5) < 0.5
        nums[i, copia] = nums[i, copia] // 10 * 10 + nums[i - 1, copia] % 10
    nums[nums == 0] = 10
    r = mo.testar_bloco(nums)
    assert r["independencia"]["p_holm"] < 1e-6
    assert r["independencia"]["repeticoes"] > 2 * r["independencia"]["esperado"]


def test_holm():
    assert mo.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


# ------------------------------------------------------------------ calibração

def test_calibracao_em_sorteios_justos():
    """Num sorteio justo: ~5% dos blocos em atenção e alertas raros (sem alarme falso frequente)."""
    rng = np.random.default_rng(6)
    blocos_com_atencao, pares_com_alerta, pares, p_unidade = 0, 0, 0, []
    for _ in range(60):
        blocos = []
        anterior = None
        for b in range(8):
            nums = justo(rng)
            testes = mo.testar_bloco(nums, anterior)
            anterior = nums[-1]
            blocos.append({"de": b, "ate": b, "testes": testes})
            blocos_com_atencao += any(v["p_holm"] < mo.ALFA for v in testes.values())
            p_unidade.append(testes["unidade"]["p"])
        pares += len(blocos) - 1
        pares_com_alerta += len({tuple(a["blocos"]) for a in mo.classificar(blocos)["alertas"]})
    assert blocos_com_atencao / 480 < 0.10            # Holm controla ~5% por bloco
    assert pares_com_alerta / pares < 0.02            # alerta exige repetição: raro por acaso
    assert 0.02 < np.mean(np.array(p_unidade) < 0.05) < 0.09  # valores-p bem calibrados


# ------------------------------------------------------------------ regras de situação

def _blocos(*ps_por_bloco):
    """Cada argumento: dict teste -> (p, p_holm) para os testes alterados; o resto fica p=0,5."""
    blocos = []
    for i, alterados in enumerate(ps_por_bloco):
        testes = {t: {"p": 0.5, "p_holm": 1.0} for t in mo.TESTES}
        for t, (p, ph) in alterados.items():
            testes[t] = {"p": p, "p_holm": ph}
        blocos.append({"de": i * 10, "ate": i * 10 + 9, "testes": testes})
    return blocos


def test_situacao_normal():
    assert mo.classificar(_blocos({}, {}, {}))["situacao"] == mo.NORMAL


def test_atencao_sem_repeticao():
    r = mo.classificar(_blocos({}, {}, {"unidade": (0.001, 0.006)}))
    assert r["situacao"] == mo.ATENCAO and r["atencao"] == ["unidade"] and r["alertas"] == []


def test_alerta_com_repeticao_em_blocos_seguidos():
    r = mo.classificar(_blocos({}, {"dezena": (0.03, 0.18)}, {"dezena": (0.001, 0.006)}))
    assert r["situacao"] == mo.ALERTA
    assert r["alertas"][0]["teste"] == "dezena" and r["alertas"][0]["blocos"] == [1, 2]


@pytest.mark.parametrize("blocos", [
    _blocos({"dezena": (0.001, 0.006)}, {}, {"dezena": (0.001, 0.006)}),      # não são seguidos
    _blocos({}, {"dezena": (0.03, 0.18)}, {"dezena": (0.04, 0.24)}),          # nenhum passa no Holm
    _blocos({}, {"dezena": (0.001, 0.006)}, {"unidade": (0.001, 0.006)}),     # testes diferentes
])
def test_sem_alerta_quando_nao_ha_repeticao_valida(blocos):
    assert mo.classificar(blocos)["alertas"] == []


def test_alerta_antigo_nao_muda_a_situacao_atual():
    r = mo.classificar(_blocos({"unidade": (0.001, 0.006)}, {"unidade": (0.01, 0.06)}, {}, {}))
    assert len(r["alertas"]) == 1 and r["situacao"] == mo.NORMAL


# ------------------------------------------------------------------ histórico inteiro

def test_blocos_alinhados_ao_mais_recente():
    m = mo.monitorar(historico_artificial(620, semente=7))
    assert len(m["blocos"]) == 2 and m["descartados"] == 120
    assert m["blocos"][-1]["ate"] == 620 and m["blocos"][0]["de"] == 121
    assert m["resumo"]["testes"] == 12


def test_historico_curto_ou_vazio():
    assert mo.monitorar(historico_artificial(100))["blocos"] == []
    assert mo.monitorar(pd.DataFrame())["situacao"] is None


def test_concurso_incompleto_fica_de_fora():
    df = historico_artificial(10)
    df = df[~((df["concurso"] == 5) & (df["premio"] == 3))]
    concursos, _, numeros = mo.matriz(df)
    assert 5 not in concursos and numeros.shape == (9, 5)


def test_desvio_persistente_vira_alerta():
    def gerador(rng):
        n = rng.integers(1, 96_001, 5)
        vicio = rng.random(5) < 0.12
        n[vicio] = n[vicio] // 10 * 10 + 3
        return np.where(n == 0, 3, n)

    justo_df = historico_artificial(250, semente=8)
    viciado = historico_artificial(500, semente=9, gerador=gerador)
    viciado["concurso"] += 250
    m = mo.monitorar(pd.concat([justo_df, viciado], ignore_index=True))
    assert m["situacao"] == mo.ALERTA
    assert any(a["teste"] == "unidade" and a["blocos"] == [1, 2] for a in m["alertas"])


# ------------------------------------------------------------------ interface

def _abrir():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Monitoramento do sorteio").run()
    assert not at.exception, at.exception
    return at


def test_interface_sem_dados():
    at = _abrir()
    assert any("pelo menos 250" in w.value for w in at.warning)


def test_interface_com_historico():
    gravar_historico(historico_artificial(520, semente=10))
    at = _abrir()
    assert [t.label for t in at.tabs] == ["Bloco mais recente", "Ao longo do tempo", "Unidade por bloco",
                                          "Alertas", "Como funciona"]
    html = " ".join(m.value for m in at.markdown)
    assert any(rotulo in html for rotulo in ("Normal", "Atenção", "Alerta"))
    assert at.metric[0].value == "2"
    assert len(at.get("plotly_chart")) == 3
