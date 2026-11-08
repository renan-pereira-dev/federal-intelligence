"""Tela "Previsão do próximo sorteio" no AppTest. Banco temporário e dados ARTIFICIAIS."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_preditivo import gravar_historico, historico_artificial

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGINA = "Previsão do próximo sorteio"


@pytest.fixture(autouse=True)
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "app_previsao.db"))
    monkeypatch.setenv("FI_MOTOR_CONFIG", "rapido")


def _abrir() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value(PAGINA).run()
    assert not at.exception, at.exception
    return at


def test_pagina_listada_no_menu():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert PAGINA in at.sidebar.radio[0].options


def test_sem_historico_pede_dados_e_nao_mostra_numeros():
    at = _abrir()
    assert any("precisam ser carregados" in w.value for w in at.warning)
    assert not [b for b in at.button if b.label == "GERAR PREVISÃO"]


def test_gerar_previsao_e_nova_previsao():
    gravar_historico(historico_artificial(200, semente=7))
    at = _abrir()
    assert any("Nenhuma previsão gerada" in i.value for i in at.info)
    [b for b in at.button if b.label == "GERAR PREVISÃO"][0].click().run()
    assert not at.exception, at.exception
    assert any("gerada e gravada" in s.value for s in at.success)
    html = " ".join(m.value for m in at.markdown)
    assert "fi-destaque" in html
    assert "Sem vantagem comprovada sobre o acaso" in html or "Vantagem observada" in html
    assert "1 em 96.000" in html  # concurso alvo estimado para domingo
    assert [t.label for t in at.tabs][:6] == [
        "Ranking", "A · Probabilidade teórica", "B · Pontuação preditiva",
        "C · Frequência histórica", "D · Desempenho no backtest", "Modelos e indicadores"]

    [b for b in at.button if b.label == "GERAR NOVA PREVISÃO"][0].click().run()
    assert not at.exception, at.exception
    from src.preditivo import armazenamento
    assert len(armazenamento.listar_previsoes()) == 2
