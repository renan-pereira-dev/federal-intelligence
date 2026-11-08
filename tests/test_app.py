"""Teste de fumaça da interface com o AppTest do Streamlit.

Usa um banco temporário (FI_DB_PATH) e dados ARTIFICIAIS.
"""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGINAS = ["Início", "Gerador", "Histórico", "Conferência", "Probabilidades", "Fontes e regras"]


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "app_teste.db"))


def _abrir(pagina: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(pagina).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("pagina", PAGINAS)
def test_paginas_abrem_sem_erro(pagina):
    _abrir(pagina)


def test_historico_vazio_pede_importacao():
    at = _abrir("Histórico")
    assert any("precisam ser importados" in w.value for w in at.warning)


def test_gerar_e_conferir():
    at = _abrir("Gerador")
    at.button[0].click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Candidato experimental" in html
    assert "não representa maior chance de ganhar" in html

    at.sidebar.radio[0].set_value("Conferência").run()
    at.number_input[0].set_value(5800).run()
    at.date_input[0].set_value(date(2026, 11, 4)).run()
    for i, numero in enumerate(["95655", "39888", "62547", "01339", "74262"]):
        at.text_input[i].set_value(numero)
    botoes = [b for b in at.button if b.label == "Conferir e salvar"]
    botoes[0].click().run()
    assert not at.exception, at.exception
    assert any("Conferência salva" in s.value for s in at.success)


def _semear_historico():
    from src import database as db
    from src.importer import importar

    linhas = ["concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5"]
    for c in range(1, 61):
        numeros = [f"{(c * 7919 + p * 104729) % 99999 + 1:05d}" for p in range(5)]
        ano = 1962 + c % 60
        linhas.append(f"{c};{ano}-03-01;" + ";".join(numeros))
    r = importar("\n".join(linhas) + "\n")
    assert r.linhas_validas == 60
    db.inicializar()
    db.registrar_importacao(r.validos, "artificial.csv", r.linhas_total, 0, 0, r.relatorio())


@pytest.mark.parametrize("pagina", ["Histórico", "Taxa histórica", "Gerador"])
def test_paginas_com_historico_artificial(pagina):
    _semear_historico()
    at = _abrir(pagina)
    assert not at.exception, at.exception


def test_abas_e_filtro_premio_zero_com_historico():
    _semear_historico()
    at = _abrir("Taxa histórica")
    texto = " ".join(m.value for m in at.markdown)
    assert "Taxa de acerto histórica por terminação" in " ".join(s.value for s in at.subheader)
    assert "não muda a chance" in texto


def test_cache_atualiza_apos_nova_importacao():
    from src import database as db
    from src.importer import importar

    _semear_historico()
    at = _abrir("Histórico")
    assert at.metric[0].value == "60"
    r = importar("concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5\n61;01/01/2000;1;2;3;4;5\n")
    db.registrar_importacao(r.validos, "mais.csv", 1, 0, 0, r.relatorio())
    at.run()
    assert not at.exception
    assert at.metric[0].value == "61"


def test_abas_novas_do_historico():
    _semear_historico()
    at = _abrir("Histórico")
    assert [t.label for t in at.tabs] == ["Frequências", "Por posição", "Terminações", "Décadas", "Épocas",
                                          "Repetições", "Distribuições", "Dados"]
    assert len(at.get("plotly_chart")) >= 12
