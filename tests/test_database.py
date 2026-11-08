"""Testes do SQLite usando um banco temporário. Dados ARTIFICIAIS."""

import json
import sqlite3

import pytest

from src import database as db
from src.evaluator import conferir
from src.generator import gerar_candidatos, parametros_geracao
from src.importer import importar
from src.statistics import frequencia_candidato

CSV = (
    "concurso;data;extracao;premio_1;premio_2;premio_3;premio_4;premio_5\n"
    "100;01/01/2020;domingo;00123;01234;12345;00001;90000\n"
    "101;05/01/2020;domingo;54321;00123;11111;22222;33333\n"
)


@pytest.fixture
def banco(tmp_path):
    caminho = tmp_path / "teste.db"
    db.inicializar(caminho)
    return caminho


def test_tabelas_criadas(banco):
    with sqlite3.connect(banco) as c:
        nomes = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"contests", "winning_numbers", "generated_candidates", "imports"} <= nomes


def test_importacao_grava_e_preserva_zeros(banco):
    r = importar(CSV, db.concursos_existentes(banco))
    db.registrar_importacao(r.validos, "teste.csv", r.linhas_total, r.linhas_rejeitadas, r.duplicados, {}, banco)
    hist = db.carregar_historico(banco)
    assert len(hist) == 10
    assert "00001" in set(hist["numero"])
    assert frequencia_candidato("00123", hist)["exato"] == 2

    # Reimportar o mesmo arquivo não duplica concursos.
    r2 = importar(CSV, db.concursos_existentes(banco))
    assert r2.linhas_validas == 0 and r2.duplicados == 2


def test_candidatos_e_conferencia(banco):
    numeros = gerar_candidatos("quartou", 5)
    lote = db.salvar_candidatos("quartou", numeros, parametros_geracao("quartou", 5), banco)
    df = db.candidatos_do_lote(lote, banco)
    assert list(df["numero"]) == numeros

    premios = ["00123", "01234", "12345", "00001", "90000"]
    contest_id = db.salvar_concurso_manual(500, "2026-11-04", "quartou", premios, banco)
    linha = df.iloc[0]
    resultado = conferir(linha["numero"], premios, "quartou")
    db.salvar_conferencia(int(linha["id"]), contest_id, resultado.para_dict(), banco)

    df2 = db.candidatos_do_lote(lote, banco)
    salvo = json.loads(df2.iloc[0]["conferencia"])
    assert salvo["candidato"] == linha["numero"]
    assert df2.iloc[0]["concurso"] == 500


def test_concurso_manual_conflitante_nao_sobrescreve(banco):
    premios = ["00123", "01234", "12345", "00001", "90000"]
    primeiro = db.salvar_concurso_manual(600, "2026-11-04", "quartou", premios, banco)
    assert db.salvar_concurso_manual(600, "2026-11-04", "quartou", premios, banco) == primeiro
    with pytest.raises(ValueError, match="Nada foi alterado"):
        db.salvar_concurso_manual(600, "2026-11-04", "quartou", ["11111"] * 5, banco)


def test_banco_rejeita_numero_sem_5_digitos(banco):
    with pytest.raises(sqlite3.IntegrityError):
        db.salvar_candidatos("quartou", ["1234"], {}, banco)


def test_migracao_de_banco_antigo(tmp_path):
    caminho = tmp_path / "antigo.db"
    with sqlite3.connect(caminho) as c:
        c.execute(
            "CREATE TABLE contests (id INTEGER PRIMARY KEY AUTOINCREMENT, concurso INTEGER NOT NULL UNIQUE, "
            "data TEXT NOT NULL, extracao TEXT NOT NULL DEFAULT '', origem TEXT NOT NULL, "
            "import_id INTEGER, criado_em TEXT NOT NULL)"
        )
        c.execute("INSERT INTO contests (concurso, data, origem, criado_em) VALUES (1, '2020-01-01', 'manual', 'x')")
    db.inicializar(caminho)
    db.inicializar(caminho)  # rodar duas vezes não pode falhar
    with sqlite3.connect(caminho) as c:
        colunas = {r[1] for r in c.execute("PRAGMA table_info(contests)")}
        assert {"premio_zero", "dados_originais"} <= colunas
        assert c.execute("SELECT premio_zero FROM contests").fetchone()[0] == 0


def test_importacao_grava_premio_zero_e_dados_originais(banco):
    csv = (
        '"concurso";"data";"bilhete_1";"valor_nominal_1";"bilhete_2";"valor_nominal_2";"bilhete_3";'
        '"valor_nominal_3";"bilhete_4";"valor_nominal_4";"bilhete_5";"valor_nominal_5"\n'
        '"1";"2026-07-01";"\'012345";"0";"\'000001";"0";"\'099999";"0";"\'050000";"0";"\'001000";"0"\n'
        '"2";"2026-08-01";"\'012345";"500";"\'000001";"40";"\'099999";"30";"\'050000";"20";"\'001000";"10"\n'
    )
    r = importar(csv)
    db.registrar_importacao(r.validos, "t.csv", r.linhas_total, r.linhas_rejeitadas, r.duplicados,
                            r.relatorio(), banco)
    hist = db.carregar_historico(banco)
    assert set(hist.loc[hist["premio_zero"], "concurso"]) == {1}
    assert hist.loc[hist["concurso"] == 1, "numero"].tolist() == ["12345", "00001", "99999", "50000", "01000"]
    with sqlite3.connect(banco) as c:
        originais = json.loads(c.execute("SELECT dados_originais FROM contests WHERE concurso = 2").fetchone()[0])
        relatorio = json.loads(c.execute("SELECT relatorio FROM imports").fetchone()[0])
    assert originais["valor_nominal_1"] == "500"
    assert relatorio["concursos_premio_zero"] == [1]
