"""Banco SQLite local da aplicação.

Tabelas:
- imports: cada importação de CSV e seu relatório.
- contests: concursos (importados ou digitados na conferência).
- winning_numbers: os cinco números premiados de cada concurso.
- generated_candidates: candidatos gerados, parâmetros e resultado da conferência.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PADRAO = RAIZ / "data" / "federal_intelligence.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    arquivo TEXT,
    linhas_total INTEGER NOT NULL,
    linhas_validas INTEGER NOT NULL,
    linhas_rejeitadas INTEGER NOT NULL,
    linhas_duplicadas INTEGER NOT NULL,
    relatorio TEXT
);

CREATE TABLE IF NOT EXISTS contests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concurso INTEGER NOT NULL UNIQUE,
    data TEXT NOT NULL,
    extracao TEXT NOT NULL DEFAULT '',
    origem TEXT NOT NULL CHECK (origem IN ('importacao', 'manual')),
    import_id INTEGER REFERENCES imports(id) ON DELETE SET NULL,
    criado_em TEXT NOT NULL,
    premio_zero INTEGER NOT NULL DEFAULT 0,
    dados_originais TEXT
);

CREATE TABLE IF NOT EXISTS winning_numbers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contest_id INTEGER NOT NULL REFERENCES contests(id) ON DELETE CASCADE,
    premio INTEGER NOT NULL CHECK (premio BETWEEN 1 AND 5),
    numero TEXT NOT NULL CHECK (length(numero) = 5 AND numero GLOB '[0-9][0-9][0-9][0-9][0-9]'),
    UNIQUE (contest_id, premio)
);

CREATE TABLE IF NOT EXISTS generated_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lote TEXT NOT NULL,
    gerado_em TEXT NOT NULL,
    extracao TEXT NOT NULL,
    posicao INTEGER NOT NULL,
    numero TEXT NOT NULL CHECK (length(numero) = 5 AND numero GLOB '[0-9][0-9][0-9][0-9][0-9]'),
    parametros TEXT NOT NULL,
    contest_id INTEGER REFERENCES contests(id) ON DELETE SET NULL,
    conferencia TEXT,
    conferido_em TEXT,
    UNIQUE (lote, numero)
);

CREATE INDEX IF NOT EXISTS idx_candidates_lote ON generated_candidates(lote);
CREATE INDEX IF NOT EXISTS idx_winning_contest ON winning_numbers(contest_id);
"""


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolver_caminho(caminho: Path | str | None = None) -> Path:
    """Usa o caminho informado, a variável FI_DB_PATH (testes) ou data/federal_intelligence.db."""
    if caminho:
        return Path(caminho)
    return Path(os.environ.get("FI_DB_PATH") or CAMINHO_PADRAO)


@contextmanager
def conectar(caminho: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    caminho = resolver_caminho(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conexao = sqlite3.connect(caminho)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA foreign_keys = ON")
    try:
        yield conexao
        conexao.commit()
    except Exception:
        conexao.rollback()
        raise
    finally:
        conexao.close()


# Colunas adicionadas depois da primeira versão; bancos antigos recebem ALTER TABLE.
MIGRACOES_CONTESTS = {
    "premio_zero": "ALTER TABLE contests ADD COLUMN premio_zero INTEGER NOT NULL DEFAULT 0",
    "dados_originais": "ALTER TABLE contests ADD COLUMN dados_originais TEXT",
}


def inicializar(caminho: Path | str | None = None) -> None:
    with conectar(caminho) as conexao:
        conexao.executescript(ESQUEMA)
        colunas = {linha["name"] for linha in conexao.execute("PRAGMA table_info(contests)")}
        for coluna, comando in MIGRACOES_CONTESTS.items():
            if coluna not in colunas:
                conexao.execute(comando)


# ---------------------------------------------------------------- concursos

def concursos_existentes(caminho: Path | str | None = None) -> set[int]:
    with conectar(caminho) as conexao:
        return {linha[0] for linha in conexao.execute("SELECT concurso FROM contests")}


def buscar_concurso(concurso: int, caminho: Path | str | None = None) -> dict | None:
    with conectar(caminho) as conexao:
        linha = conexao.execute(
            "SELECT id, concurso, data, extracao, origem FROM contests WHERE concurso = ?",
            (concurso,),
        ).fetchone()
        if linha is None:
            return None
        numeros = conexao.execute(
            "SELECT premio, numero FROM winning_numbers WHERE contest_id = ? ORDER BY premio",
            (linha["id"],),
        ).fetchall()
    return {**dict(linha), "premios": [n["numero"] for n in numeros]}


def _inserir_concurso(
    conexao, concurso, data, extracao, premios, origem, import_id=None,
    premio_zero=False, dados_originais=None,
) -> int:
    cursor = conexao.execute(
        "INSERT INTO contests (concurso, data, extracao, origem, import_id, criado_em, "
        "premio_zero, dados_originais) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            int(concurso), data, extracao or "", origem, import_id, _agora(),
            int(bool(premio_zero)), dados_originais,
        ),
    )
    contest_id = cursor.lastrowid
    conexao.executemany(
        "INSERT INTO winning_numbers (contest_id, premio, numero) VALUES (?, ?, ?)",
        [(contest_id, i, numero) for i, numero in enumerate(premios, start=1)],
    )
    return contest_id


def salvar_concurso_manual(
    concurso: int, data: str, extracao: str, premios: list[str], caminho: Path | str | None = None
) -> int:
    """Grava um concurso digitado. Se já existir com os mesmos números, reaproveita.

    Se já existir com números diferentes, gera erro em vez de sobrescrever.
    """
    if len(premios) != 5:
        raise ValueError("Informe exatamente cinco números premiados.")
    existente = buscar_concurso(concurso, caminho)
    if existente is not None:
        if existente["premios"] == list(premios) and existente["data"] == data:
            return existente["id"]
        raise ValueError(
            f"O concurso {concurso} já está gravado com data ou números diferentes "
            f"({existente['data']}: {', '.join(existente['premios'])}). Nada foi alterado."
        )
    with conectar(caminho) as conexao:
        return _inserir_concurso(conexao, concurso, data, extracao, premios, "manual")


def registrar_importacao(
    validos: pd.DataFrame,
    arquivo: str,
    linhas_total: int,
    linhas_rejeitadas: int,
    linhas_duplicadas: int,
    relatorio: dict,
    caminho: Path | str | None = None,
) -> int:
    """Grava a importação e os concursos válidos numa única transação."""
    with conectar(caminho) as conexao:
        cursor = conexao.execute(
            "INSERT INTO imports (criado_em, arquivo, linhas_total, linhas_validas, "
            "linhas_rejeitadas, linhas_duplicadas, relatorio) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _agora(), arquivo, linhas_total, len(validos), linhas_rejeitadas,
                linhas_duplicadas, json.dumps(relatorio, ensure_ascii=False),
            ),
        )
        import_id = cursor.lastrowid
        for registro in validos.to_dict("records"):
            premios = [registro[f"premio_{i}"] for i in range(1, 6)]
            _inserir_concurso(
                conexao, registro["concurso"], registro["data"], registro["extracao"],
                premios, "importacao", import_id,
                premio_zero=registro.get("premio_zero", False),
                dados_originais=registro.get("dados_originais"),
            )
    return import_id


def listar_importacoes(caminho: Path | str | None = None) -> pd.DataFrame:
    with conectar(caminho) as conexao:
        return pd.read_sql_query(
            "SELECT id, criado_em, arquivo, linhas_total, linhas_validas, linhas_rejeitadas, "
            "linhas_duplicadas FROM imports ORDER BY id DESC",
            conexao,
        )


def carregar_historico(caminho: Path | str | None = None) -> pd.DataFrame:
    """Histórico em formato longo: uma linha por número premiado."""
    with conectar(caminho) as conexao:
        df = pd.read_sql_query(
            "SELECT c.concurso, c.data, c.extracao, c.origem, c.premio_zero, w.premio, w.numero "
            "FROM contests c JOIN winning_numbers w ON w.contest_id = c.id "
            "ORDER BY c.concurso, w.premio",
            conexao,
            dtype={"numero": str, "extracao": str},
        )
    df["premio_zero"] = df["premio_zero"].astype(bool)
    return df


def versao_historico(caminho: Path | str | None = None) -> tuple:
    """Assinatura barata do histórico, usada como chave do cache das telas.

    Muda sempre que concursos ou importações são gravados ou apagados. Usa o
    contador do AUTOINCREMENT (sqlite_sequence), que nunca volta atrás, nem
    depois de apagar o histórico. Gerar candidatos não a altera.
    """
    with conectar(caminho) as conexao:
        return tuple(conexao.execute(
            "SELECT (SELECT count(*) FROM contests), "
            "(SELECT coalesce(max(seq), 0) FROM sqlite_sequence WHERE name = 'contests'), "
            "(SELECT count(*) FROM imports), "
            "(SELECT coalesce(max(seq), 0) FROM sqlite_sequence WHERE name = 'imports')"
        ).fetchone())


def listar_conferencias(caminho: Path | str | None = None) -> pd.DataFrame:
    """Candidatos já conferidos, com a extração e o resultado gravado (JSON)."""
    with conectar(caminho) as conexao:
        return pd.read_sql_query(
            "SELECT g.numero, g.extracao AS extracao_lote, g.conferencia, c.concurso "
            "FROM generated_candidates g JOIN contests c ON c.id = g.contest_id "
            "WHERE g.conferencia IS NOT NULL",
            conexao,
            dtype={"numero": str},
        )


def apagar_historico(caminho: Path | str | None = None) -> None:
    """Remove concursos e importações. Candidatos gerados são mantidos."""
    with conectar(caminho) as conexao:
        conexao.execute("UPDATE generated_candidates SET contest_id = NULL")
        conexao.execute("DELETE FROM winning_numbers")
        conexao.execute("DELETE FROM contests")
        conexao.execute("DELETE FROM imports")


# ---------------------------------------------------------------- candidatos

def salvar_candidatos(
    extracao: str, numeros: list[str], parametros: dict, caminho: Path | str | None = None
) -> str:
    lote = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    gerado_em = parametros.get("gerado_em") or _agora()
    parametros_json = json.dumps(parametros, ensure_ascii=False)
    with conectar(caminho) as conexao:
        conexao.executemany(
            "INSERT INTO generated_candidates (lote, gerado_em, extracao, posicao, numero, parametros) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(lote, gerado_em, extracao, i, n, parametros_json) for i, n in enumerate(numeros, 1)],
        )
    return lote


def listar_lotes(caminho: Path | str | None = None) -> pd.DataFrame:
    with conectar(caminho) as conexao:
        return pd.read_sql_query(
            "SELECT lote, MIN(gerado_em) AS gerado_em, extracao, COUNT(*) AS quantidade, "
            "SUM(conferencia IS NOT NULL) AS conferidos "
            "FROM generated_candidates GROUP BY lote, extracao ORDER BY gerado_em DESC, lote DESC",
            conexao,
        )


def candidatos_do_lote(lote: str, caminho: Path | str | None = None) -> pd.DataFrame:
    with conectar(caminho) as conexao:
        df = pd.read_sql_query(
            "SELECT g.id, g.posicao, g.numero, g.extracao, g.gerado_em, g.conferencia, "
            "g.conferido_em, c.concurso "
            "FROM generated_candidates g LEFT JOIN contests c ON c.id = g.contest_id "
            "WHERE g.lote = ? ORDER BY g.posicao",
            conexao,
            params=(lote,),
            dtype={"numero": str},
        )
    return df


def salvar_conferencia(
    candidato_id: int, contest_id: int, resultado: dict, caminho: Path | str | None = None
) -> None:
    with conectar(caminho) as conexao:
        conexao.execute(
            "UPDATE generated_candidates SET contest_id = ?, conferencia = ?, conferido_em = ? "
            "WHERE id = ?",
            (contest_id, json.dumps(resultado, ensure_ascii=False), _agora(), candidato_id),
        )
