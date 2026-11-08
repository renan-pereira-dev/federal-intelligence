"""Tabelas próprias do motor no mesmo arquivo SQLite.

Só cria tabelas novas (CREATE TABLE IF NOT EXISTS). Não altera nem apaga as
tabelas existentes (contests, winning_numbers, generated_candidates, imports).
Previsões nunca são apagadas: cada execução recebe um identificador único.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import database as db
from src.evaluator import conferir

ESQUEMA_MOTOR = """
CREATE TABLE IF NOT EXISTS prediction_backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    versao_historico TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config TEXT NOT NULL,
    metricas TEXT NOT NULL,
    UNIQUE (versao_historico, config_hash)
);

CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    criado_em TEXT NOT NULL,
    tipo_execucao TEXT NOT NULL,
    concurso_alvo INTEGER NOT NULL,
    data_alvo TEXT,
    data_status TEXT NOT NULL,
    extracao TEXT NOT NULL,
    versao_historico TEXT NOT NULL,
    ultimo_concurso INTEGER NOT NULL,
    backtest_id INTEGER REFERENCES prediction_backtests(id),
    validado INTEGER NOT NULL,
    rotulo_principal TEXT NOT NULL,
    resumo TEXT NOT NULL,
    conferido_em TEXT,
    resultado TEXT
);

CREATE TABLE IF NOT EXISTS prediction_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL REFERENCES predictions(id),
    posicao INTEGER NOT NULL CHECK (posicao BETWEEN 1 AND 10),
    numero TEXT NOT NULL CHECK (length(numero) = 5 AND numero GLOB '[0-9][0-9][0-9][0-9][0-9]'),
    pontuacao REAL NOT NULL,
    detalhes TEXT NOT NULL,
    conferencia TEXT,
    UNIQUE (prediction_id, posicao),
    UNIQUE (prediction_id, numero)
);

CREATE INDEX IF NOT EXISTS idx_prediction_items ON prediction_items(prediction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_alvo ON predictions(concurso_alvo);
"""


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def inicializar(caminho: Path | str | None = None) -> None:
    with db.conectar(caminho) as conexao:
        conexao.executescript(ESQUEMA_MOTOR)


def _json(valor) -> str:
    return json.dumps(valor, ensure_ascii=False, default=_padrao_json)


def _padrao_json(valor):
    if hasattr(valor, "item"):  # escalares numpy
        return valor.item()
    raise TypeError(f"Tipo não serializável: {type(valor).__name__}")


# --------------------------------------------------------------- backtests

def buscar_backtest(versao: tuple, config_hash: str, caminho=None) -> dict | None:
    with db.conectar(caminho) as conexao:
        linha = conexao.execute(
            "SELECT id, criado_em, metricas FROM prediction_backtests "
            "WHERE versao_historico = ? AND config_hash = ?",
            (_json(list(versao)), config_hash),
        ).fetchone()
    if linha is None:
        return None
    return {"id": linha["id"], "criado_em": linha["criado_em"], **json.loads(linha["metricas"])}


def salvar_backtest(versao: tuple, config_hash: str, config: dict, metricas: dict, caminho=None) -> int:
    with db.conectar(caminho) as conexao:
        cursor = conexao.execute(
            "INSERT OR IGNORE INTO prediction_backtests (criado_em, versao_historico, config_hash, config, metricas) "
            "VALUES (?, ?, ?, ?, ?)",
            (_agora(), _json(list(versao)), config_hash, _json(config), _json(metricas)),
        )
        if cursor.rowcount:
            return cursor.lastrowid
    return buscar_backtest(versao, config_hash, caminho)["id"]


# --------------------------------------------------------------- previsões

def salvar_previsao(previsao: dict, itens: list[dict], caminho=None) -> str:
    """Grava uma previsão e seus 10 candidatos numa única transação."""
    identificador = datetime.now().strftime("PREV-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with db.conectar(caminho) as conexao:
        conexao.execute(
            "INSERT INTO predictions (id, criado_em, tipo_execucao, concurso_alvo, data_alvo, data_status, "
            "extracao, versao_historico, ultimo_concurso, backtest_id, validado, rotulo_principal, resumo) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                identificador, previsao.get("criado_em") or _agora(), previsao["tipo_execucao"],
                int(previsao["concurso_alvo"]), previsao.get("data_alvo"), previsao["data_status"],
                previsao["extracao"], _json(list(previsao["versao_historico"])),
                int(previsao["ultimo_concurso"]), previsao.get("backtest_id"),
                int(bool(previsao["validado"])), previsao["rotulo_principal"], _json(previsao["resumo"]),
            ),
        )
        conexao.executemany(
            "INSERT INTO prediction_items (prediction_id, posicao, numero, pontuacao, detalhes) "
            "VALUES (?, ?, ?, ?, ?)",
            [(identificador, i["posicao"], i["numero"], float(i["pontuacao"]), _json(i["detalhes"]))
             for i in itens],
        )
    return identificador


def listar_previsoes(caminho=None) -> pd.DataFrame:
    with db.conectar(caminho) as conexao:
        df = pd.read_sql_query(
            "SELECT p.id, p.criado_em, p.tipo_execucao, p.concurso_alvo, p.data_alvo, p.data_status, "
            "p.extracao, p.ultimo_concurso, p.validado, p.rotulo_principal, p.conferido_em, p.resultado, "
            "(SELECT numero FROM prediction_items i WHERE i.prediction_id = p.id AND i.posicao = 1) AS principal "
            "FROM predictions p ORDER BY p.criado_em DESC, p.id DESC",
            conexao,
            dtype={"principal": str},
        )
    return df


def carregar_previsao(identificador: str, caminho=None) -> dict | None:
    with db.conectar(caminho) as conexao:
        linha = conexao.execute("SELECT * FROM predictions WHERE id = ?", (identificador,)).fetchone()
        if linha is None:
            return None
        itens = conexao.execute(
            "SELECT posicao, numero, pontuacao, detalhes, conferencia FROM prediction_items "
            "WHERE prediction_id = ? ORDER BY posicao",
            (identificador,),
        ).fetchall()
    previsao = dict(linha)
    previsao["resumo"] = json.loads(previsao["resumo"])
    previsao["versao_historico"] = json.loads(previsao["versao_historico"])
    previsao["resultado"] = json.loads(previsao["resultado"]) if previsao["resultado"] else None
    previsao["itens"] = [
        {"posicao": i["posicao"], "numero": i["numero"], "pontuacao": i["pontuacao"],
         "detalhes": json.loads(i["detalhes"]),
         "conferencia": json.loads(i["conferencia"]) if i["conferencia"] else None}
        for i in itens
    ]
    return previsao


def conferir_pendentes(caminho=None) -> list[str]:
    """Compara automaticamente as previsões cujo concurso alvo já está no banco.

    Usa as mesmas regras da tela Conferência (src/evaluator.py). Só o acerto
    exato conta como prêmio principal; o resto é coincidência parcial.
    Devolve os identificadores conferidos agora.
    """
    with db.conectar(caminho) as conexao:
        pendentes = conexao.execute(
            "SELECT p.id, p.extracao, p.concurso_alvo FROM predictions p "
            "JOIN contests c ON c.concurso = p.concurso_alvo WHERE p.conferido_em IS NULL"
        ).fetchall()
    conferidas = []
    for previsao in pendentes:
        concurso = db.buscar_concurso(previsao["concurso_alvo"], caminho)
        if not concurso or len(concurso["premios"]) != 5:
            continue
        premios = concurso["premios"]
        itens = carregar_previsao(previsao["id"], caminho)["itens"]
        resumo = {"concurso": concurso["concurso"], "data": concurso["data"], "premios": premios,
                  "acertos_exatos": 0, "com_coincidencia": 0, "principal_exato": False,
                  "coincidencias_milhar": 0, "coincidencias_centena": 0, "coincidencias_dezena": 0}
        atualizacoes = []
        for item in itens:
            resultado = conferir(item["numero"], premios, previsao["extracao"])
            dados = resultado.para_dict()
            for tipo in set(resultado.terminacoes.values()):
                resumo[f"coincidencias_{tipo}"] += 1
            atualizacoes.append((_json(dados), previsao["id"], item["posicao"]))
            if resultado.acertou_principal:
                resumo["acertos_exatos"] += 1
                if item["posicao"] == 1:
                    resumo["principal_exato"] = True
            elif resultado.tem_coincidencia:
                resumo["com_coincidencia"] += 1
        with db.conectar(caminho) as conexao:
            conexao.executemany(
                "UPDATE prediction_items SET conferencia = ? WHERE prediction_id = ? AND posicao = ?",
                atualizacoes,
            )
            conexao.execute(
                "UPDATE predictions SET conferido_em = ?, resultado = ? WHERE id = ?",
                (_agora(), _json(resumo), previsao["id"]),
            )
        conferidas.append(previsao["id"])
    return conferidas
