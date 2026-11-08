"""Orquestra uma execução do motor (botões GERAR PREVISÃO e GERAR NOVA PREVISÃO).

1. Lê o histórico do banco e a sua versão.
2. Backtest walk-forward: reaproveita o gravado para esta versão e configuração;
   roda só se ainda não existir.
3. Modelos finais: carrega os gravados para esta versão; treina só se faltar.
   GERAR NOVA PREVISÃO treina de novo com uma nova semente de amostragem.
4. Avalia as 10.000 milhares com dados até o último concurso e combina os
   modelos (peso igual).
5. Escolhe 10 candidatos com milhares distintas, dentro do universo da extração.
   O 1º algarismo é sorteado com a fonte segura do Python (secrets), porque o
   modelo não o avalia.
6. Grava a previsão (identificador único) sem apagar as anteriores.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import database as db
from src.preditivo import armazenamento, backtest, estrategias
from src.preditivo.atributos import (
    IndiceHistorico,
    descrever_atributo,
    historico_do_numero,
    resumo_estatistico,
)
from src.preditivo.modelos import (
    CONFIG_PADRAO,
    CONFIG_RAPIDA,
    NOMES_MODELOS,
    ORDEM_MODELOS,
    Configuracao,
    carregar_ou_treinar,
    importancias,
    ranking_milhares,
)
from src.probability import obter_extracao

GERAR = "GERAR PREVISÃO"
GERAR_NOVA = "GERAR NOVA PREVISÃO"
ROTULO_VALIDADO = "Número principal (modelo com vantagem no backtest)"
ROTULO_SEM_VANTAGEM = "1º do ranking do modelo"
AVISO_SEM_VANTAGEM = "Sem vantagem comprovada sobre o acaso"
# Indicadores usados para explicar cada candidato (os estruturais ficam de fora).
PREFIXOS_FATORES = ("freq_", "atraso_", "tendencia_", "repetiu_", "posicao_")


class ErroMotor(Exception):
    """Algo impede o motor de calcular; a mensagem é mostrada ao usuário."""


def config_ativa() -> Configuracao:
    return CONFIG_RAPIDA if os.environ.get("FI_MOTOR_CONFIG") == "rapido" else CONFIG_PADRAO


def obter_backtest(indice: IndiceHistorico, versao: tuple, config: Configuracao, caminho,
                   progresso=None) -> tuple[dict, bool]:
    """Backtest desta versão do histórico: do banco, ou calculado agora e gravado."""
    gravado = armazenamento.buscar_backtest(versao, config.hash(), caminho)
    if gravado is not None:
        return gravado, False
    try:
        metricas = backtest.executar(indice, config, progresso)
    except ValueError as exc:
        metricas = {"erro": str(exc), "validado": False}
    identificador = armazenamento.salvar_backtest(versao, config.hash(), asdict(config), metricas, caminho)
    return {"id": identificador, "criado_em": datetime.now().isoformat(timespec="seconds"), **metricas}, True


def _fatores(posicao: int, atributos: pd.DataFrame, percentis: pd.DataFrame) -> list[dict]:
    """Os 3 indicadores em que esta milhar mais se afasta das demais."""
    colunas = [c for c in atributos.columns if c.startswith(PREFIXOS_FATORES)]
    linha = percentis.iloc[posicao][colunas]
    destaque = (linha - 0.5).abs().sort_values(ascending=False).head(3).index
    saida = []
    for coluna in destaque:
        valor = float(atributos.iloc[posicao][coluna])
        saida.append({
            "atributo": coluna,
            "descricao": descrever_atributo(coluna),
            "valor": round(valor, 4),
            "percentil": round(float(linha[coluna]) * 100, 1),
        })
    return saida


def gerar_previsao(extracao: str, alvo: dict, nova: bool = False, caminho: Path | str | None = None,
                   config: Configuracao | None = None,
                   progresso: Callable[[float, str], None] | None = None) -> str:
    """Executa o motor, grava a previsão e devolve o identificador."""
    inicio_relogio = time.perf_counter()
    caminho = db.resolver_caminho(caminho)
    config = config or config_ativa()
    armazenamento.inicializar(caminho)
    universo = obter_extracao(extracao)
    if not universo.habilitada:
        raise ErroMotor(f"A extração {universo.nome} não está habilitada.")

    historico = db.carregar_historico(caminho)
    if historico.empty:
        raise ErroMotor("Não há histórico no banco. Importe os resultados na tela Histórico.")
    versao = db.versao_historico(caminho)
    try:
        indice = IndiceHistorico(historico)
    except ValueError as exc:
        raise ErroMotor(f"Histórico inconsistente: {exc}") from exc
    if indice.total < config.minimo_treino + 1:
        raise ErroMotor(
            f"Histórico insuficiente: {indice.total} concursos. O motor precisa de pelo menos "
            f"{config.minimo_treino + 1} para treinar sem usar dados do futuro."
        )

    avisar = progresso or (lambda fracao, texto: None)
    avisar(0.0, "Verificando o backtest desta versão do histórico...")
    metricas, backtest_novo = obter_backtest(
        indice, versao, config, caminho, lambda f, t: avisar(0.05 + 0.75 * f, t))

    semente = secrets.randbits(31) if nova else config.semente
    avisar(0.82, "Treinando os modelos finais..." if nova else "Carregando ou treinando os modelos finais...")
    pacote, treinou = carregar_ou_treinar(caminho, versao, indice, config, semente)
    if not pacote["modelos"]:
        raise ErroMotor("Nenhum modelo pôde ser treinado: " + "; ".join(
            f"{NOMES_MODELOS.get(k, k)}: {v}" for k, v in pacote["status"].items()))

    avisar(0.9, "Avaliando as 10.000 milhares...")
    tabela, atributos = ranking_milhares(pacote, indice, indice.total)
    percentis_atributos = atributos.rank(pct=True)
    gerador = secrets.SystemRandom()
    # Estratégia de seleção: o motor padrão, a menos que uma estratégia APROVADA no
    # laboratório tenha sido ativada explicitamente (reversível em estrategias.reverter).
    ativa = estrategias.estrategia_ativa(caminho)
    if ativa["estrategia"] == estrategias.MOTOR_PADRAO:
        escolhidos = backtest.selecionar(tabela, universo.numero_maximo, gerador.choice)
    else:
        alternativa = estrategias.candidatos_proximo(
            indice, pacote, ativa["estrategia"], universo.numero_maximo, ativa.get("pesos"),
            semente=secrets.randbits(63))
        linha_da_milhar = {int(m): i for i, m in enumerate(tabela["milhar"].to_numpy())}
        escolhidos = [(n, linha_da_milhar[int(n) % 10_000]) for n in alternativa["candidatos"]]
    if len(escolhidos) < backtest.QUANTIDADE:
        raise ErroMotor("Não foi possível formar 10 candidatos dentro do universo da extração.")

    validado = bool(metricas.get("validado"))
    itens = []
    for posicao, (numero, linha) in enumerate(escolhidos, start=1):
        registro = tabela.iloc[linha]
        itens.append({
            "posicao": posicao,
            "numero": numero,
            "pontuacao": round(float(registro["pontuacao"]), 2),
            "detalhes": {
                "milhar": numero[1:],
                "posicao_ranking_milhar": int(linha) + 1,
                "contribuicoes": {k: round(float(registro[k]) * 100, 2) for k in pacote["modelos"]},
                "fatores": _fatores(linha, atributos, percentis_atributos),
                "historico": historico_do_numero(indice, numero),
                "primeiro_algarismo": "sorteado ao acaso (secrets) entre os válidos na faixa "
                                      f"{universo.faixa_texto}; o modelo não avalia este algarismo",
            },
        })

    pesos = importancias(pacote)
    pesos_medios = (
        pesos.groupby("atributo")["peso"].mean().sort_values(ascending=False).head(15)
        if not pesos.empty else pd.Series(dtype=float)
    )
    resumo = {
        "alvo": alvo,
        "extracao_nome": universo.nome,
        "universo": universo.faixa_texto,
        "probabilidade": {"chance_texto": universo.chance_texto, "um_em": universo.chance_um_em},
        "modelos": {k: {"nome": NOMES_MODELOS[k], "status": pacote["status"].get(k, "não executado")}
                    for k in ORDEM_MODELOS},
        "modelos_no_conjunto": list(pacote["modelos"]),
        "treino": pacote["treino"],
        "treinou_agora": treinou,
        "backtest_calculado_agora": backtest_novo,
        "backtest": metricas,
        "importancias": [{"atributo": a, "descricao": descrever_atributo(a), "peso": round(float(p), 4)}
                         for a, p in pesos_medios.items()],
        "estatisticas": resumo_estatistico(indice),
        "config": asdict(config),
        "estrategia_selecao": ativa["estrategia"],
        "semente_amostragem": int(semente),
        "duracao_segundos": round(time.perf_counter() - inicio_relogio, 1),
        "aviso": None if validado else AVISO_SEM_VANTAGEM,
    }
    previsao = {
        "tipo_execucao": GERAR_NOVA if nova else GERAR,
        "concurso_alvo": alvo["concurso"],
        "data_alvo": alvo.get("data"),
        "data_status": alvo.get("status", "estimada"),
        "extracao": extracao,
        "versao_historico": versao,
        "ultimo_concurso": int(indice.concursos[-1]),
        "backtest_id": metricas.get("id"),
        "validado": validado,
        "rotulo_principal": ROTULO_VALIDADO if validado else ROTULO_SEM_VANTAGEM,
        "resumo": resumo,
    }
    avisar(0.97, "Gravando a previsão...")
    identificador = armazenamento.salvar_previsao(previsao, itens, caminho)
    avisar(1.0, "Previsão gravada.")
    return identificador
