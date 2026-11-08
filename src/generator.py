"""Gerador de números candidatos experimentais.

Usa sorteio uniforme sem reposição com a fonte segura de aleatoriedade do
sistema operacional (secrets.SystemRandom). O histórico nunca é usado aqui.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from random import Random

from src.probability import formatar_numero, obter_extracao

QUANTIDADES_PERMITIDAS = (5, 10, 20)
METODO = "secrets.SystemRandom().sample (uniforme, sem reposição)"


class ExtracaoNaoHabilitada(RuntimeError):
    """A extração existe, mas a geração não foi habilitada por falta de regra confirmada."""


def gerar_candidatos(chave_extracao: str, quantidade: int, rng: Random | None = None) -> list[str]:
    """Gera `quantidade` números distintos, uniformes no universo da extração.

    `rng` só deve ser informado em testes. Sem ele, usa secrets.SystemRandom.
    """
    if quantidade not in QUANTIDADES_PERMITIDAS:
        raise ValueError(f"Quantidade deve ser uma de {QUANTIDADES_PERMITIDAS}, recebido {quantidade}.")

    extracao = obter_extracao(chave_extracao)
    if not extracao.habilitada:
        raise ExtracaoNaoHabilitada(
            f"A geração para {extracao.nome} está desabilitada até a regra oficial ser confirmada."
        )

    fonte = rng if rng is not None else secrets.SystemRandom()
    universo = range(extracao.numero_minimo, extracao.numero_maximo + 1)
    sorteados = fonte.sample(universo, quantidade)
    return [formatar_numero(n) for n in sorteados]


def parametros_geracao(chave_extracao: str, quantidade: int) -> dict:
    """Parâmetros registrados no banco junto com cada lote gerado."""
    extracao = obter_extracao(chave_extracao)
    return {
        "extracao": extracao.chave,
        "quantidade": quantidade,
        "universo_min": formatar_numero(extracao.numero_minimo),
        "universo_max": formatar_numero(extracao.numero_maximo),
        "metodo": METODO,
        "usa_historico": False,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
    }
