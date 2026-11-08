"""Próximo concurso: número, data e extração.

Regra publicada na página oficial da Loteria Federal (consultada em 07/11/2026):
- "QUARTOU: Sorteio realizado todas as quartas-feiras";
- "Extração de Sábado: sorteios realizados aos domingos";
- "ENRICOU: sorteio mensal realizado em um domingo do mês";
- "Especial de Natal: sorteio anual realizado em dezembro";
- "Caso alguma data de sorteio coincida com um feriado, a CAIXA define a
  alteração do dia de sua realização."

A data só é marcada como CONFIRMADA quando a consulta ao serviço oficial da
CAIXA devolve a data do próximo concurso. Caso contrário, é ESTIMADA pela regra
acima (próxima quarta ou domingo depois do último concurso do banco), e a
extração também é estimada: em um domingo pode ser a extração regular ou o
ENRICOU, e a regra não diz qual domingo do mês.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta

URL_SERVICO_OFICIAL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/federal"
URL_PAGINA_OFICIAL = "https://loterias.caixa.gov.br/Paginas/Federal.aspx"
REGRA_CALENDARIO = (
    "Página oficial da Loteria Federal: QUARTOU todas as quartas-feiras; extração regular aos "
    "domingos; ENRICOU em um domingo do mês; Especial de Natal em dezembro; feriados podem "
    "mudar o dia."
)
QUARTA, DOMINGO = 2, 6


def estimar_proximo(ultimo_concurso: int, ultima_data: str, hoje: date | None = None) -> dict:
    """Estimativa pela regra publicada (sem consultar a internet)."""
    ultima = date.fromisoformat(ultima_data)
    dia = ultima + timedelta(days=1)
    while dia.weekday() not in (QUARTA, DOMINGO):
        dia += timedelta(days=1)
    extracao = "quartou" if dia.weekday() == QUARTA else "domingo"
    hoje = hoje or date.today()
    observacoes = [
        "Data estimada pela regra publicada: o calendário oficial não pôde ser confirmado. "
        "Feriados e sorteios especiais podem mudar o dia.",
    ]
    if extracao == "domingo":
        observacoes.append(
            "Aos domingos a extração pode ser a regular ou o ENRICOU (um domingo por mês). "
            "Confira na CAIXA e ajuste a extração se necessário."
        )
    if dia.month == 12:
        observacoes.append("Em dezembro também há o Especial de Natal, em data definida pela CAIXA.")
    if dia < hoje:
        observacoes.append(
            "A data estimada já passou: o concurso pode ter sido sorteado e ainda não estar no "
            "banco. Importe o resultado mais recente para atualizar."
        )
    return {
        "concurso": int(ultimo_concurso) + 1,
        "data": dia.isoformat(),
        "status": "estimada",
        "extracao": extracao,
        "extracao_status": "estimada",
        "fonte": REGRA_CALENDARIO,
        "observacoes": observacoes,
    }


def _ler_data(texto: str) -> str | None:
    for formato in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(texto.strip(), formato).date().isoformat()
        except (ValueError, AttributeError):
            continue
    return None


def interpretar_servico(dados: dict, estimativa: dict) -> dict:
    """Aplica a resposta do serviço oficial à estimativa. Só confirma o que vier preenchido."""
    resultado = dict(estimativa, observacoes=list(estimativa["observacoes"]))
    proximo = dados.get("numeroConcursoProximo")
    data_oficial = _ler_data(dados.get("dataProximoConcurso") or "")
    if proximo and int(proximo) != estimativa["concurso"]:
        resultado["observacoes"].insert(0,
            f"O serviço oficial indica o concurso {proximo} como próximo, mas o banco local vai até "
            f"o {estimativa['concurso'] - 1}. Importe os resultados que faltam.")
        return resultado
    if data_oficial:
        resultado["data"] = data_oficial
        resultado["status"] = "confirmada"
        resultado["fonte"] = f"Serviço oficial da CAIXA ({URL_SERVICO_OFICIAL})"
        resultado["observacoes"] = [o for o in resultado["observacoes"] if not o.startswith("Data estimada")]
        dia = date.fromisoformat(data_oficial)
        if dia.weekday() == QUARTA:
            resultado["extracao"] = "quartou"
        elif dia.weekday() != DOMINGO:
            resultado["observacoes"].append("A data oficial não é quarta nem domingo; confira a extração.")
    else:
        resultado["observacoes"].insert(0,
            "O serviço oficial da CAIXA respondeu, mas sem a data do próximo concurso.")
    return resultado


def consultar_oficial(estimativa: dict, timeout: float = 8.0) -> dict:
    """Tenta confirmar a data no serviço oficial. Sem internet, devolve a estimativa."""
    requisicao = urllib.request.Request(
        URL_SERVICO_OFICIAL, headers={"User-Agent": "FederalIntelligence/0.1", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(requisicao, timeout=timeout) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
    except Exception as exc:
        resultado = dict(estimativa, observacoes=list(estimativa["observacoes"]))
        resultado["observacoes"].insert(0, f"Não foi possível consultar o serviço oficial ({type(exc).__name__}).")
        return resultado
    return interpretar_servico(dados, estimativa)
