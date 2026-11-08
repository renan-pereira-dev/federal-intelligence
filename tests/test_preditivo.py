"""Testes do motor de previsão. Usam SOMENTE dados ARTIFICIAIS e bancos temporários."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src import database as db
from src.importer import importar
from src.preditivo import armazenamento, backtest, calendario, modelos, motor
from src.preditivo.atributos import IndiceHistorico, calcular_atributos, historico_do_numero

CONFIG_TESTE = replace(modelos.CONFIG_RAPIDA, concursos_teste=60, bloco=30, minimo_treino=60,
                       janela_treino=200, arvores=10, iteracoes=20)


def historico_artificial(concursos: int, semente: int = 1, gerador=None) -> pd.DataFrame:
    """Histórico longo artificial (uma linha por prêmio)."""
    rng = np.random.default_rng(semente)
    inicio = date(2000, 1, 5)
    linhas = []
    for c in range(1, concursos + 1):
        numeros = gerador(rng) if gerador else rng.integers(1, 96001, 5)
        for p, n in enumerate(numeros, 1):
            linhas.append({"concurso": c, "data": (inicio + timedelta(days=7 * c)).isoformat(),
                           "extracao": "", "premio": p, "numero": f"{int(n):05d}"})
    return pd.DataFrame(linhas)


def gravar_historico(df: pd.DataFrame) -> None:
    largo = df.pivot(index=["concurso", "data"], columns="premio", values="numero").reset_index()
    linhas = ["concurso;data;premio_1;premio_2;premio_3;premio_4;premio_5"]
    linhas += [";".join([str(r["concurso"]), r["data"]] + [r[p] for p in range(1, 6)]) for _, r in largo.iterrows()]
    resultado = importar("\n".join(linhas) + "\n")
    db.inicializar()
    db.registrar_importacao(resultado.validos, "artificial.csv", resultado.linhas_total, 0, 0, resultado.relatorio())


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "motor_teste.db"))


# ------------------------------------------------------------------ atributos

def test_contagens_e_atrasos_simples():
    df = historico_artificial(3, gerador=lambda rng: [11111, 22222, 33333, 44444, 55555])
    indice = IndiceHistorico(df)
    assert indice.contagem("t4", [1111], 3)[0] == 3
    assert indice.contagem("t4", [1111], 3, janela=1)[0] == 1
    assert indice.contagem("t4", [1111], 0)[0] == 0
    assert indice.atraso("t4", [1111], 3)[0] == 1
    assert indice.atraso("t4", [9999], 3)[0] == 4  # nunca saiu: t + 1
    assert indice.contagem("exato", [11111], 2)[0] == 2


def test_atributos_nao_usam_o_futuro():
    df = historico_artificial(300, semente=3)
    t = 200
    antes = calcular_atributos(IndiceHistorico(df), np.arange(10_000), t)
    alterado = df.copy()
    futuro = alterado["concurso"] > t  # concursos de índice >= t
    alterado.loc[futuro, "numero"] = "00007"
    depois = calcular_atributos(IndiceHistorico(alterado), np.arange(10_000), t)
    pd.testing.assert_frame_equal(antes, depois)


def test_atributos_mudam_quando_o_passado_muda():
    df = historico_artificial(300, semente=3)
    alterado = df.copy()
    alterado.loc[alterado["concurso"] == 150, "numero"] = "00007"
    a = calcular_atributos(IndiceHistorico(df), [7], 200)
    b = calcular_atributos(IndiceHistorico(alterado), [7], 200)
    assert not a.equals(b)


def test_historico_do_numero_preserva_zeros():
    df = historico_artificial(5, gerador=lambda rng: [123, 1, 99999 - 3, 4500, 70])
    h = historico_do_numero(IndiceHistorico(df), "00123")
    assert h["numero"] == "00123" and h["vezes_exato"] == 5
    assert h["terminacoes"]["milhar"]["valor"] == "0123"


def test_concurso_incompleto_e_recusado():
    df = historico_artificial(3).iloc[:-1]
    with pytest.raises(ValueError):
        IndiceHistorico(df)


# ------------------------------------------------------------------ modelos

def test_treino_usa_somente_concursos_anteriores():
    indice = IndiceHistorico(historico_artificial(200))
    pacote = modelos.treinar(indice, 150, CONFIG_TESTE)
    assert pacote["treino"]["ultimo_concurso"] == indice.concursos[149]
    assert pacote["treino"]["fim_indice"] == 150


def test_todos_os_modelos_treinam_e_xgboost_registrado():
    indice = IndiceHistorico(historico_artificial(150))
    pacote = modelos.treinar(indice, 150, CONFIG_TESTE)
    for chave in ("estatistico", "logistica", "floresta", "gradiente", "rede_neural"):
        assert pacote["status"][chave] == "treinado"
    assert "xgboost" in pacote["status"]


def test_xgboost_ausente_fica_como_nao_disponivel(monkeypatch):
    monkeypatch.setattr(modelos, "_xgboost_disponivel", lambda: (False, "não disponível (teste)"))
    pacote = modelos.treinar(IndiceHistorico(historico_artificial(150)), 150, CONFIG_TESTE)
    assert "xgboost" not in pacote["modelos"]
    assert pacote["status"]["xgboost"].startswith("não disponível")


def test_historico_curto_para_treino():
    with pytest.raises(ValueError, match="insuficiente"):
        modelos.treinar(IndiceHistorico(historico_artificial(30)), 30, CONFIG_TESTE)


def test_ranking_cobre_as_10000_milhares():
    indice = IndiceHistorico(historico_artificial(150))
    pacote = modelos.treinar(indice, 150, CONFIG_TESTE)
    tabela, atributos = modelos.ranking_milhares(pacote, indice, 150)
    assert sorted(tabela["milhar"]) == list(range(10_000))
    assert tabela["pontuacao"].between(0, 100).all()
    assert tabela["pontuacao"].is_monotonic_decreasing
    lote = modelos.ranking_em_lote(pacote, indice, [150])
    assert lote[0]["milhar"].tolist() == tabela["milhar"].tolist()


def test_modelos_sao_gravados_e_reaproveitados(tmp_path):
    indice = IndiceHistorico(historico_artificial(150))
    banco = tmp_path / "x.db"
    _, treinou = modelos.carregar_ou_treinar(banco, (150, 150, 1, 1), indice, CONFIG_TESTE)
    _, treinou_de_novo = modelos.carregar_ou_treinar(banco, (150, 150, 1, 1), indice, CONFIG_TESTE)
    _, outra_versao = modelos.carregar_ou_treinar(banco, (151, 151, 2, 2), indice, CONFIG_TESTE)
    assert treinou and not treinou_de_novo and outra_versao


def test_arquivo_de_modelo_corrompido_e_refeito(tmp_path):
    indice = IndiceHistorico(historico_artificial(150))
    banco = tmp_path / "x.db"
    arquivo = modelos.arquivo_modelo(banco, (1, 1, 1, 1), CONFIG_TESTE, CONFIG_TESTE.semente)
    arquivo.write_bytes(bytes.fromhex("785e6b6005000107" "0086"))  # gravação interrompida
    pacote, treinou = modelos.carregar_ou_treinar(banco, (1, 1, 1, 1), indice, CONFIG_TESTE)
    assert treinou and pacote["modelos"]
    assert arquivo.stat().st_size > 1000
    assert not list(arquivo.parent.glob("*.tmp"))


# ------------------------------------------------------------------ backtest

def test_backtest_detecta_sinal_artificial():
    # Sorteio VICIADO de propósito: a dezena final é sempre 00 a 09 na segunda metade.
    def viciado(rng):
        return rng.integers(0, 9600, 5) * 10 + rng.integers(0, 10, 5) // 10 + 1

    metricas = backtest.executar(IndiceHistorico(historico_artificial(200, gerador=viciado)), CONFIG_TESTE)
    assert metricas["validado"] is True
    assert metricas["criterios"] == {"abaixo_do_acaso": True, "melhor_que_estrutura": True}
    assert metricas["conjunto_contra_estrutura"]["diferenca"] < 0
    assert metricas["testes_posicao"]["conjunto"]["media"] < 0.4


def test_backtest_em_sorteio_uniforme_nao_valida():
    metricas = backtest.executar(IndiceHistorico(historico_artificial(200, semente=11)), CONFIG_TESTE)
    assert metricas["validado"] is False
    assert metricas["concursos_testados"] == 60
    assert 0 <= metricas["precision_10"] <= 1 and 0 <= metricas["recall_10"] <= 1
    assert set(metricas["coincidencias"]) == {"exato", "milhar", "centena", "dezena"}
    assert len(metricas["por_periodo"]) == 2
    assert "conjunto" in metricas["testes_posicao"]
    assert "estrutura" in metricas["testes_posicao"]
    assert metricas["validado"] == all(metricas["criterios"].values())


def test_trava_exige_os_dois_criterios(monkeypatch):
    """Ficar abaixo de 0,5 não basta: sem superar a estrutura do universo, não valida."""
    monkeypatch.setattr(backtest, "teste_pareado", lambda a, b: {
        "n": len(a), "diferenca": 0.0, "ic95": [0.0, 0.0], "z": 0.0, "p": 1.0})

    def viciado(rng):
        return rng.integers(0, 9600, 5) * 10 + 1

    metricas = backtest.executar(IndiceHistorico(historico_artificial(200, gerador=viciado)), CONFIG_TESTE)
    assert metricas["criterios"]["abaixo_do_acaso"] is True
    assert metricas["criterios"]["melhor_que_estrutura"] is False
    assert metricas["validado"] is False


def test_posicoes_no_ranking_e_estrutura():
    assert backtest.bilhetes_validos(96_000).sum() == 96_000
    score = np.arange(10_000, dtype=float)
    u = backtest.posicoes_no_ranking(score, np.array([9999, 0]))
    assert u[0] == pytest.approx(0.5 / 10_000) and u[1] == pytest.approx(1 - 0.5 / 10_000)


def test_atributos_sem_duplicados():
    indice = IndiceHistorico(historico_artificial(1200, semente=4))  # > 1000 concursos
    X = calcular_atributos(indice, np.arange(10_000), 1150)
    assert "posicao_5_500" not in X and "posicao_5_total" not in X
    colunas = list(X.columns)
    for i, a in enumerate(colunas):
        for b in colunas[i + 1:]:
            assert not np.allclose(X[a], X[b]), f"{a} = {b}"


def test_backtest_historico_curto_informa_erro():
    with pytest.raises(ValueError, match="insuficiente"):
        backtest.executar(IndiceHistorico(historico_artificial(80)), CONFIG_TESTE)


def test_selecao_respeita_universo_e_nao_repete():
    tabela = pd.DataFrame({"milhar": [9999, 9999, 5, 1234, 0] + list(range(10, 20))})
    escolhidos = backtest.selecionar(tabela, 92_000, lambda opcoes: max(opcoes))
    numeros = [n for n, _ in escolhidos]
    assert len(numeros) == 10 and len(set(numeros)) == 10
    assert all(len(n) == 5 and 1 <= int(n) <= 92_000 for n in numeros)
    assert numeros[0] == "89999"  # 99999 está fora de 00001–92000


# ------------------------------------------------------------------ motor e banco

ALVO = {"concurso": 201, "data": "2003-11-12", "status": "estimada", "extracao": "quartou",
        "extracao_status": "estimada", "fonte": "teste", "observacoes": []}


def test_motor_grava_previsoes_sem_apagar_anteriores():
    gravar_historico(historico_artificial(200, semente=5))
    with db.conectar() as c:
        antes = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                 for t in ("contests", "winning_numbers", "generated_candidates", "imports")}
    primeira = motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE)
    segunda = motor.gerar_previsao("quartou", ALVO, nova=True, config=CONFIG_TESTE)
    assert primeira != segunda
    lista = armazenamento.listar_previsoes()
    assert set(lista["id"]) == {primeira, segunda}
    assert set(lista["tipo_execucao"]) == {motor.GERAR, motor.GERAR_NOVA}
    for identificador in (primeira, segunda):
        p = armazenamento.carregar_previsao(identificador)
        numeros = [i["numero"] for i in p["itens"]]
        assert len(numeros) == 10 and len(set(numeros)) == 10
        assert all(len(n) == 5 and n.isdigit() and 1 <= int(n) <= 92_000 for n in numeros)
        assert [i["posicao"] for i in p["itens"]] == list(range(1, 11))
        assert all(0 <= i["pontuacao"] <= 100 for i in p["itens"])
        assert p["resumo"]["probabilidade"]["chance_texto"] == "1 em 92.000"
        assert p["resumo"]["backtest"]["concursos_testados"] == 60
    with db.conectar() as c:
        depois = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in antes}
    assert antes == depois  # dados existentes intactos
    # O backtest foi gravado uma vez e reaproveitado.
    with db.conectar() as c:
        assert c.execute("SELECT count(*) FROM prediction_backtests").fetchone()[0] == 1


def test_trava_de_validacao_sem_vantagem():
    gravar_historico(historico_artificial(200, semente=11))
    p = armazenamento.carregar_previsao(motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE))
    assert p["validado"] == 0
    assert p["rotulo_principal"] == "1º do ranking do modelo"
    assert p["resumo"]["aviso"] == "Sem vantagem comprovada sobre o acaso"


def test_motor_sem_historico_informa_erro():
    db.inicializar()
    with pytest.raises(motor.ErroMotor, match="Importe"):
        motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE)


def test_conferencia_automatica_apos_resultado_oficial():
    gravar_historico(historico_artificial(200, semente=5))
    p = armazenamento.carregar_previsao(motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE))
    principal = p["itens"][0]["numero"]
    segundo = p["itens"][1]["numero"]
    centena_igual = f"{(int(segundo[:2]) + 1) % 9 + 1:02d}{segundo[2:]}"  # mesma centena, outro número
    assert armazenamento.conferir_pendentes() == []  # resultado ainda não existe
    db.salvar_concurso_manual(201, "2003-11-12", "quartou", [principal, centena_igual, "00001", "00002", "00003"])
    assert armazenamento.conferir_pendentes() == [p["id"]]
    conferida = armazenamento.carregar_previsao(p["id"])
    assert conferida["resultado"]["acertos_exatos"] >= 1
    assert conferida["resultado"]["principal_exato"] is True
    assert conferida["itens"][0]["conferencia"]["premio_principal"] == [1]
    item2 = conferida["itens"][1]["conferencia"]
    assert item2["premio_principal"] == []  # coincidência parcial não é prêmio principal
    assert armazenamento.conferir_pendentes() == []  # não confere duas vezes


def test_tabelas_novas_nao_alteram_as_existentes():
    db.inicializar()
    with db.conectar() as c:
        antes = {t: c.execute(f"PRAGMA table_info({t})").fetchall()
                 for t in ("contests", "winning_numbers", "generated_candidates", "imports")}
    armazenamento.inicializar()
    armazenamento.inicializar()  # idempotente
    with db.conectar() as c:
        depois = {t: c.execute(f"PRAGMA table_info({t})").fetchall() for t in antes}
        tabelas = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert [list(map(tuple, v)) for v in antes.values()] == [list(map(tuple, v)) for v in depois.values()]
    assert {"predictions", "prediction_items", "prediction_backtests"} <= tabelas


# ------------------------------------------------------------------ calendário

def test_calendario_quarta_para_domingo_e_estimado():
    alvo = calendario.estimar_proximo(6115, "2026-11-04", hoje=date(2026, 11, 7))
    assert alvo["concurso"] == 6116
    assert alvo["data"] == "2026-11-08"
    assert alvo["status"] == "estimada" and alvo["extracao"] == "domingo"
    assert any("ENRICOU" in o for o in alvo["observacoes"])


def test_calendario_domingo_para_quarta():
    alvo = calendario.estimar_proximo(6114, "2026-11-01", hoje=date(2026, 11, 1))
    assert alvo["data"] == "2026-11-04" and alvo["extracao"] == "quartou"


def test_calendario_data_passada_avisa():
    alvo = calendario.estimar_proximo(6115, "2026-11-04", hoje=date(2026, 11, 20))
    assert any("já passou" in o for o in alvo["observacoes"])


def test_servico_oficial_confirma_so_com_data():
    estimativa = calendario.estimar_proximo(6115, "2026-11-04", hoje=date(2026, 11, 7))
    vazio = calendario.interpretar_servico({"numeroConcursoProximo": 6116, "dataProximoConcurso": ""}, estimativa)
    assert vazio["status"] == "estimada"
    ok = calendario.interpretar_servico({"numeroConcursoProximo": 6116, "dataProximoConcurso": "11/11/2026"}, estimativa)
    assert ok["status"] == "confirmada" and ok["data"] == "2026-11-11" and ok["extracao"] == "quartou"
    outro = calendario.interpretar_servico({"numeroConcursoProximo": 6118, "dataProximoConcurso": "15/11/2026"}, estimativa)
    assert outro["status"] == "estimada" and "6118" in outro["observacoes"][0]


def test_servico_oficial_indisponivel(monkeypatch):
    def falha(*a, **k):
        raise OSError("sem rede")
    monkeypatch.setattr(calendario.urllib.request, "urlopen", falha)
    estimativa = calendario.estimar_proximo(6115, "2026-11-04", hoje=date(2026, 11, 7))
    r = calendario.consultar_oficial(estimativa)
    assert r["status"] == "estimada" and "Não foi possível" in r["observacoes"][0]
