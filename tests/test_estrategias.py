"""Laboratório de estratégias. Somente dados ARTIFICIAIS e bancos temporários."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.preditivo import estrategias as es
from src.preditivo.atributos import IndiceHistorico
from src.preditivo.backtest import bilhetes_validos
from tests.test_preditivo import CONFIG_TESTE, gravar_historico, historico_artificial

CONFIG = replace(CONFIG_TESTE, bloco=20)
APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "estrategias.db"))


# ------------------------------------------------------------------ seleção e diversidade

def _ordem_agrupada():
    """Ranking 'parecido': as primeiras milhares compartilham a mesma centena e unidade."""
    primeiras = [1062, 2062, 3062, 4062, 5062, 6062, 7062, 8062, 9062, 62, 1063, 2063]
    resto = [m for m in range(10_000) if m not in primeiras]
    return np.array(primeiras + resto)


@pytest.mark.parametrize("regra", list(es.NOMES_SELECOES))
def test_selecoes_sem_duplicidade_e_no_universo(regra):
    rng = np.random.default_rng(1)
    validas = bilhetes_validos(92_000) > 0
    cand = es.selecionar(regra, _ordem_agrupada(), validas, 92_000, rng)
    assert len(cand) == 10 and len(set(cand)) == 10
    assert all(len(c) == 5 and c.isdigit() and 1 <= int(c) <= 92_000 for c in cand)


def test_diversificacao_aumenta_cobertura():
    rng = np.random.default_rng(2)
    validas = bilhetes_validos(96_000) > 0
    trad = es.diversidade(es.selecionar("tradicional", _ordem_agrupada(), validas, 96_000, rng))
    assert trad["centenas"] == 1 and trad["unidades"] == 1
    assert es.diversidade(es.selecionar("div_unidade", _ordem_agrupada(), validas, 96_000, rng))["unidades"] == 10
    assert es.diversidade(es.selecionar("div_centena", _ordem_agrupada(), validas, 96_000, rng))["centenas"] == 10
    mmr = es.diversidade(es.selecionar("mmr_0.25", _ordem_agrupada(), validas, 96_000, rng))
    assert mmr["semelhanca_media"] < trad["semelhanca_media"]


def test_selecao_aleatoria_sem_reposicao():
    cand = es.selecao_aleatoria(84_000, np.random.default_rng(3))
    assert len(set(cand)) == 10 and all(1 <= int(c) <= 84_000 for c in cand)


# ------------------------------------------------------------------ métricas

def test_contagem_de_terminacoes_confere_com_forca_bruta():
    for limite in (92_000, 96_000, 85_000, 99_999):
        numeros = np.arange(1, limite + 1)
        for d, finais in ((2, ["00", "07", "99"]), (3, ["000", "062", "999"]), (4, ["0000", "6000", "6001"])):
            esperado = int(np.isin(numeros % 10**d, [int(f) for f in finais]).sum())
            assert es.contagem_terminacoes(finais, d, limite) == esperado


def test_metricas_separam_exato_de_parcial():
    cand = ["12345", "00062", "50001", "70010", "80020", "90030", "11111", "22222", "33333", "44444"]
    sorteados = ["12345", "99062", "40001", "55555", "12345"]
    m = es.metricas_selecao(cand, sorteados, 96_000)
    assert m["exato"] == 2                      # o 12345 saiu em dois prêmios
    assert m["principal"] == 1
    assert m["milhar"] == 1                     # 40001 tem a milhar de 50001
    assert m["centena"] == 2                    # 99062 (062) e 40001 (001); exatos não contam
    assert m["dezena"] == 2
    assert m["precision"] == pytest.approx(1 / 10)
    assert m["recall"] == pytest.approx(2 / 5)
    assert m["esperado_exato"] == pytest.approx(5 * 10 / 96_000)


def test_selecao_aleatoria_fica_no_esperado():
    rng = np.random.default_rng(4)
    linhas = []
    for _ in range(3000):
        sorteados = [f"{v:05d}" for v in rng.integers(1, 96_001, 5)]
        linhas.append(es.metricas_selecao(es.selecao_aleatoria(96_000, rng), sorteados, 96_000) | {
            "algarismos_por_posicao": 0, "unidades": 0, "dezenas": 0, "centenas": 0, "milhares": 0,
            "semelhanca_media": 0})
    r = es.resumo(pd.DataFrame(linhas), None)
    for nome in ("centena", "dezena"):
        assert abs(r[nome]["z_excesso"]) < 3.5


def test_controle_estrutural_no_ranking():
    indice = IndiceHistorico(historico_artificial(120))
    from src.preditivo.atributos import calcular_atributos
    X = calcular_atributos(indice, np.arange(10_000), 120)
    notas = es.rankings_base(X, {}, {}, 96_000, np.random.default_rng(0))
    np.testing.assert_array_equal(notas["estrutura"], bilhetes_validos(96_000))
    assert notas["estrutura"][6000] > notas["estrutura"][6001]


def test_holm():
    assert es.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


# ------------------------------------------------------------------ protocolo

def test_avaliacao_reprodutivel():
    indice = IndiceHistorico(historico_artificial(120, semente=6))
    a = es.avaliar(indice, 80, 100, CONFIG)
    b = es.avaliar(indice, 80, 100, CONFIG)
    assert set(a["registros"]) == set(b["registros"])
    for k in a["registros"]:
        pd.testing.assert_frame_equal(a["registros"][k], b["registros"][k])
    assert len(a["registros"]) == 13 * len(es.NOMES_SELECOES) + 1


@pytest.fixture(scope="module")
def experimentos():
    df = historico_artificial(180, semente=7)
    alterado = df.copy()
    alterado.loc[alterado["concurso"] > 140, "numero"] = "12345"  # só a amostra final muda
    kw = dict(final=40, desenvolvimento=40, periodos=2, config=CONFIG)
    return es.executar(IndiceHistorico(df), **kw), es.executar(IndiceHistorico(alterado), **kw)


def test_pre_registro_nao_ve_a_amostra_final(experimentos):
    original, alterado = experimentos
    assert original["divisao"]["final"] == [141, 180]
    assert original["pre_registro"]["candidatas"] == alterado["pre_registro"]["candidatas"]
    assert original["desenvolvimento"] == alterado["desenvolvimento"]
    assert original["final"] != alterado["final"]


def test_relatorio_completo_e_criterios(experimentos):
    r = experimentos[0]
    assert r["quantidade_estrategias"] == 92
    for ref in (es.SELECAO_ALEATORIA, es.MOTOR_PADRAO, es.CONTROLE_ESTRUTURAL):
        assert ref in r["final"]
    for c, a in r["aprovacao"].items():
        assert set(a["criterios"]) == {"supera_aleatoria", "excesso_sobre_cobertura",
                                       "supera_estrutura", "supera_motor_padrao"}
        assert a["aprovada"] == all(a["criterios"].values())
    assert len(r["por_periodo"]) == 2
    assert all(0 <= v["p_holm"] <= 1 for v in r["exploratorio"].values())
    json.dumps({k: v for k, v in r.items() if k != "_dados"}, default=es._json)  # pode ser gravado


def test_criterios_recalculados_dos_dados_brutos(experimentos, tmp_path):
    import joblib
    r = experimentos[0]
    dados = r["_dados"]
    refeito = es.montar_relatorio(dados["desenvolvimento"], dados["final"], dados["pre_registro"],
                                  r["divisao"], 2, CONFIG)
    assert refeito["aprovacao"] == r["aprovacao"] and refeito["final"] == r["final"]
    arquivo = es.salvar(r, tmp_path / "x.db")
    assert arquivo.with_suffix(".dados.joblib").exists()
    assert "_dados" not in json.loads(arquivo.read_text(encoding="utf-8"))
    assert set(joblib.load(arquivo.with_suffix(".dados.joblib"))) == {"desenvolvimento", "final", "pre_registro"}


# ------------------------------------------------------------------ ativação reversível

def test_ativar_so_aprovada_e_reverter(tmp_path):
    banco = tmp_path / "x.db"
    assert es.estrategia_ativa(banco)["estrategia"] == es.MOTOR_PADRAO
    relatorio = {"aprovacao": {"ens_igual|div_centena": {"aprovada": False},
                               "freq_recente|tradicional": {"aprovada": True}}, "pesos_ponderado": {}}
    with pytest.raises(ValueError):
        es.ativar("ens_igual|div_centena", relatorio, "r.json", banco)
    es.ativar("freq_recente|tradicional", relatorio, "r.json", banco)
    assert es.estrategia_ativa(banco)["estrategia"] == "freq_recente|tradicional"
    voltou = es.reverter(banco)
    assert voltou["estrategia"] == es.MOTOR_PADRAO and voltou["historico"] == []


def test_motor_usa_padrao_sem_ativacao_e_estrategia_quando_ativada():
    from src import database as db
    from src.preditivo import armazenamento, motor
    from tests.test_preditivo import ALVO

    gravar_historico(historico_artificial(200, semente=5))
    with db.conectar() as c:
        antes = [c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                 for t in ("contests", "winning_numbers", "generated_candidates", "imports")]
    p1 = armazenamento.carregar_previsao(motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE))
    assert p1["resumo"]["estrategia_selecao"] == es.MOTOR_PADRAO
    arquivo = es.pasta(db.resolver_caminho()) / "estrategia_ativa.json"
    arquivo.write_text(json.dumps({"estrategia": "ens_igual|div_unidade", "relatorio": "teste",
                                   "pesos": None, "historico": []}), encoding="utf-8")
    p2 = armazenamento.carregar_previsao(motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE))
    numeros = [i["numero"] for i in p2["itens"]]
    assert p2["resumo"]["estrategia_selecao"] == "ens_igual|div_unidade"
    assert len({n[-1] for n in numeros}) == 10 and len(set(numeros)) == 10
    es.reverter(db.resolver_caminho())
    assert es.estrategia_ativa(db.resolver_caminho())["estrategia"] == es.MOTOR_PADRAO
    with db.conectar() as c:
        depois = [c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("contests", "winning_numbers", "generated_candidates", "imports")]
    assert antes == depois
    assert len(armazenamento.listar_previsoes()) == 2  # nada apagado


# ------------------------------------------------------------------ interface

def _abrir():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Laboratório de estratégias").run()
    assert not at.exception, at.exception
    return at


def test_laboratorio_sem_experimento():
    at = _abrir()
    assert any("Nenhum experimento" in i.value for i in at.info)


def test_laboratorio_com_experimento(experimentos):
    from src import database as db
    gravar_historico(historico_artificial(220, semente=8))
    es.salvar(experimentos[0], db.resolver_caminho())
    at = _abrir()
    assert [t.label for t in at.tabs] == ["Candidatos", "Aprovação", "Comparativo (final)", "Desenvolvimento",
                                          "Diversidade", "Por período", "Configuração ativa"]
    html = " ".join(m.value for m in at.markdown)
    if not experimentos[0]["alguma_aprovada"]:
        assert "Sem vantagem comprovada sobre o acaso" in html
    [b for b in at.button if b.label == "Gerar os 10 candidatos desta estratégia"][0].click().run()
    assert not at.exception, at.exception
    assert "Unidades distintas" in [m.label for m in at.metric]
