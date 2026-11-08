"""Auditoria final. Somente dados ARTIFICIAIS."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.preditivo import auditoria_final as af
from src.preditivo.atributos import IndiceHistorico
from tests.test_preditivo import CONFIG_TESTE, historico_artificial

CONFIG = replace(CONFIG_TESTE, bloco=40)


def test_metricas_inclusivas_e_exato_separado():
    cand = ["12345", "00062", "50001", "70010", "80020", "90030", "11111", "22222", "33333", "44444"]
    m = af.linha_metricas(cand, ["12345", "99062", "40001", "55555", "67891"], 96_000)
    assert m["exato_n"] == 1
    assert m["milhar_n"] == 2      # 12345 (exato também conta) e 40001
    assert m["centena_n"] == 3     # + 99062
    assert m["dezena_n"] == 3
    assert m["unidade_n"] == 5     # unidades 5, 2, 1, 5, 1 estão entre as dos candidatos
    assert m["exato_p"] == pytest.approx(10 / 96_000)
    assert m["dezena_p"] == pytest.approx(len({c[-2:] for c in cand}) * 960 / 96_000)


def test_wilson():
    baixo, alto = af.wilson(5, 10)
    assert baixo == pytest.approx(0.2366, abs=1e-3) and alto == pytest.approx(0.7634, abs=1e-3)
    assert af.wilson(0, 10)[0] == 0.0


def test_meta_70_por_nivel():
    meta = {n: af.meta_70(n, 1000) for n in af.NIVEIS}
    assert meta["exato"]["taxa_acaso_10"] == pytest.approx(1 - (1 - 10 / 96_000) ** 5)
    assert meta["dezena"]["taxa_acaso_10"] == pytest.approx(1 - 0.9 ** 5)
    assert meta["dezena"]["numeros_necessarios_pelo_acaso"] == 22
    assert not meta["dezena"]["atingivel_por_cobertura"]
    assert meta["unidade"]["atingivel_por_cobertura"]      # 10 unidades distintas = 100% sem prever nada
    assert meta["exato"]["multiplicador_necessario"] > 1000


def test_sem_vazamento_e_reprodutivel():
    df = historico_artificial(200, semente=4)
    alterado = df.copy()
    alterado.loc[alterado["concurso"] > 160, "numero"] = "12345"
    a = af.executar(IndiceHistorico(df), inicio=100, fim=160, config=CONFIG)
    b = af.executar(IndiceHistorico(alterado), inicio=100, fim=160, config=CONFIG)
    c = af.executar(IndiceHistorico(df), inicio=100, fim=160, config=CONFIG)
    pd.testing.assert_frame_equal(a["linhas"], b["linhas"])
    pd.testing.assert_frame_equal(a["linhas"], c["linhas"])
    for k in a["posicoes"]:
        np.testing.assert_array_equal(a["posicoes"][k], b["posicoes"][k])
    assert set(a["linhas"]["estrategia"]) == set(af.ESTRATEGIAS)


def test_sorteio_uniforme_sem_vantagem(monkeypatch):
    monkeypatch.setattr(af, "CORTE_INEDITO", 160)
    r = af.executar(IndiceHistorico(historico_artificial(220, semente=11)), inicio=100, config=CONFIG)
    a = af.analisar(r, periodos=2)
    assert a["conclusao"] in ("sem_vantagem", "sinal_promissor_nao_confirmado")
    assert a["trechos"]["principal_inedito"]["vantagens_comprovadas"] == []
    assert a["trechos"]["principal_inedito"]["concursos"] == 59  # concursos 101 a 159
    assert len(a["periodos"]) == 2


def test_auditoria_detecta_sinal_artificial(monkeypatch):
    """Sensibilidade: num sorteio VICIADO de propósito, a auditoria precisa achar a vantagem."""
    monkeypatch.setattr(af, "CORTE_INEDITO", 160)

    def viciado(rng):
        return rng.integers(0, 9600, 5) * 10 + 1  # todo número termina em 1 e a dezena é 01..91

    r = af.executar(IndiceHistorico(historico_artificial(220, gerador=viciado)), inicio=100, config=CONFIG)
    a = af.analisar(r, periodos=2)
    assert a["conclusao"] == "vantagem_comprovada"
    assert "motor|posicao" in a["trechos"]["principal_inedito"]["vantagens_comprovadas"]


# ------------------------------------------------------------------ regra de replicação

def _janela(*sinais):
    return {"sinais_nominais": list(sinais)}


SEM_VANTAGEM = {"vantagens_comprovadas": []}


def test_sinal_nas_duas_janelas_e_promissor():
    r = af.classificar(SEM_VANTAGEM, _janela("motor|dezena|esperado"), _janela("motor|dezena|esperado"))
    assert r["conclusao"] == "sinal_promissor_nao_confirmado"
    assert r["sinais_replicados"] == ["motor|dezena|esperado"]


@pytest.mark.parametrize("j1, j2", [
    (_janela("motor|dezena|esperado"), _janela()),                         # só na primeira
    (_janela(), _janela("motor|dezena|esperado")),                         # só na segunda
    (_janela("motor|dezena|esperado"), _janela("motor|centena|esperado")),  # sinais diferentes
    (_janela(), _janela()),                                                # nenhum
])
def test_sinal_em_uma_janela_so_nao_e_promissor(j1, j2):
    r = af.classificar(SEM_VANTAGEM, j1, j2)
    assert r["conclusao"] == "sem_vantagem" and r["sinais_replicados"] == []


def test_so_o_sinal_comum_conta():
    r = af.classificar(SEM_VANTAGEM, _janela("a", "b", "c"), _janela("c", "d"))
    assert r["sinais_replicados"] == ["c"]


def test_vantagem_comprovada_tem_prioridade():
    r = af.classificar({"vantagens_comprovadas": ["motor|posicao"]}, _janela(), _janela())
    assert r["conclusao"] == "vantagem_comprovada"


def test_janela_usa_p_menor_que_005_e_direcao_favoravel():
    """Na janela, entra só p < 0,05 favorável; p >= 0,05 ou direção contrária ficam de fora."""
    rng = np.random.default_rng(0)
    concursos = np.arange(1, 401)
    linhas = []
    for c in concursos:
        for e in af.ESTRATEGIAS:
            linha = {"concurso": int(c), "estrategia": e, "limite": 96_000}
            for nivel in af.NIVEIS:
                linha[f"{nivel}_p"] = 0.1
                linha[f"{nivel}_n"] = int(rng.random() < 0.1)
            if e == "motor":
                linha["dezena_n"] = 1  # 100% contra 41% esperado: favorável, p minúsculo
                linha["unidade_n"] = 0  # 0% contra 41%: desfavorável
            linhas.append(linha)
    df = pd.DataFrame(linhas)
    pos = {k: rng.random(5 * len(concursos)) for k in ("motor", "estrutura", "aleatoria")}
    t = af.analisar_trecho(df, pos, lambda c: c <= 200)
    assert "motor|dezena|esperado" in t["sinais_nominais"]
    assert "motor|unidade|esperado" not in t["sinais_nominais"]
    assert "motor|unidade|esperado" in t["desfavoraveis_nominais"]
    assert all(t["p_holm"][k] >= 0 for k in t["sinais_nominais"])


def test_analisar_divide_o_inedito_em_duas_janelas(monkeypatch):
    monkeypatch.setattr(af, "CORTE_INEDITO", 180)
    r = af.executar(IndiceHistorico(historico_artificial(220, semente=11)), inicio=100, config=CONFIG)
    a = af.analisar(r, periodos=2)
    j1, j2 = a["replicacao"]["janela_1"], a["replicacao"]["janela_2"]
    assert j1["ate"] < j2["de"]                              # janelas distintas e em ordem
    assert j1["concursos"] + j2["concursos"] == a["trechos"]["principal_inedito"]["concursos"]
    assert j1["de"] == 101 and j2["ate"] == 179
    assert a["sinais_replicados"] == af.sinais_replicados(j1, j2)
    esperado = af.classificar(a["trechos"]["principal_inedito"], j1, j2)["conclusao"]
    assert a["conclusao"] == esperado
    assert "Replicação no trecho inédito" in af.gerar_relatorio(a)
