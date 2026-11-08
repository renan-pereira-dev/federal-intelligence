"""Importação do resultado oficial. Respostas SIMULADAS do serviço e banco temporário (sem internet)."""

import io
import json
import urllib.error
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import database as db
from src import oficial
from tests.test_preditivo import gravar_historico, historico_artificial

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HOJE = date(2026, 11, 8)


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setenv("FI_DB_PATH", str(tmp_path / "oficial.db"))
    monkeypatch.setenv("FI_MOTOR_CONFIG", "rapido")


def resposta(numero, data="08/11/2026", numeros=("012345", "054321", "000777", "099999", "010001"),
             valores=(500000, 35000, 30000, 25000, 20000), proximo=None):
    return {
        "numero": numero, "dataApuracao": data, "listaDezenas": list(numeros),
        "dezenasSorteadasOrdemSorteio": list(numeros), "numeroConcursoProximo": proximo or numero + 1,
        "dataProximoConcurso": "", "localSorteio": "ESPAÇO DA SORTE", "observacao": "",
        "listaRateioPremio": [{"faixa": i, "descricaoFaixa": f"{i} acertos", "valorPremio": float(v)}
                              for i, v in enumerate(valores, 1)],
    }


def servico(respostas: dict, ultimo: int):
    """Simula buscar_json: dict concurso -> JSON; concurso ausente -> indisponível."""
    def buscar(concurso=None):
        n = ultimo if concurso is None else concurso
        if n not in respostas:
            raise oficial.ConcursoIndisponivel(f"Concurso {n} não disponível no serviço (HTTP 500).")
        return respostas[n]
    return buscar


def _base(concursos=3):
    """Banco com concursos 1..N artificiais, datas em 2026."""
    db.inicializar()
    for c in range(1, concursos + 1):
        db.salvar_concurso_manual(c, f"2026-10-{c:02d}", "", [f"{c:05d}", "11111", "22222", "33333", "44444"])


# ------------------------------------------------------------------ interpretação

def test_interpreta_seis_digitos_e_preserva_zeros():
    r = oficial.interpretar(resposta(10), 10, hoje=HOJE)
    assert r.premios == ["12345", "54321", "00777", "99999", "10001"]
    assert r.data == "2026-11-08" and r.premio_zero is False
    assert r.dados_originais["valor_nominal_1"] == "500000"
    assert r.dados_originais["lista_original"][2] == "000777"


def test_premio_zero_so_com_os_cinco_valores_zerados():
    assert oficial.interpretar(resposta(10, valores=(0, 0, 0, 0, 0)), 10, HOJE).premio_zero is True
    sem_rateio = resposta(10)
    sem_rateio["listaRateioPremio"] = []
    assert oficial.interpretar(sem_rateio, 10, HOJE).premio_zero is False


@pytest.mark.parametrize("alterar, trecho", [
    (lambda d: d.update(numero=11), "veio o 11"),
    (lambda d: d.update(dataApuracao="09/11/2026"), "futuro"),
    (lambda d: d.update(dataApuracao="31/02/2026"), "data inválida"),
    (lambda d: d.update(listaDezenas=["012345"] * 4, dezenasSorteadasOrdemSorteio=[]), "5 números"),
    (lambda d: d.update(listaDezenas=["000000"] * 5, dezenasSorteadasOrdemSorteio=[]), "00000"),
    (lambda d: d.update(listaDezenas=["1234567"] * 5, dezenasSorteadasOrdemSorteio=[]), "dígitos"),
    (lambda d: d.update(dezenasSorteadasOrdemSorteio=["011111"] * 5), "não coincidem"),
    (lambda d: d.update(numero=None), "inválido"),
])
def test_respostas_invalidas_sao_recusadas(alterar, trecho):
    dados = resposta(10)
    alterar(dados)
    with pytest.raises(oficial.ErroValidacao, match=trecho):
        oficial.interpretar(dados, 10, hoje=HOJE)


# ------------------------------------------------------------------ consulta HTTP

class _Resposta(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_buscar_json_e_erros_de_rede():
    ok = oficial.buscar_json(5, abrir=lambda url, t: _Resposta(json.dumps(resposta(5)).encode()))
    assert ok["numero"] == 5

    def http500(url, t):
        raise urllib.error.HTTPError(url, 500, "erro", {}, None)
    with pytest.raises(oficial.ConcursoIndisponivel):
        oficial.buscar_json(6116, abrir=http500)

    chamadas = []

    def sem_rede(url, t):
        chamadas.append(url)
        raise urllib.error.URLError("sem rede")
    with pytest.raises(oficial.ErroServico, match="internet"):
        oficial.buscar_json(None, abrir=sem_rede)
    assert len(chamadas) == oficial.TENTATIVAS  # tentou de novo antes de desistir

    with pytest.raises(oficial.ErroServico, match="JSON"):
        oficial.buscar_json(5, abrir=lambda url, t: _Resposta(b"<html>"))


# ------------------------------------------------------------------ prévia e gravação

def test_previa_nao_grava_e_classifica():
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("000003", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026"), 5: resposta(5, "07/11/2026")}
    previa = oficial.previsualizar(buscar=servico(respostas, 5), hoje=HOJE)
    assert [(i["concurso"], i["situacao"]) for i in previa["itens"]] == [
        (3, oficial.IDENTICO), (4, oficial.NOVO), (5, oficial.NOVO)]
    assert oficial.resumo_previa(previa)["pode_gravar"]
    assert db.buscar_concurso(4) is None  # prévia não grava


def test_gravar_novos_em_uma_importacao():
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("000003", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026", valores=(0, 0, 0, 0, 0)), 5: resposta(5, "07/11/2026")}
    previa = oficial.previsualizar(buscar=servico(respostas, 5), hoje=HOJE)
    versao_antes = db.versao_historico()
    r = oficial.gravar(previa)
    assert r["gravados"] == [4, 5]
    assert db.buscar_concurso(5)["premios"] == ["12345", "54321", "00777", "99999", "10001"]
    assert db.versao_historico() != versao_antes  # cache e modelos percebem a mudança
    historico = db.carregar_historico()
    assert bool(historico.loc[historico["concurso"] == 4, "premio_zero"].all())
    with db.conectar() as c:
        linha = c.execute("SELECT arquivo, linhas_duplicadas, relatorio FROM imports WHERE id = ?",
                          (r["import_id"],)).fetchone()
    assert linha["arquivo"] == oficial.FONTE and linha["linhas_duplicadas"] == 1
    assert json.loads(linha["relatorio"])["concursos"] == [4, 5]
    # Repetir não duplica nada.
    previa2 = oficial.previsualizar(buscar=servico(respostas, 5), hoje=HOJE)
    assert {i["situacao"] for i in previa2["itens"]} == {oficial.IDENTICO}
    assert oficial.gravar(previa2)["gravados"] == []


def test_conflito_nao_grava_nada():
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("099999", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026")}
    previa = oficial.previsualizar(buscar=servico(respostas, 4), hoje=HOJE)
    assert previa["itens"][0]["situacao"] == oficial.CONFLITO
    assert not oficial.resumo_previa(previa)["pode_gravar"]
    with pytest.raises(oficial.ErroValidacao, match="conflito"):
        oficial.gravar(previa)
    assert db.buscar_concurso(4) is None
    assert db.buscar_concurso(3)["premios"][0] == "00003"  # banco intacto


def test_lacuna_no_servico_e_limite_por_consulta():
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("000003", "011111", "022222", "033333", "044444")),
                 5: resposta(5, "05/11/2026"), 6: resposta(6, "06/11/2026"), 7: resposta(7, "07/11/2026")}
    previa = oficial.previsualizar(buscar=servico(respostas, 7), maximo=3, hoje=HOJE)
    assert [(i["concurso"], i["situacao"]) for i in previa["itens"]] == [
        (3, oficial.IDENTICO), (4, oficial.INDISPONIVEL), (5, oficial.NOVO), (6, oficial.NOVO)]
    assert previa["restantes"] == 1
    assert oficial.gravar(previa)["gravados"] == [5, 6]


def test_sem_internet_nada_muda():
    _base(3)

    def fora(concurso=None):
        raise oficial.ErroServico("Não foi possível consultar o serviço oficial (URLError).")
    previa = oficial.previsualizar(buscar=fora, hoje=HOJE)
    assert previa["erro"] and previa["itens"] == []
    with pytest.raises(oficial.ErroValidacao):
        oficial.gravar(previa)


def test_banco_vazio_traz_so_o_ultimo():
    previa = oficial.previsualizar(buscar=servico({9: resposta(9, "07/11/2026")}, 9), hoje=HOJE)
    assert [i["concurso"] for i in previa["itens"]] == [9]


def test_conflito_surgido_depois_da_previa_e_barrado():
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("000003", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026")}
    previa = oficial.previsualizar(buscar=servico(respostas, 4), hoje=HOJE)
    db.salvar_concurso_manual(4, "2026-11-04", "", ["55555"] * 5)  # outra sessão gravou diferente
    with pytest.raises(oficial.ErroValidacao, match="mudou no banco"):
        oficial.gravar(previa)


# ------------------------------------------------------------------ conferência automática

def test_gravar_confere_as_previsoes():
    from src.preditivo import armazenamento, motor
    from tests.test_preditivo import ALVO, CONFIG_TESTE

    gravar_historico(historico_artificial(200, semente=5))  # concursos 1..200, datas antigas
    p = armazenamento.carregar_previsao(motor.gerar_previsao("quartou", ALVO, config=CONFIG_TESTE))
    principal = p["itens"][0]["numero"]
    with db.conectar() as c:
        candidatos = c.execute("SELECT count(*) FROM generated_candidates").fetchone()[0]
    ultimo = db.buscar_concurso(200)
    respostas = {
        200: resposta(200, f"{ultimo['data'][8:]}/{ultimo['data'][5:7]}/{ultimo['data'][:4]}",
                      ["0" + n for n in ultimo["premios"]]),
        201: resposta(201, "12/11/2003", ["0" + principal, "011111", "022222", "033333", "044444"]),
    }
    previa = oficial.previsualizar(buscar=servico(respostas, 201), hoje=HOJE)
    r = oficial.gravar(previa)
    assert r["gravados"] == [201] and r["previsoes_conferidas"] == [p["id"]]
    conferida = armazenamento.carregar_previsao(p["id"])
    assert conferida["resultado"]["principal_exato"] is True
    assert len(armazenamento.listar_previsoes()) == 1  # nada apagado
    with db.conectar() as c:
        assert c.execute("SELECT count(*) FROM generated_candidates").fetchone()[0] == candidatos


# ------------------------------------------------------------------ interface

def _abrir(pagina):
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value(pagina).run()
    assert not at.exception, at.exception
    return at


def _botao(at, inicio):
    return [b for b in at.button if b.label.startswith(inicio)][0]


def test_interface_previa_e_confirmacao(monkeypatch):
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("000003", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026")}
    monkeypatch.setattr(oficial, "buscar_json", servico(respostas, 4))
    at = _abrir("Histórico")
    _botao(at, "Buscar resultados oficiais").click().run()
    assert not at.exception, at.exception
    assert db.buscar_concurso(4) is None  # só a prévia
    _botao(at, "Confirmar e gravar 1").click().run()
    assert not at.exception, at.exception
    assert db.buscar_concurso(4)["premios"][0] == "12345"
    assert any("Gravado(s): concurso(s) 4" in s.value for s in at.success)
    assert at.metric[0].value == "4"  # cache atualizado: a tela já vê 4 concursos


def test_interface_conflito_sem_botao_de_gravar(monkeypatch):
    _base(3)
    respostas = {3: resposta(3, "03/10/2026", ("099999", "011111", "022222", "033333", "044444")),
                 4: resposta(4, "04/11/2026")}
    monkeypatch.setattr(oficial, "buscar_json", servico(respostas, 4))
    at = _abrir("Histórico")
    _botao(at, "Buscar resultados oficiais").click().run()
    assert any("conflito" in e.value for e in at.error)
    assert not [b for b in at.button if b.label.startswith("Confirmar e gravar")]


def test_interface_sem_internet(monkeypatch):
    _base(3)

    def fora(concurso=None):
        raise oficial.ErroServico("Não foi possível consultar o serviço oficial (URLError).")
    monkeypatch.setattr(oficial, "buscar_json", fora)
    at = _abrir("Previsão do próximo sorteio")
    _botao(at, "Buscar resultados oficiais").click().run()
    assert not at.exception, at.exception
    assert any("Nada foi alterado" in e.value for e in at.error)
