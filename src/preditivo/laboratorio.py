"""Área experimental "Laboratório de estratégias" (Streamlit).

Mostra o último experimento gravado em data/experimentos/estrategias_*.json e
gera, sob demanda, os 10 candidatos de qualquer estratégia para o próximo
concurso. Nada aqui altera o motor padrão; a ativação só é possível para uma
estratégia aprovada e pode ser desfeita.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src.preditivo import calendario, estrategias as es
from src.preditivo.atributos import IndiceHistorico
from src.preditivo.modelos import CONFIG_PADRAO, carregar_ou_treinar
from src.probability import EXTRACOES, extracoes_habilitadas

AVISO = (
    "Área experimental. A pontuação é a posição da milhar no ranking da estratégia (100 = primeira), "
    "<b>não é probabilidade</b>. Em um sorteio justo, todos os números elegíveis têm a mesma chance. "
    "Diversificar os candidatos aumenta a cobertura de terminações, não a chance do prêmio principal."
)


def _f(v, casas=4):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


def _aviso(texto: str) -> None:
    st.markdown(f'<div class="fi-aviso">{texto}</div>', unsafe_allow_html=True)


def tabela_estrategias(tab: dict, exploratorio: dict | None = None) -> pd.DataFrame:
    linhas = []
    for chave, r in tab.items():
        pos = r.get("posicao")
        c = r["centena"]
        linhas.append({
            "Estratégia": es.nome_estrategia(chave),
            "Exatos": r["exato"]["observado"],
            "Exatos esperados": round(r["exato"]["esperado_cobertura"], 2),
            "Milhar": r["milhar"]["observado"],
            "Milhar esperada": round(r["milhar"]["esperado_cobertura"], 2),
            "Centena": c["observado"],
            "Centena IC95%": f"{c['ic95'][0]:.1f}–{c['ic95'][1]:.1f}",
            "Centena esperada (cobertura)": round(c["esperado_cobertura"], 2),
            "Excesso centena (z)": round(c["z_excesso"], 2),
            "Dezena": r["dezena"]["observado"],
            "Dezena esperada": round(r["dezena"]["esperado_cobertura"], 2),
            "Precision@10": r["precision_10"],
            "Recall@10": r["recall_10"],
            "Posição média": None if not pos else round(pos["media"], 4),
            "Posição IC95%": "—" if not pos else f"{pos['ic95'][0]:.4f}–{pos['ic95'][1]:.4f}",
            "Centenas distintas": round(r["diversidade"]["centenas"], 2),
            "Dezenas distintas": round(r["diversidade"]["dezenas"], 2),
            "p Holm vs aleatória (centena)": None if not exploratorio or chave not in exploratorio
            else round(exploratorio[chave]["p_holm"], 3),
            "_chave": chave,
        })
    return pd.DataFrame(linhas).sort_values("Excesso centena (z)", ascending=False)


def pagina_laboratorio() -> None:
    st.title("Laboratório de estratégias")
    st.caption("Experimentos de ranking, combinação e diversificação dos 10 candidatos, comparados com o "
               "acaso, com o motor padrão e com o controle estrutural.")
    _aviso(AVISO)
    caminho = db.resolver_caminho()
    ultimo = es.ultimo_relatorio(caminho)
    if ultimo is None:
        st.info("Nenhum experimento foi executado ainda. Para rodar (cerca de 25 minutos, só leitura do banco): "
                "`.venv\\Scripts\\python.exe -m src.preditivo.estrategias`")
        return
    arquivo, r = ultimo
    d = r["divisao"]
    st.caption(f"Experimento {arquivo.name} · {r['criado_em']} · {r['quantidade_estrategias']} estratégias · "
               f"desenvolvimento {d['desenvolvimento'][0]}–{d['desenvolvimento'][1]} · amostra final "
               f"{d['final'][0]}–{d['final'][1]}")

    if not r["alguma_aprovada"]:
        _aviso("<b>Sem vantagem comprovada sobre o acaso.</b> Nenhuma estratégia pré-registrada cumpriu todos os "
               "critérios na amostra final. O motor padrão continua em uso.")

    abas = st.tabs(["Candidatos", "Aprovação", "Comparativo (final)", "Desenvolvimento",
                    "Diversidade", "Por período", "Configuração ativa"])
    with abas[0]:
        _aba_candidatos(r, caminho)
    with abas[1]:
        _aba_aprovacao(r)
    with abas[2]:
        _aba_comparativo(r["final"], r.get("exploratorio"), "final")
    with abas[3]:
        st.caption("Dentro da amostra: foi aqui que as candidatas foram escolhidas. Resultados bons aqui podem "
                   "ser sorte da escolha entre muitas estratégias.")
        _aba_comparativo(r["desenvolvimento"], None, "dev")
    with abas[4]:
        _aba_diversidade(r)
    with abas[5]:
        _aba_periodos(r)
    with abas[6]:
        _aba_ativa(r, arquivo, caminho)


def _aba_candidatos(r: dict, caminho) -> None:
    chaves = r["estrategias_testadas"]
    padrao = chaves.index(es.MOTOR_PADRAO) if es.MOTOR_PADRAO in chaves else 0
    estrategia = st.selectbox("Estratégia", chaves, index=padrao, format_func=es.nome_estrategia, key="lab_estr")
    opcoes = [e.chave for e in extracoes_habilitadas()]
    extracao = st.selectbox("Extração (define o universo)", opcoes, format_func=lambda k: EXTRACOES[k].nome,
                            index=opcoes.index("domingo") if "domingo" in opcoes else 0, key="lab_ext")
    if not st.button("Gerar os 10 candidatos desta estratégia", key="lab_gerar"):
        st.caption("Os candidatos são calculados na hora, com o histórico atual e os modelos já treinados. "
                   "Não são gravados no banco.")
        return
    historico = db.carregar_historico(caminho)
    if historico.empty:
        st.warning("Os dados históricos precisam ser carregados.")
        return
    with st.spinner("Calculando..."):
        indice = IndiceHistorico(historico)
        pacote, _ = carregar_ou_treinar(caminho, db.versao_historico(caminho), indice, CONFIG_PADRAO)
        res = es.candidatos_proximo(indice, pacote, estrategia, EXTRACOES[extracao].numero_maximo,
                                    r.get("pesos_ponderado"))
    ultimo = int(indice.concursos[-1])
    alvo = calendario.estimar_proximo(ultimo, indice.datas[-1])
    st.markdown(f"**Concurso {alvo['concurso']}** (data {alvo['status']}: {alvo['data']}) · "
                f"{es.nome_estrategia(estrategia)}")
    cartoes = "".join(
        f'<div class="fi-card"><div class="fi-rotulo">{i}º</div><div class="fi-numero">{n}</div>'
        f'<div class="fi-valor">Pontuação do ranking: {"—" if p is None else _f(p, 2)}</div></div>'
        for i, (n, p) in enumerate(zip(res["candidatos"], res["pontuacao"]), 1))
    st.markdown(f'<div class="fi-grid">{cartoes}</div>', unsafe_allow_html=True)
    dv = res["diversidade"]
    c = st.columns(5)
    c[0].metric("Unidades distintas", dv["unidades"])
    c[1].metric("Dezenas distintas", dv["dezenas"])
    c[2].metric("Centenas distintas", dv["centenas"])
    c[3].metric("Milhares distintas", dv["milhares"])
    c[4].metric("Semelhança média", _f(dv["semelhanca_media"], 3))
    aprov = r["aprovacao"].get(estrategia)
    if not aprov or not aprov["aprovada"]:
        _aviso("Sem vantagem comprovada sobre o acaso para esta estratégia.")


def _aba_aprovacao(r: dict) -> None:
    pr = r["pre_registro"]
    st.markdown(f"**Pré-registro** (feito só com o desenvolvimento, em {pr['criado_em']}, impressão "
                f"`{pr['impressao']}`)")
    for c in pr["candidatas"]:
        st.caption(f"{es.nome_estrategia(c)}: {pr['regras'][c]}")
    nomes = {"supera_aleatoria": "a) mais centenas que a seleção aleatória",
             "excesso_sobre_cobertura": "b) excesso sobre a própria cobertura",
             "supera_estrutura": "c) posição melhor que o controle estrutural",
             "supera_motor_padrao": "d) melhor que o motor padrão"}
    linhas = []
    for c, a in r["aprovacao"].items():
        linha = {"Candidata": es.nome_estrategia(c)}
        for k, rotulo in nomes.items():
            letra = rotulo[0]
            linha[rotulo] = f"{'sim' if a['criterios'][k] else 'não'} (p Holm {_f(a['p_holm'][letra], 3)})"
        linha["Aprovada"] = "sim" if a["aprovada"] else "não"
        linhas.append(linha)
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption("Correção de Holm na família das candidatas × 4 critérios. Todos os critérios precisam ser cumpridos.")


def _aba_comparativo(tab: dict, exploratorio: dict | None, chave: str) -> None:
    df = tabela_estrategias(tab, exploratorio)
    so_principais = st.toggle("Mostrar só referências e as melhores 10", value=True, key=f"lab_filtro_{chave}")
    if so_principais:
        refs = df["_chave"].isin(es.REFERENCIAS)
        df = pd.concat([df[refs], df[~refs].head(10)])
    st.dataframe(df.drop(columns="_chave"), hide_index=True)
    st.caption("Coincidências parciais excluem os acertos exatos. 'Esperada (cobertura)' é o que o acaso daria "
               "com as mesmas terminações cobertas; o excesso (z) mede previsão, não cobertura. Precision@10 = "
               "candidatos iguais a algum prêmio ÷ 10; Recall@10 = prêmios encontrados entre os candidatos ÷ 5.")


def _aba_diversidade(r: dict) -> None:
    st.write("Pergunta: escolher 10 candidatos parecidos reduz a cobertura? E diversificar melhora o resultado "
             "além do que a cobertura explica?")
    linhas = []
    fin = r["final"]
    for k in [es.MOTOR_PADRAO, *r["diversidade"], es.SELECAO_ALEATORIA]:
        f = fin[k]
        comp = r["diversidade"].get(k, {})
        linhas.append({
            "Seleção": es.nome_estrategia(k),
            "Centenas distintas": round(f["diversidade"]["centenas"], 2),
            "Dezenas distintas": round(f["diversidade"]["dezenas"], 2),
            "Centena observada": f["centena"]["observado"],
            "Centena esperada (cobertura)": round(f["centena"]["esperado_cobertura"], 2),
            "Excesso (z)": round(f["centena"]["z_excesso"], 2),
            "vs motor: dif. por concurso": _f(comp.get("bruto_vs_motor", {}).get("centena", {}).get("diferenca"), 4),
            "vs motor: p": _f(comp.get("bruto_vs_motor", {}).get("centena", {}).get("p"), 3),
            "vs aleatória: p": _f(comp.get("bruto_vs_aleatoria", {}).get("centena", {}).get("p"), 3),
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption("Se o excesso (z) fica perto de zero em todas, a diferença bruta vem só da cobertura de "
               "terminações, e não de previsão. p sem correção; veja a aba Aprovação para os ajustados.")


def _aba_periodos(r: dict) -> None:
    chaves = [k for k in r["por_periodo"][0] if k not in ("de", "ate")]
    fig = go.Figure()
    for k in chaves:
        fig.add_trace(go.Scatter(
            x=[f"{p['de']}–{p['ate']}" for p in r["por_periodo"]],
            y=[p[k]["centena"]["observado"] - p[k]["centena"]["esperado_cobertura"] for p in r["por_periodo"]],
            mode="lines+markers", name=es.nome_estrategia(k)))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(template="plotly_dark", height=360, title="Excesso de centenas por período (0 = acaso)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)
    linhas = []
    for p in r["por_periodo"]:
        for k in chaves:
            pos = p[k].get("posicao")
            linhas.append({"Período": f"{p['de']}–{p['ate']}", "Estratégia": es.nome_estrategia(k),
                           "Centena": p[k]["centena"]["observado"],
                           "Esperada": round(p[k]["centena"]["esperado_cobertura"], 2),
                           "Posição média": None if not pos else round(pos["media"], 4)})
    st.dataframe(pd.DataFrame(linhas), hide_index=True)


def _aba_ativa(r: dict, arquivo, caminho) -> None:
    ativa = es.estrategia_ativa(caminho)
    st.metric("Estratégia de seleção em uso pelo motor", es.nome_estrategia(ativa["estrategia"]))
    aprovadas = [k for k, a in r["aprovacao"].items() if a["aprovada"]]
    if aprovadas:
        escolha = st.selectbox("Estratégia aprovada", aprovadas, format_func=es.nome_estrategia)
        if st.button("Ativar no motor"):
            es.ativar(escolha, r, arquivo.name, caminho)
            st.rerun()
    else:
        st.caption("Nenhuma estratégia aprovada: não há o que ativar.")
    if ativa["historico"] or ativa["estrategia"] != es.MOTOR_PADRAO:
        if st.button("Voltar à configuração anterior"):
            es.reverter(caminho)
            st.rerun()
        st.json(ativa["historico"], expanded=False)
