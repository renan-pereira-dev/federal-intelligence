"""Tela "Monitoramento do sorteio" (Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src import monitoramento as mo

CORES = {mo.NORMAL: ("#173A2A", "#8FE0B5", "Normal"),
         mo.ATENCAO: ("#3A2A12", "#F2C27B", "Atenção"),
         mo.ALERTA: ("#3A1616", "#F29B9B", "Alerta")}
EXPLICACAO = {
    mo.NORMAL: "Nenhum teste do bloco mais recente ficou significativo após a correção de Holm. O sorteio "
               "se comporta como esperado de um sorteio justo.",
    mo.ATENCAO: "Algum teste do bloco mais recente ficou significativo após a correção de Holm, mas sem "
                "repetição no bloco anterior. Num sorteio justo isso acontece em cerca de 5% dos blocos: "
                "é preciso esperar o próximo bloco para saber se persiste.",
    mo.ALERTA: "O mesmo teste ficou significativo em dois blocos seguidos e independentes. Isso é raro num "
               "sorteio justo e merece investigação (dados importados, mudança de regra ou defeito real).",
}


@st.cache_data(show_spinner="Calculando o monitoramento...", max_entries=4)
def _monitorar(caminho: str, versao: tuple) -> dict:
    return mo.monitorar(db.carregar_historico(caminho))


def _f(v, casas=4):
    return "—" if v is None else f"{v:.{casas}f}".replace(".", ",")


def _layout(fig, altura=360):
    fig.update_layout(template="plotly_dark", height=altura, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def pagina_monitoramento() -> None:
    st.title("Monitoramento do sorteio")
    st.caption("Verifica, bloco a bloco, se os números continuam se comportando como um sorteio justo: "
               "uniformes e independentes. Não prevê resultados.")
    caminho = db.resolver_caminho()
    m = _monitorar(str(caminho), db.versao_historico(caminho))
    if not m["blocos"]:
        st.warning(f"São necessários pelo menos {mo.BLOCO} concursos completos no histórico "
                   f"(há {m['concursos']}). Importe os resultados na tela Histórico.")
        return

    fundo, texto, rotulo = CORES[m["situacao"]]
    ultimo = m["blocos"][-1]
    st.markdown(
        f'<div style="background:{fundo};border-radius:14px;padding:16px 20px;margin:6px 0 14px 0">'
        f'<div style="color:{texto};font-size:1.6rem;font-weight:700">{rotulo}</div>'
        f'<div style="color:#E6E9EF">Bloco mais recente: concursos {ultimo["de"]}–{ultimo["ate"]} '
        f'({ultimo["data_de"]} a {ultimo["data_ate"]}), {mo.BLOCO} concursos.</div>'
        f'<div style="color:#C9D1DE;margin-top:6px">{EXPLICACAO[m["situacao"]]}</div></div>',
        unsafe_allow_html=True)
    if m["atencao"]:
        st.caption("Testes em atenção: " + ", ".join(mo.TESTES[t] for t in m["atencao"]))

    r = m["resumo"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Blocos analisados", len(m["blocos"]), help=f"{m['descartados']} concursos mais antigos ficaram "
                                                           "de fora por não completarem um bloco.")
    c2.metric("Testes com p < 0,05", f"{r['p_menor_005']} de {r['testes']}",
              help=f"Num sorteio justo, espera-se cerca de {_f(r['esperado_p_menor_005'], 1)} por acaso.")
    c3.metric("Alertas (desvios persistentes)", len(m["alertas"]))

    abas = st.tabs(["Bloco mais recente", "Ao longo do tempo", "Unidade por bloco", "Alertas", "Como funciona"])
    with abas[0]:
        linhas = []
        for t, nome in mo.TESTES.items():
            v = ultimo["testes"][t]
            linhas.append({"Teste": nome, "Estatística": round(v["estatistica"], 3),
                           "Graus de liberdade": str(v["gl"]) if v["gl"] is not None else "—",
                           "p": round(v["p"], 4), "p (Holm)": round(v["p_holm"], 4),
                           "Situação": "significativo" if v["p_holm"] < mo.ALFA else "dentro do esperado"})
        st.dataframe(pd.DataFrame(linhas), hide_index=True)
        u = ultimo["testes"]["unidade"]
        fig = go.Figure(go.Bar(x=[str(i) for i in range(10)], y=u["observados"], marker_color="#4FB3A9",
                               name="Observado"))
        fig.add_hline(y=u["esperado_por_categoria"], line_dash="dot", annotation_text="esperado")
        fig.update_layout(title="Último algarismo no bloco mais recente")
        st.plotly_chart(_layout(fig, 300))
        ind = ultimo["testes"]["independencia"]
        st.caption(f"Independência: a unidade de um prêmio repetiu a do mesmo prêmio no concurso anterior "
                   f"{ind['repeticoes']} vezes em {ind['pares']} pares (esperado {_f(ind['esperado'], 0)}).")
    with abas[1]:
        rotulos = [f"{b['de']}–{b['ate']}" for b in m["blocos"]]
        fig = go.Figure()
        for t, nome in mo.TESTES.items():
            fig.add_trace(go.Scatter(x=rotulos, y=[b["testes"][t]["p"] for b in m["blocos"]],
                                     mode="lines+markers", name=nome))
        fig.add_hline(y=mo.ALFA, line_dash="dot", annotation_text="p = 0,05")
        fig.update_yaxes(type="log", title="valor-p (escala log; mais baixo = mais desvio)")
        fig.update_layout(title="Valor-p de cada teste por bloco")
        st.plotly_chart(_layout(fig, 420))
        st.caption("Num sorteio justo, os pontos se espalham entre 0 e 1, e cerca de 1 em cada 20 fica abaixo "
                   "da linha por acaso. O que importa é um mesmo teste ficar abaixo em blocos seguidos.")
    with abas[2]:
        matriz = np.array([np.array(b["testes"]["unidade"]["observados"]) / b["testes"]["unidade"]["esperado_por_categoria"]
                           for b in m["blocos"]])
        fig = go.Figure(go.Heatmap(z=matriz.T, x=[f"{b['de']}–{b['ate']}" for b in m["blocos"]],
                                   y=[str(i) for i in range(10)], colorscale="RdBu", zmid=1,
                                   colorbar=dict(title="obs./esp.")))
        fig.update_layout(title="Último algarismo: observado ÷ esperado em cada bloco (1 = acaso)")
        fig.update_yaxes(title="algarismo")
        st.plotly_chart(_layout(fig, 420))
    with abas[3]:
        if m["alertas"]:
            st.dataframe(pd.DataFrame([{
                "Teste": mo.TESTES[a["teste"]], "Concursos": f"{a['de']}–{a['ate']}",
                "p nos dois blocos": f"{_f(a['p'][0])} e {_f(a['p'][1])}",
                "p (Holm) nos dois blocos": f"{_f(a['p_holm'][0])} e {_f(a['p_holm'][1])}",
            } for a in m["alertas"]]), hide_index=True)
        else:
            st.success("Nenhum desvio persistente em todo o histórico analisado.")
        isolados = [{"Bloco": f"{b['de']}–{b['ate']}", "Teste": mo.TESTES[t], "p (Holm)": round(v["p_holm"], 4)}
                    for b in m["blocos"] for t, v in b["testes"].items() if v["p_holm"] < mo.ALFA]
        if isolados:
            st.markdown("**Desvios isolados** (significativos num bloco só, sem repetição)")
            st.dataframe(pd.DataFrame(isolados), hide_index=True)
    with abas[4]:
        st.markdown(
            f"""
- **Blocos independentes** de {mo.BLOCO} concursos ({5 * mo.BLOCO} números), contados a partir do mais recente.
- **Seis testes por bloco:** unidade, dezena, algarismos da 3ª e 4ª posição (qui-quadrado contra a
  distribuição uniforme), unidade igual nos cinco prêmios (tabela de contingência) e independência entre
  concursos seguidos (binomial).
- **Correção de Holm** entre os seis testes de cada bloco.
- **Atenção:** algum teste significativo após Holm no bloco mais recente, sem repetição.
- **Alerta:** o mesmo teste com p < 0,05 em dois blocos seguidos, sendo pelo menos um significativo após Holm.
- **Fora dos testes:** as posições 1 e 2, porque dependem da quantidade de bilhetes de cada época (o
  universo mudou ao longo das décadas). Um desvio ali seria mudança de regra, não defeito do sorteio.
- **Concursos com prêmio zero** (julho de 2026) entram: os números foram sorteados normalmente.
- O monitoramento se atualiza sozinho quando novos resultados entram no banco.
- Um sorteio justo não garante que nada fique significativo: por acaso, cerca de 5% dos blocos ficam em
  atenção. A regra de alerta exige repetição justamente para não confundir acaso com defeito.
            """
        )
