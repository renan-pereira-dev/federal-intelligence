"""Federal Intelligence - Beta 0.1

Laboratório experimental de análise estatística da Loteria Federal.
Execute com:  streamlit run app.py   (ou dê dois cliques em iniciar.bat)
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src import statistics as est
from src.evaluator import conferir
from src.generator import QUANTIDADES_PERMITIDAS, gerar_candidatos, parametros_geracao
from src.importer import ErroImportacao, importar, modelo_csv, normalizar_numero
from src.importer import DESCRICAO_AJUSTES
from src.preditivo.interface import pagina_previsao
from src.preditivo.laboratorio import pagina_laboratorio
from src.oficial_interface import secao_resultados_oficiais
from src.monitoramento_interface import pagina_monitoramento
from src.probability import (
    CHANCE_DERIVADOS_TEXTO,
    DATA_CONSULTA,
    EXTRACOES,
    FONTE_MANUAL,
    FONTE_MANUAL_URL,
    FONTE_PAGINA,
    FONTE_REGULAMENTO,
    FONTE_REGULAMENTO_URL,
    JUSTIFICATIVA_INDEPENDENCIA,
    JUSTIFICATIVA_PRINCIPAL,
    REGRA_UNIVERSO,
    TERMINACOES_NOMES,
    explicacao_probabilidade,
    extracoes_habilitadas,
    formatar_milhar,
    justificativa_terminacao,
    probabilidade_algum_premio,
    probabilidade_terminacao,
)

st.set_page_config(
    page_title="Federal Intelligence",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

AVISO_FREQUENCIA = (
    "A pontuação ou frequência histórica não representa maior chance de ganhar. "
    "Em um sorteio justo, todos os números elegíveis possuem a mesma probabilidade."
)
AVISO_CANDIDATO = (
    "Os números abaixo são candidatos experimentais gerados ao acaso. Não são bilhetes "
    "reservados, disponíveis ou garantidos para compra, e não aumentam a chance de ganhar."
)
CORES = {"observado": "#4FB3A9", "esperado": "#8A93A6"}

st.markdown(
    """
    <style>
    .fi-subtitulo {color:#9AA4B5; margin-top:-0.8rem; margin-bottom:1.2rem;}
    .fi-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px;}
    .fi-card {background:#161B26; border:1px solid #273042; border-radius:14px; padding:16px 18px;}
    .fi-numero {font-family:'Consolas','Courier New',monospace; font-size:2.1rem; font-weight:700;
                letter-spacing:0.18em; color:#E6E9EF;}
    .fi-rotulo {font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#8A93A6;}
    .fi-valor {font-size:0.92rem; color:#C9D1DE; margin-bottom:6px;}
    .fi-tag {display:inline-block; font-size:0.7rem; padding:2px 8px; border-radius:999px;
             background:#1F3B39; color:#7FD3C9; margin-bottom:8px;}
    .fi-aviso {border-left:4px solid #E0A458; background:#1E1A13; padding:12px 16px;
               border-radius:6px; color:#EBD9BC; margin:12px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

db.inicializar()


# ------------------------------------------------------------------ utilidades

def aviso(texto: str) -> None:
    st.markdown(f'<div class="fi-aviso">{texto}</div>', unsafe_allow_html=True)


def layout_grafico(fig: go.Figure, altura: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def nome_extracao(chave: str) -> str:
    return EXTRACOES[chave].nome if chave in EXTRACOES else "Não informada"


# ------------------------------------------------------------------ cache
#
# Os cálculos pesados ficam em cache na memória do Streamlit. A chave inclui o
# caminho do banco e a "versão" do histórico (contagem e maior id de concursos
# e importações): importar ou apagar dados muda a versão e invalida o cache
# sozinho. Os filtros também fazem parte da chave.

def _chave_banco() -> tuple[str, tuple]:
    caminho = db.resolver_caminho()
    return str(caminho), db.versao_historico(caminho)


@st.cache_data(show_spinner=False, max_entries=4)
def _historico_em_cache(caminho: str, versao: tuple) -> pd.DataFrame:
    return db.carregar_historico(caminho)


def carregar_historico() -> pd.DataFrame:
    return _historico_em_cache(*_chave_banco())


@st.cache_data(show_spinner=False, max_entries=32)
def _dados_filtrados(caminho: str, versao: tuple, incluir: bool, inicio: int, fim: int,
                     extracoes: tuple | None) -> pd.DataFrame:
    dados = est.filtrar_historico(_historico_em_cache(caminho, versao), incluir, inicio, fim)
    if extracoes is not None:
        dados = dados[dados["extracao"].isin(extracoes)]
    return dados


@st.cache_data(show_spinner="Calculando estatísticas...", max_entries=32)
def analise_historico(caminho: str, versao: tuple, incluir: bool, inicio: int, fim: int,
                      extracoes: tuple | None) -> dict:
    """Todos os cálculos da tela Histórico para um conjunto de filtros."""
    d = _dados_filtrados(caminho, versao, incluir, inicio, fim, extracoes)
    return {
        "algarismos": est.frequencia_algarismos(d),
        "contagens_centena": est.distribuicao_contagens(d, 3),
        "contagens_dezena": est.distribuicao_contagens(d, 2),
        "posicao": est.frequencia_por_posicao(d),
        "esperado_posicao": est.esperado_por_posicao(d),
        "esperado_disponivel": est.esperado_disponivel(d),
        "terminacoes": {k: est.terminacoes(d, k) for k in (1, 2, 3, 4)},
        "repeticoes": est.repeticoes(d),
        "faixas": est.distribuicao_faixas(d),
        "soma": est.distribuicao_soma_algarismos(d),
        "paridade": est.distribuicao_paridade(d),
        "por_ano": est.concursos_por_ano(d),
        "decadas": est.resumo_decadas(d),
        "primeiro_decada": est.primeiro_algarismo_por_decada(d),
        "eras": est.maior_numero_por_ano(d),
    }


@st.cache_data(show_spinner="Calculando taxas históricas...", max_entries=32)
def analise_taxa(caminho: str, versao: tuple, incluir: bool, inicio: int, fim: int) -> dict:
    """Todos os cálculos da tela Taxa histórica para um conjunto de filtros."""
    d = _dados_filtrados(caminho, versao, incluir, inicio, fim, None)
    return {
        "concursos": int(d["concurso"].nunique()),
        "numeros": int(len(d)),
        "repeticoes": est.repeticoes(d),
        "coincidencia": {k: est.taxa_coincidencia(d, k) for k in (1, 2, 3, 4)},
        "unidade_primeiro": est.teste_terminacoes(d, 1, apenas_primeiro=True),
        "uniformidade": {k: est.teste_terminacoes(d, k) for k in (1, 2, 3, 4)},
        "qui_quadrado": est.resumo_testes_qui_quadrado(d),
    }


def pct(valor: float, casas: int = 2) -> str:
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def filtros_historico(historico: pd.DataFrame, chave: str) -> tuple[bool, int, int]:
    """Filtros de período e de concursos com prêmio zero. Devolve (incluir, início, fim)."""
    anos = pd.to_datetime(historico.drop_duplicates("concurso")["data"]).dt.year
    ano_min, ano_max = int(anos.min()), int(anos.max())
    zerados = historico.loc[historico["premio_zero"], "concurso"].unique()
    col1, col2 = st.columns([3, 2])
    with col1:
        if ano_min < ano_max:
            inicio, fim = st.slider("Período (anos)", ano_min, ano_max, (ano_min, ano_max), key=f"anos_{chave}")
        else:
            inicio, fim = ano_min, ano_max
    with col2:
        incluir = st.toggle(
            f"Incluir concursos com prêmio zero ({len(zerados)})",
            value=True,
            key=f"zero_{chave}",
            help="Concursos marcados porque todos os valores de prêmio informados são R$ 0,00, "
                 "como os de julho de 2026, sem venda de bilhetes. Os números foram sorteados "
                 "normalmente; desligue para tirá-los das análises.",
            disabled=len(zerados) == 0,
        )
    return bool(incluir), int(inicio), int(fim)


# ------------------------------------------------------------------ páginas

def pagina_inicio() -> None:
    st.title("Federal Intelligence")
    st.markdown(
        '<p class="fi-subtitulo">Laboratório experimental de análise estatística da Loteria Federal.</p>',
        unsafe_allow_html=True,
    )
    aviso(
        "Ferramenta educacional e demonstrativa. Não incentiva apostas e não promete ganhos. "
        + AVISO_FREQUENCIA
    )

    historico = carregar_historico()
    lotes = db.listar_lotes()
    c1, c2, c3 = st.columns(3)
    c1.metric("Concursos no histórico", historico["concurso"].nunique() if not historico.empty else 0)
    c2.metric("Lotes de candidatos gerados", len(lotes))
    c3.metric("Extrações habilitadas", len(extracoes_habilitadas()))

    st.subheader("O que você pode fazer aqui")
    st.markdown(
        """
- **Gerador**: sorteia candidatos ao acaso, de forma uniforme, dentro da faixa oficial de cada extração.
- **Histórico**: importa resultados (CSV) e mostra frequências e distribuições, sempre comparadas
  com o que o acaso produziria.
- **Conferência**: digite os cinco números premiados de um concurso e compare com seus candidatos.
- **Probabilidades**: as chances oficiais da CAIXA e o porquê de cada uma.
- **Taxa histórica**: quanto cada tipo de coincidência aconteceu desde 1962, comparado com a teoria.
- **Monitoramento do sorteio**: confere, bloco a bloco, se o sorteio continua uniforme e independente.
- **Previsão do próximo sorteio**: motor experimental (estatística e aprendizado de máquina) que ordena
  candidatos e mostra, separadamente, a probabilidade oficial e o desempenho real do modelo no backtest.
- **Fontes e regras**: de onde vieram as regras usadas pelo aplicativo.
        """
    )


def pagina_gerador() -> None:
    st.title("Gerador de candidatos")
    st.caption("Sorteio uniforme com a fonte segura de aleatoriedade do Python (módulo secrets).")

    opcoes = [e.chave for e in extracoes_habilitadas()]
    if not opcoes:
        st.error("Nenhuma extração está habilitada: as regras oficiais não foram confirmadas.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        chave = st.selectbox(
            "Extração",
            opcoes,
            format_func=lambda k: f"{EXTRACOES[k].nome} — {EXTRACOES[k].descricao}",
        )
    with col2:
        quantidade = st.segmented_control(
            "Quantidade de números", QUANTIDADES_PERMITIDAS, default=5, required=True
        )

    extracao = EXTRACOES[chave]
    st.info(
        f"Faixa válida: **{extracao.faixa_texto}** · Chance oficial do prêmio principal: "
        f"**{extracao.chance_texto}** por bilhete."
    )

    if st.button("Gerar candidatos", type="primary"):
        numeros = gerar_candidatos(chave, int(quantidade))
        parametros = parametros_geracao(chave, int(quantidade))
        lote = db.salvar_candidatos(chave, numeros, parametros)
        st.session_state["ultimo_lote"] = {"lote": lote, "extracao": chave, "numeros": numeros}

    ultimo = st.session_state.get("ultimo_lote")
    if not ultimo:
        return

    extracao = EXTRACOES[ultimo["extracao"]]
    st.subheader(f"Lote {ultimo['lote']}")
    aviso(AVISO_CANDIDATO)
    aviso(AVISO_FREQUENCIA)

    historico = carregar_historico()
    cards = []
    linhas_csv = []
    for numero in ultimo["numeros"]:
        freq = est.frequencia_candidato(numero, historico)
        if freq is None:
            texto_freq = "Sem histórico importado"
        else:
            texto_freq = (
                f"Exato: {freq['exato']} · Milhar: {freq['milhar']} · "
                f"Centena: {freq['centena']} · Dezena: {freq['dezena']}"
                f"<br><span class='fi-rotulo'>em {formatar_milhar(freq['total_numeros'])} números do histórico</span>"
            )
        cards.append(
            f"""<div class="fi-card">
                <span class="fi-tag">Candidato experimental</span>
                <div class="fi-numero">{numero}</div>
                <div class="fi-rotulo">Extração</div><div class="fi-valor">{extracao.nome}</div>
                <div class="fi-rotulo">Probabilidade oficial</div><div class="fi-valor">{extracao.chance_texto}</div>
                <div class="fi-rotulo">Por quê</div><div class="fi-valor">1 bilhete entre
                {formatar_milhar(extracao.bilhetes_por_serie)} com a mesma chance</div>
                <div class="fi-rotulo">Frequência histórica</div><div class="fi-valor">{texto_freq}</div>
            </div>"""
        )
        linhas_csv.append({
            "lote": ultimo["lote"], "numero": numero, "extracao": extracao.nome,
            "probabilidade_oficial": extracao.chance_texto,
            "frequencia_exata": None if freq is None else freq["exato"],
        })
    st.markdown(f'<div class="fi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    with st.expander("Explicação da probabilidade", expanded=True):
        st.write(explicacao_probabilidade(extracao.chave))
        st.write(JUSTIFICATIVA_PRINCIPAL)
        st.write(
            "A frequência histórica só conta quantas vezes um número, ou o seu final, já saiu. "
            "Ela descreve o passado e não muda a chance do próximo sorteio. Os candidatos foram "
            "gerados sem consultar o histórico."
        )

    csv = pd.DataFrame(linhas_csv).to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button("Baixar candidatos (CSV)", csv, f"candidatos_{ultimo['lote']}.csv", "text/csv")


def secao_importacao() -> None:
    st.markdown(
        "O arquivo deve ter o cabeçalho abaixo (separador `;` ou `,`). A coluna `extracao` "
        "pode ficar vazia; quando preenchida, use `quartou`, `domingo`, `enricou` ou `natal`."
    )
    st.code(modelo_csv().strip(), language="text")
    st.download_button(
        "Baixar modelo vazio (CSV)", modelo_csv().encode("utf-8-sig"), "modelo_historico.csv", "text/csv"
    )
    st.caption(
        "Também é aceito o formato com bilhete_1 a bilhete_5 (com apóstrofo e 6 dígitos começando "
        "com 0). Colunas extras, como valores dos prêmios e fonte, são guardadas."
    )
    arquivo = st.file_uploader("Arquivo CSV com resultados desde 1962", type=["csv"])
    if arquivo is None:
        return
    try:
        resultado = importar(arquivo.getvalue(), db.concursos_existentes())
    except ErroImportacao as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas lidas", resultado.linhas_total)
    c2.metric("Válidas", resultado.linhas_validas)
    c3.metric("Rejeitadas", resultado.linhas_rejeitadas)
    c4.metric("Duplicadas", resultado.duplicados)
    for ajuste, quantidade in resultado.ajustes.items():
        st.info(f"{formatar_milhar(quantidade)} número(s): {DESCRICAO_AJUSTES[ajuste]}.")
    if resultado.lacunas:
        st.warning(
            f"{len(resultado.lacunas)} concurso(s) ausente(s) na sequência: "
            + ", ".join(map(str, resultado.lacunas[:60]))
            + (" ..." if len(resultado.lacunas) > 60 else "")
            + ". Eles não são preenchidos."
        )
    if resultado.concursos_premio_zero:
        st.info(
            f"{len(resultado.concursos_premio_zero)} concurso(s) com todos os prêmios informados como "
            "R$ 0,00 serão marcados: " + ", ".join(map(str, resultado.concursos_premio_zero)) + "."
        )
    if resultado.metadados_arquivo:
        with st.expander("Informações iguais em todas as linhas (guardadas uma vez)"):
            st.json(resultado.metadados_arquivo)
    outros = [a for a in resultado.avisos if not any(d in a for d in DESCRICAO_AJUSTES.values())]
    for texto in outros[:20]:
        st.warning(texto)
    if len(outros) > 20:
        st.warning(f"... e mais {len(outros) - 20} aviso(s).")
    if not resultado.rejeitados.empty:
        st.markdown("**Linhas rejeitadas** (não serão gravadas):")
        st.dataframe(resultado.rejeitados, hide_index=True, height=240)
    if resultado.linhas_validas == 0:
        st.info("Nenhuma linha válida para gravar.")
        return
    st.dataframe(resultado.validos.drop(columns=["dados_originais"]).head(50), hide_index=True, height=240)
    if st.button(f"Gravar {resultado.linhas_validas} concurso(s) no banco", type="primary"):
        db.registrar_importacao(
            resultado.validos, arquivo.name, resultado.linhas_total,
            resultado.linhas_rejeitadas, resultado.duplicados, resultado.relatorio(),
        )
        st.cache_data.clear()
        st.success("Importação gravada.")
        st.rerun()


def pagina_historico() -> None:
    st.title("Histórico")
    historico = carregar_historico()

    if historico.empty:
        st.warning(
            "Nenhum histórico carregado. Os dados precisam ser importados antes da análise. "
            "O aplicativo não inclui resultados prontos e não inventa dados: use um arquivo "
            "CSV com resultados oficiais."
        )
        secao_importacao()
        with st.expander("Buscar o último resultado oficial (CAIXA)"):
            secao_resultados_oficiais("historico_vazio")
        return

    with st.expander("Buscar resultados oficiais (CAIXA)"):
        secao_resultados_oficiais("historico")
    with st.expander("Importar mais resultados"):
        secao_importacao()

    caminho, versao = _chave_banco()
    incluir, inicio, fim = filtros_historico(historico, "historico")
    extracoes_presentes = sorted(historico["extracao"].fillna("").unique())
    extracoes = None
    if len(extracoes_presentes) > 1:
        selecionadas = st.multiselect(
            "Filtrar por extração", extracoes_presentes, default=extracoes_presentes, format_func=nome_extracao
        )
        extracoes = tuple(selecionadas)
    dados = _dados_filtrados(caminho, versao, incluir, inicio, fim, extracoes)
    if dados.empty:
        st.info("Nenhum concurso com os filtros escolhidos.")
        return
    a = analise_historico(caminho, versao, incluir, inicio, fim, extracoes)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concursos", formatar_milhar(dados["concurso"].nunique()))
    c2.metric("Números premiados", formatar_milhar(len(dados)))
    c3.metric("Primeira data", pd.to_datetime(dados["data"]).min().strftime("%d/%m/%Y"))
    c4.metric("Última data", pd.to_datetime(dados["data"]).max().strftime("%d/%m/%Y"))

    aviso(AVISO_FREQUENCIA)

    abas = st.tabs(["Frequências", "Por posição", "Terminações", "Décadas", "Épocas",
                    "Repetições", "Distribuições", "Dados"])

    with abas[0]:
        freq = a["algarismos"]
        fig = go.Figure()
        fig.add_bar(x=freq["algarismo"], y=freq["observado"], name="Observado", marker_color=CORES["observado"])
        if "esperado" in freq:
            fig.add_scatter(x=freq["algarismo"], y=freq["esperado"], name="Esperado ao acaso",
                            mode="markers+lines", marker_color=CORES["esperado"])
        fig.update_layout(title="Frequência dos algarismos (todas as posições)",
                          xaxis_title="Algarismo", yaxis_title="Ocorrências")
        st.plotly_chart(layout_grafico(fig))
        st.caption(
            "O 0 e os algarismos baixos aparecem mais porque, na 1ª posição, a faixa de números "
            "nunca chega a 9 por completo (ex.: 00001 a 92000). Isso é efeito do plano, não do acaso."
        )

        st.markdown("#### Distribuição de frequências das terminações")
        digitos_dist = st.segmented_control(
            "Terminação", [2, 3], default=3, required=True, key="dist_digitos",
            format_func=lambda d: "Dezena (00 a 99)" if d == 2 else "Centena (000 a 999)",
        )
        dist = a["contagens_dezena"] if digitos_dist == 2 else a["contagens_centena"]
        dist = dist[(dist["terminacoes_observadas"] > 0) | (dist["terminacoes_esperadas"] >= 0.05)]
        fig = go.Figure()
        fig.add_bar(x=dist["vezes"], y=dist["terminacoes_observadas"], name="Observado",
                    marker_color=CORES["observado"])
        fig.add_scatter(x=dist["vezes"], y=dist["terminacoes_esperadas"], name="Esperado (binomial)",
                        mode="lines", line=dict(color=CORES["esperado"], width=3))
        fig.update_layout(
            title="Quantas terminações saíram k vezes",
            xaxis_title="Número de vezes que a terminação saiu (k)",
            yaxis_title="Quantidade de terminações",
        )
        st.plotly_chart(layout_grafico(fig))
        st.caption(
            "Por quê: se cada terminação tem chance 1/10^d em cada número sorteado, a quantidade de "
            "vezes que uma terminação sai segue uma distribuição binomial (n números, chance 1/10^d). "
            "A linha mostra quantas terminações deveriam sair k vezes por puro acaso. Barras "
            "acompanhando a linha indicam que nenhuma terminação está \"quente\" ou \"fria\" além do normal."
        )

    with abas[1]:
        fig = px.imshow(a["posicao"], text_auto=True, color_continuous_scale="Teal", aspect="auto",
                        labels=dict(x="Algarismo", y="Posição", color="Ocorrências"))
        fig.update_layout(title="Frequência por posição")
        st.plotly_chart(layout_grafico(fig, 380))
        if a["esperado_posicao"] is not None:
            st.markdown("**Esperado ao acaso** (o primeiro algarismo não é uniforme, "
                        "porque a faixa termina antes de 99999):")
            st.dataframe(a["esperado_posicao"].round(1), height=230)
        else:
            st.caption(
                "Sem o tipo de extração de cada concurso não há um esperado exato por posição: ele "
                "depende da quantidade de bilhetes de cada época. Veja a aba Épocas."
            )

    with abas[2]:
        col1, col2 = st.columns(2)
        with col1:
            t1 = a["terminacoes"][1].sort_values("terminacao")
            fig = px.bar(t1, x="terminacao", y="observado", title="Unidade final",
                         color_discrete_sequence=[CORES["observado"]])
            fig.add_hline(y=t1["esperado_aprox"].iloc[0], line_dash="dot", line_color=CORES["esperado"],
                          annotation_text="esperado")
            st.plotly_chart(layout_grafico(fig))
        with col2:
            t2 = a["terminacoes"][2].sort_values("terminacao")
            fig = px.bar(t2, x="terminacao", y="observado", title="Dezena final (00 a 99)",
                         color_discrete_sequence=[CORES["observado"]])
            fig.add_hline(y=t2["esperado_aprox"].iloc[0], line_dash="dot", line_color=CORES["esperado"])
            fig.update_xaxes(type="category", showticklabels=False)
            st.plotly_chart(layout_grafico(fig))
        digitos = st.segmented_control("Terminações mais frequentes com", [2, 3, 4], default=3,
                                       format_func=lambda d: f"{d} dígitos", required=True)
        st.dataframe(a["terminacoes"][int(digitos)].head(20).round(2), hide_index=True)

    with abas[3]:
        dec = a["decadas"]
        st.markdown(
            "Cada década é analisada separadamente. Assim dá para ver se o comportamento das "
            "terminações mudou ao longo do tempo, mesmo com planos de bilhetes diferentes."
        )
        fig = px.bar(dec, x="decada", y="concursos", title="Concursos por década",
                     color_discrete_sequence=[CORES["observado"]], text="concursos")
        st.plotly_chart(layout_grafico(fig, 320))
        if dec["inicio"].max() == 2020:
            st.caption("A década de 2020 ainda está em andamento.")

        fig = go.Figure()
        fig.add_scatter(
            x=dec["decada"], y=dec["taxa_unidade"] * 100, mode="markers", name="Taxa histórica",
            marker=dict(color=CORES["observado"], size=11),
            error_y=dict(type="data", symmetric=False,
                         array=(dec["taxa_unidade_max"] - dec["taxa_unidade"]) * 100,
                         arrayminus=(dec["taxa_unidade"] - dec["taxa_unidade_min"]) * 100),
        )
        fig.add_hline(y=dec["taxa_unidade_teorica"].iloc[0] * 100, line_dash="dot",
                      line_color=CORES["esperado"], annotation_text="teórica 40,95%")
        fig.update_layout(title="Taxa histórica de coincidência da unidade, por década",
                          yaxis_title="% de finais que coincidiram com algum prêmio")
        st.plotly_chart(layout_grafico(fig))
        st.caption(
            "Por quê: teórica = 1 − 0,9^5 ≈ 40,95% (chance de um final coincidir com pelo menos "
            "um dos 5 prêmios independentes). As barras são o intervalo de 95% de cada década: "
            "se a linha passa por dentro delas, a década é compatível com o acaso."
        )

        p = dec.melt(id_vars="decada", value_vars=["p_1_digito", "p_2_digitos", "p_3_digitos"],
                     var_name="teste", value_name="p_valor").dropna()
        p["teste"] = p["teste"].map({"p_1_digito": "Unidade", "p_2_digitos": "Dezena", "p_3_digitos": "Centena"})
        fig = px.bar(p, x="decada", y="p_valor", color="teste", barmode="group",
                     title="Teste qui-quadrado por década (p-valor)",
                     color_discrete_sequence=[CORES["observado"], "#7AA6E0", "#E0A458"])
        fig.add_hline(y=0.05, line_dash="dot", line_color="#E07A7A", annotation_text="0,05")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(layout_grafico(fig))
        n_testes = len(p)
        st.caption(
            f"Por quê: p-valor abaixo de 0,05 indica diferença maior que a habitual. Com {n_testes} "
            f"testes, espera-se por acaso cerca de {0.05 * n_testes:.1f} abaixo da linha, mesmo num "
            "sorteio perfeitamente justo. A centena só é testada quando a década tem números "
            "suficientes (pelo menos 5 esperados por terminação)."
        )
        tabela = dec[["decada", "concursos", "numeros", "menor_numero", "maior_numero"]].copy()
        tabela["taxa unidade"] = dec["taxa_unidade"].map(lambda v: pct(v))
        for col, nome in (("p_1_digito", "p unidade"), ("p_2_digitos", "p dezena"), ("p_3_digitos", "p centena")):
            tabela[nome] = dec[col].round(3)
        st.dataframe(tabela, hide_index=True)

    with abas[4]:
        st.markdown(
            "A quantidade de bilhetes mudou ao longo das décadas. O aplicativo só tem a regra "
            "oficial dos planos atuais (Manual v23); as épocas antigas são mostradas pelo que os "
            "dados revelam."
        )
        eras = a["eras"]
        fig = px.line(eras, x="ano", y="maior_numero", markers=True,
                      title="Maior número sorteado por ano", color_discrete_sequence=[CORES["observado"]])
        for e in EXTRACOES.values():
            fig.add_hline(y=e.numero_maximo, line_dash="dot", line_color=CORES["esperado"],
                          annotation_text=f"{e.nome} hoje", annotation_position="bottom right")
        st.plotly_chart(layout_grafico(fig, 400))
        st.caption(
            "Linhas pontilhadas: limite dos planos atuais. Anos com números acima delas usaram "
            "outros planos."
        )
        fig = px.imshow(a["primeiro_decada"].round(1), text_auto=True, aspect="auto",
                        color_continuous_scale="Teal",
                        labels=dict(x="1º algarismo (dezena de milhar)", y="Década", color="%"))
        fig.update_layout(title="Distribuição do 1º algarismo por década (%)")
        st.plotly_chart(layout_grafico(fig, 420))
        st.caption(
            "Por quê: num sorteio uniforme de 00001 a N, o 1º algarismo só pode ir até o 1º "
            "algarismo de N. Uma década sem nenhum 7, 8 ou 9 na frente indica um plano com menos "
            "bilhetes; isso é uma observação dos dados, não uma regra oficial da época."
        )
        st.dataframe(eras, hide_index=True)

    with abas[5]:
        rep = a["repeticoes"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Números sorteados", formatar_milhar(rep["total_numeros"]))
        c2.metric("Números distintos", formatar_milhar(rep["numeros_distintos"]))
        c3.metric("Números que saíram mais de uma vez", formatar_milhar(len(rep["numeros_repetidos"])))
        if rep["pares_esperados_aprox"] is not None:
            st.caption(
                f"Ao acaso, seriam esperados cerca de {rep['pares_esperados_aprox']:.1f} pares de "
                "números iguais neste volume de sorteios (o mesmo efeito do 'paradoxo do aniversário')."
            )
        st.markdown("**Números repetidos no histórico**")
        if rep["numeros_repetidos"].empty:
            st.write("Nenhum número exato se repetiu.")
        else:
            st.dataframe(rep["numeros_repetidos"], hide_index=True)
        st.markdown("**Concursos com o mesmo número em mais de um prêmio** (permitido pelo Regulamento, item 11)")
        if rep["concursos_com_repeticao_interna"].empty:
            st.write("Nenhum.")
        else:
            st.dataframe(rep["concursos_com_repeticao_interna"], hide_index=True)

    with abas[6]:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(a["faixas"], x="faixa", y="observado",
                         title="Números por faixa de 10.000", color_discrete_sequence=[CORES["observado"]])
            st.plotly_chart(layout_grafico(fig))
            fig = px.bar(a["paridade"], x="tipo", y="observado",
                         title="Final par ou ímpar", color_discrete_sequence=[CORES["observado"]])
            st.plotly_chart(layout_grafico(fig, 300))
        with col2:
            fig = px.bar(a["soma"], x="soma", y="observado",
                         title="Soma dos 5 algarismos", color_discrete_sequence=[CORES["observado"]])
            st.plotly_chart(layout_grafico(fig))
            fig = px.bar(a["por_ano"], x="ano", y="concursos",
                         title="Concursos por ano no histórico", color_discrete_sequence=[CORES["observado"]])
            st.plotly_chart(layout_grafico(fig, 300))
        st.caption("As faixas mais altas tiveram menos números possíveis em várias épocas, por isso aparecem menos.")

    with abas[7]:
        st.dataframe(dados, hide_index=True)
        st.markdown("**Importações**")
        st.dataframe(db.listar_importacoes(), hide_index=True)
        with st.popover("Apagar histórico"):
            st.write("Remove todos os concursos e importações. Os candidatos gerados são mantidos.")
            if st.checkbox("Confirmo que quero apagar o histórico"):
                if st.button("Apagar agora", type="primary"):
                    db.apagar_historico()
                    st.cache_data.clear()
                    st.rerun()


def pagina_conferencia() -> None:
    st.title("Conferência de resultados")
    aviso(
        "Só o acerto exato de um dos cinco números corresponde ao prêmio principal. "
        "Milhar, centena, dezena, aproximação e outros finais são mostrados como coincidências "
        "compatíveis com as faixas derivadas do Manual da CAIXA (item 5.4.4). O valor e a "
        "acumulação dessas faixas dependem do plano impresso no verso de cada bilhete. "
        "Candidatos gerados não são bilhetes comprados."
    )

    lotes = db.listar_lotes()
    if lotes.empty:
        st.info("Gere candidatos na página Gerador antes de conferir.")
        return

    st.subheader("1. Resultado do concurso")
    col1, col2, col3 = st.columns(3)
    concurso = col1.number_input("Número do concurso", min_value=1, step=1, value=None, placeholder="Ex.: 5800")
    data_sorteio = col2.date_input("Data do sorteio", value=None, format="DD/MM/YYYY",
                                   min_value=date(1962, 1, 1), max_value=date.today())
    chave = col3.selectbox("Extração", list(EXTRACOES), format_func=nome_extracao)

    existente = db.buscar_concurso(int(concurso)) if concurso else None
    if existente:
        st.caption(
            f"O concurso {existente['concurso']} já está no banco ({existente['data']}): "
            f"{', '.join(existente['premios'])}. Os campos abaixo foram preenchidos com esses números."
        )

    colunas = st.columns(5)
    premios_txt = []
    for i, coluna in enumerate(colunas):
        padrao = existente["premios"][i] if existente else ""
        premios_txt.append(coluna.text_input(f"{i + 1}º prêmio", value=padrao, max_chars=5,
                                             key=f"premio_{i}_{concurso}"))

    st.subheader("2. Candidatos")
    lote = st.selectbox(
        "Lote gerado",
        lotes["lote"],
        format_func=lambda l: (
            f"{l} · {nome_extracao(lotes.set_index('lote').loc[l, 'extracao'])} · "
            f"{lotes.set_index('lote').loc[l, 'quantidade']} números"
        ),
    )
    candidatos = db.candidatos_do_lote(lote)
    st.write("Candidatos: " + ", ".join(candidatos["numero"]))
    if (candidatos["extracao"] != chave).any():
        st.warning("Este lote foi gerado para outra extração. A conferência é feita mesmo assim, "
                   "com as regras da extração escolhida acima.")

    if not st.button("Conferir e salvar", type="primary"):
        return

    erros = []
    if not concurso:
        erros.append("Informe o número do concurso.")
    if data_sorteio is None:
        erros.append("Informe a data do sorteio.")
    premios = []
    for i, texto in enumerate(premios_txt, 1):
        numero, erro, _ = normalizar_numero(texto)
        if erro or len(texto.strip()) != 5:
            erros.append(f"{i}º prêmio: digite exatamente 5 dígitos (ex.: 01339).")
        else:
            premios.append(numero)
    if erros:
        for erro in erros:
            st.error(erro)
        return

    try:
        contest_id = db.salvar_concurso_manual(int(concurso), data_sorteio.isoformat(), chave, premios)
    except ValueError as exc:
        st.error(str(exc))
        return

    linhas = []
    for _, candidato in candidatos.iterrows():
        resultado = conferir(candidato["numero"], premios, chave)
        db.salvar_conferencia(int(candidato["id"]), contest_id, resultado.para_dict())
        linhas.append({
            "Candidato": candidato["numero"],
            "Resultado": resultado.resumo(),
            "Detalhes": "; ".join(resultado.itens()) or "—",
            "Não avaliado": "; ".join(resultado.nao_avaliado) or "—",
            "_exato": resultado.acertou_principal,
            "_parcial": resultado.tem_coincidencia,
            "_obs": resultado.observacoes,
        })
    tabela = pd.DataFrame(linhas)

    st.success(f"Conferência salva para o concurso {int(concurso)}.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acertos exatos", int(tabela["_exato"].sum()))
    c2.metric("Com coincidências parciais", int((~tabela["_exato"] & tabela["_parcial"]).sum()))
    c3.metric("Sem coincidência", int((~tabela["_exato"] & ~tabela["_parcial"]).sum()))
    st.dataframe(tabela[["Candidato", "Resultado", "Detalhes", "Não avaliado"]], hide_index=True)
    observacoes = sorted({o for lista in tabela["_obs"] for o in lista})
    for texto in observacoes:
        st.caption(texto)


def pagina_probabilidades() -> None:
    st.title("Probabilidades oficiais")
    st.write(
        "A chance de acertar o prêmio principal depende só de quantos bilhetes a CAIXA emite em "
        "cada extração. Todo número dentro da faixa tem a mesma chance teórica."
    )
    tabela = pd.DataFrame([
        {
            "Extração": e.nome,
            "Bilhetes por série": formatar_milhar(e.bilhetes_por_serie),
            "Séries": e.series,
            "Faixa válida": e.faixa_texto,
            "Chance do prêmio principal": e.chance_texto,
            "Geração": "Habilitada" if e.habilitada else "Desabilitada",
        }
        for e in EXTRACOES.values()
    ])
    st.dataframe(tabela, hide_index=True)

    cards = "".join(
        f"""<div class="fi-card"><span class="fi-tag">{e.nome}</span>
        <div class="fi-numero" style="font-size:1.5rem;letter-spacing:0.04em">{e.chance_texto}</div>
        <div class="fi-valor">{e.descricao}</div></div>"""
        for e in EXTRACOES.values()
    )
    st.markdown(f'<div class="fi-grid">{cards}</div>', unsafe_allow_html=True)

    st.subheader("Como ler esses números")
    for e in EXTRACOES.values():
        st.markdown(f"- {explicacao_probabilidade(e.chave)}")
    st.info(JUSTIFICATIVA_PRINCIPAL)

    st.subheader("Coincidências de terminação: chance teórica e o porquê")
    st.write(
        "Estas são chances de **coincidência** do final do seu número com o final de um prêmio. "
        "Elas não são chances de prêmio: valor e regras de pagamento dependem do plano do bilhete."
    )
    st.caption(JUSTIFICATIVA_INDEPENDENCIA)
    linhas = []
    for e in EXTRACOES.values():
        for digitos in (1, 2, 3, 4):
            p = probabilidade_terminacao(e.numero_maximo, digitos)
            if p is None:
                um, cinco = "varia conforme a terminação", "varia conforme a terminação"
            else:
                um = f"1 em {formatar_milhar(p.denominator)}"
                cinco = pct(float(probabilidade_algum_premio(p)), 3)
            linhas.append({
                "Extração": e.nome,
                "Terminação": f"{TERMINACOES_NOMES[digitos]} ({digitos} dígito{'s' if digitos > 1 else ''})",
                "Chance para 1 prêmio": um,
                "Chance de coincidir com algum dos 5": cinco,
            })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    with st.expander("Por que cada chance de terminação tem esse valor"):
        for e in EXTRACOES.values():
            st.markdown(f"**{e.nome}**")
            for digitos in (1, 2, 3, 4):
                st.markdown(f"- {justificativa_terminacao(e.numero_maximo, digitos)}")

    st.subheader("Prêmios derivados")
    st.write(CHANCE_DERIVADOS_TEXTO)
    st.write(
        "O aplicativo não calcula chances de milhar, centena, dezena ou outras faixas derivadas, "
        "porque a quantidade e o valor delas dependem do plano de sorteio de cada bilhete."
    )
    aviso(AVISO_FREQUENCIA)


def interpretar_p(p: float) -> str:
    if p != p:  # NaN
        return "sem dados suficientes"
    if p >= 0.05:
        return "compatível com um sorteio uniforme"
    return ("diferença maior que a habitual; em muitos testes, cerca de 1 em 20 mostra isso "
            "por puro acaso, e isso não indica tendência futura")


def pagina_taxa_historica() -> None:
    st.title("Taxa de acerto histórica")
    st.write(
        "Aqui o histórico é usado para **conferir a teoria**: quanto cada tipo de coincidência "
        "aconteceu no passado, comparado com a chance teórica, e o porquê de cada número."
    )
    aviso(AVISO_FREQUENCIA + " A taxa histórica descreve o passado; ela não muda a chance do próximo sorteio.")

    historico = carregar_historico()
    if historico.empty:
        st.warning("Nenhum histórico carregado. Importe os resultados na tela Histórico.")
        return
    caminho, versao = _chave_banco()
    incluir, inicio, fim = filtros_historico(historico, "taxa")
    if _dados_filtrados(caminho, versao, incluir, inicio, fim, None).empty:
        st.info("Nenhum concurso com os filtros escolhidos.")
        return
    a = analise_taxa(caminho, versao, incluir, inicio, fim)
    concursos = a["concursos"]
    st.caption(f"Base analisada: {formatar_milhar(concursos)} concursos e "
               f"{formatar_milhar(a['numeros'])} números premiados.")

    # 1. Prêmio principal ------------------------------------------------------
    st.subheader("1. Prêmio principal (número exato)")
    rep = a["repeticoes"]
    mais = rep["numeros_repetidos"]
    maximo_vezes = int(mais["vezes"].max()) if not mais.empty else 1
    st.write(
        f"A chance de um bilhete acertar um prêmio principal é de 1 em dezenas de milhares "
        f"(ex.: 1 em 92.000 na QUARTOU!). Mesmo com {formatar_milhar(concursos)} concursos, um número "
        f"específico costuma sair zero ou uma vez; o que mais saiu apareceu {maximo_vezes} vez(es). "
        "Por isso, não existe uma taxa histórica confiável para um número exato: a amostra é "
        "pequena demais diante do tamanho do universo."
    )
    st.caption(JUSTIFICATIVA_PRINCIPAL)

    # 2. Taxa por terminação ---------------------------------------------------
    st.subheader("2. Taxa de acerto histórica por terminação")
    st.write(
        "Para cada concurso, contamos quantas terminações diferentes saíram nos cinco prêmios. "
        "Isso dá a fração de bilhetes que teriam coincidido com pelo menos um prêmio: a taxa de "
        "acerto histórica de um bilhete qualquer."
    )
    linhas = []
    for digitos in (1, 2, 3, 4):
        t = a["coincidencia"][digitos]
        linhas.append({
            "Terminação": TERMINACOES_NOMES[digitos],
            "Taxa histórica": pct(t["taxa_observada"], 3),
            "Intervalo de 95%": f"{pct(t['faixa_95_min'], 3)} a {pct(t['faixa_95_max'], 3)}",
            "Chance teórica": pct(t["taxa_teorica"], 3),
            "Teoria dentro do intervalo?": "Sim" if t["compativel"] else "Não",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    with st.expander("Por quê: de onde vem cada número desta tabela", expanded=True):
        st.markdown(
            "- **Chance teórica** = 1 − (1 − 1/10^d)^5, onde d é o número de dígitos da terminação. "
            "Cada prêmio termina numa terminação específica com chance 1/10^d, porque nos planos "
            "atuais a quantidade de bilhetes é múltiplo de 1.000. "
            + JUSTIFICATIVA_INDEPENDENCIA
            + "\n- **Taxa histórica** = média, entre os concursos, de (terminações distintas nos 5 "
            "prêmios) ÷ 10^d.\n"
            "- **Intervalo de 95%** = taxa histórica ± 1,96 × erro padrão. Se a teoria cai dentro "
            "dele, a diferença observada é do tamanho esperado para o acaso.\n"
            "- Para épocas antigas, a quantidade oficial de bilhetes não está documentada no app; "
            "a chance teórica de terminação continua válida quando essa quantidade é múltiplo de "
            "10^d, e fica muito próxima disso nos demais casos. A milhar é a mais sensível a isso."
        )

    # 3. Unidade do 1º prêmio --------------------------------------------------
    st.subheader("3. Unidade do 1º prêmio, final por final")
    teste = a["unidade_primeiro"]
    tabela = teste["tabela"]
    fig = go.Figure()
    fig.add_bar(x=tabela["terminacao"], y=tabela["taxa_observada"] * 100, name="Taxa histórica",
                marker_color=CORES["observado"])
    fig.add_scatter(x=tabela["terminacao"], y=tabela["faixa_95_max"] * 100, mode="lines",
                    name="Faixa de 95% do acaso", line=dict(color=CORES["esperado"], dash="dot"))
    fig.add_scatter(x=tabela["terminacao"], y=tabela["faixa_95_min"] * 100, mode="lines",
                    showlegend=False, line=dict(color=CORES["esperado"], dash="dot"))
    fig.add_hline(y=10, line_color=CORES["esperado"], annotation_text="teórica 10%")
    fig.update_layout(title="Em quantos % dos concursos o 1º prêmio terminou em cada algarismo",
                      xaxis_title="Final", yaxis_title="%")
    st.plotly_chart(layout_grafico(fig))
    st.write(
        f"Qui-quadrado = {teste['qui_quadrado']:.2f} com {teste['graus_liberdade']} graus de "
        f"liberdade; p-valor = {teste['p_valor']:.3f}: {interpretar_p(teste['p_valor'])}. "
        f"Finais fora da faixa de 95%: {teste['fora_da_faixa']} (ao acaso, espera-se cerca de "
        f"{teste['fora_da_faixa_esperado']:.1f} em 10)."
    )
    st.caption(
        "Por quê: a chance teórica é 1/10 para cada final, porque a quantidade de bilhetes é "
        "múltiplo de 10. A faixa de 95% é 10% ± 1,96 × √(0,1 × 0,9 / n), com n = número de "
        "concursos. O qui-quadrado mede o quanto os dez finais, juntos, se afastam do esperado."
    )

    # 4. Uniformidade das terminações -----------------------------------------
    st.subheader("4. As terminações saem por igual?")
    linhas = []
    for digitos in (1, 2, 3, 4):
        t = a["uniformidade"][digitos]
        linhas.append({
            "Terminação": TERMINACOES_NOMES[digitos],
            "Números analisados": formatar_milhar(t["n"]),
            "Esperado por terminação": f"{t['esperado_por_terminacao']:.1f}".replace(".", ","),
            "Qui-quadrado": round(t["qui_quadrado"], 2),
            "Graus de liberdade": t["graus_liberdade"],
            "p-valor": round(t["p_valor"], 4) if t["teste_valido"] else None,
            "Leitura": interpretar_p(t["p_valor"]) if t["teste_valido"]
                       else "teste não confiável: menos de 5 esperados por terminação",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption(
        "Por quê: o teste qui-quadrado compara as contagens observadas com n ÷ 10^d em cada "
        "terminação. Ele só é confiável quando se esperam pelo menos 5 ocorrências por terminação; "
        "por isso a milhar normalmente fica sem p-valor."
    )

    qq = a["qui_quadrado"]
    fig = go.Figure()
    cores = [CORES["observado"] if v else "#5A6070" for v in qq["valido"]]
    fig.add_bar(x=qq["teste"], y=qq["razao"], marker_color=cores,
                text=[f"p = {p:.3f}" if v else "não confiável" for p, v in zip(qq["p_valor"], qq["valido"])],
                textposition="outside", name="χ² ÷ valor crítico")
    fig.add_hline(y=1, line_dash="dot", line_color="#E07A7A",
                  annotation_text="limite de 95% (p = 0,05)")
    fig.update_layout(title="Testes qui-quadrado: distância até o limite do acaso",
                      yaxis_title="χ² observado ÷ valor crítico de 95%", showlegend=False)
    fig.update_yaxes(range=[0, max(1.3, float(qq["razao"].max()) * 1.2)])
    st.plotly_chart(layout_grafico(fig, 380))
    st.caption(
        "Por quê: cada barra divide o qui-quadrado observado pelo valor que só é ultrapassado "
        "em 5% dos sorteios justos (valor crítico). Abaixo de 1 = dentro do esperado para o "
        "acaso. Barras cinza são testes sem amostra suficiente (menos de 5 esperados por "
        "terminação), mostrados só para referência."
    )
    st.dataframe(
        qq.assign(
            qui_quadrado=qq["qui_quadrado"].round(2),
            valor_critico_95=qq["valor_critico_95"].round(2),
            razao=qq["razao"].round(3),
            p_valor=qq["p_valor"].round(4),
        ).rename(columns={"teste": "Teste", "n": "Números", "qui_quadrado": "χ²",
                          "graus_liberdade": "Graus de liberdade", "valor_critico_95": "Valor crítico 95%",
                          "razao": "χ² ÷ crítico", "p_valor": "p-valor", "valido": "Confiável"}),
        hide_index=True,
    )

    # 5. Candidatos do app ----------------------------------------------------
    st.subheader("5. Taxa de acerto dos seus candidatos")
    resumo = est.taxa_candidatos(db.listar_conferencias())
    if resumo is None:
        st.info("Nenhum candidato foi conferido ainda (tela Conferência).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidatos conferidos", resumo["conferidos"])
        c2.metric("Acertos exatos", resumo["exatos"],
                  help=f"Esperado ao acaso: no máximo {resumo['esperado_exatos_max']:.5f}")
        c3.metric("Unidade do 1º prêmio", resumo["unidade_primeiro"],
                  help=f"Esperado ao acaso: cerca de {resumo['esperado_unidade_primeiro']:.1f}")
        st.caption(
            f"Esperado ao acaso: até {resumo['esperado_exatos_max']:.5f} acerto(s) exato(s) "
            f"(5 ÷ N por candidato) e cerca de {resumo['esperado_unidade_primeiro']:.1f} "
            "coincidência(s) de unidade do 1º prêmio (1 ÷ 10 por candidato). Os candidatos são "
            "gerados ao acaso, então a taxa deles segue a mesma teoria."
        )


def pagina_fontes() -> None:
    st.title("Fontes e regras")
    st.caption(f"Documentos consultados em {DATA_CONSULTA}.")
    st.markdown(
        f"""
**Documentos oficiais**
- [{FONTE_MANUAL}]({FONTE_MANUAL_URL})
- [{FONTE_REGULAMENTO}]({FONTE_REGULAMENTO_URL})
- [Página oficial da Loteria Federal]({FONTE_PAGINA})

**Faixa de números de cada extração**

{REGRA_UNIVERSO}
        """
    )
    for e in EXTRACOES.values():
        st.markdown(f"- **{e.nome}** ({e.faixa_texto}): {e.fonte_quantidade} {e.fonte_chance}")
    st.markdown(
        """
**Regras usadas na conferência** (Manual v23, item 5.4.4)
- Prêmio principal: número igual a um dos cinco números sorteados.
- Milhar, centena e dezena de qualquer um dos cinco prêmios principais.
- Aproximação imediatamente anterior e posterior ao 1º prêmio.
- Dezena final igual a uma das 3 dezenas anteriores ou 3 posteriores à dezena do 1º prêmio,
  exceto quem já ganhou pela aproximação.
- Unidade do 1º prêmio.

**O que o aplicativo não faz**
- O Gerador não usa o histórico para escolher números. Só a tela Previsão do próximo sorteio usa,
  como experimento separado, sempre com a probabilidade oficial ao lado e o resultado do backtest.
- Não mostra pontuação ou frequência como se fosse chance de ganhar.
- Não calcula valores de prêmio, nem se faixas se acumulam: isso depende do plano de sorteio
  impresso no bilhete (Manual, item 5.4.9).
- Não avalia casos que o Manual não define, como a aproximação do primeiro ou do último bilhete.
        """
    )


# ------------------------------------------------------------------ navegação

PAGINAS = {
    "Início": pagina_inicio,
    "Gerador": pagina_gerador,
    "Histórico": pagina_historico,
    "Conferência": pagina_conferencia,
    "Probabilidades": pagina_probabilidades,
    "Taxa histórica": pagina_taxa_historica,
    "Monitoramento do sorteio": pagina_monitoramento,
    "Previsão do próximo sorteio": pagina_previsao,
    "Laboratório de estratégias": pagina_laboratorio,
    "Fontes e regras": pagina_fontes,
}

with st.sidebar:
    st.markdown("### Federal Intelligence")
    st.caption("Beta 0.1 · uso educacional")
    escolha = st.radio("Navegação", list(PAGINAS), label_visibility="collapsed")
    st.divider()
    st.caption("Não incentiva apostas e não promete ganhos.")

PAGINAS[escolha]()
