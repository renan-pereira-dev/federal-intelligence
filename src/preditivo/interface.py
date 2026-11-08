"""Tela "Previsão do próximo sorteio" (Streamlit).

Fica separada do app.py para manter o motor independente. Tudo o que é exibido
vem do que foi gravado no banco; sem previsão gravada, nada é mostrado.
"""

from __future__ import annotations

import html
import json
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src.preditivo import armazenamento, calendario, motor
from src.preditivo.modelos import NOMES_MODELOS
from src.probability import EXTRACOES, explicacao_probabilidade, extracoes_habilitadas

AVISO_GERAL = (
    "Motor experimental. A pontuação preditiva é uma <b>ordem de preferência do modelo</b>, "
    "não uma probabilidade. A pontuação ou frequência histórica não representa maior chance de "
    "ganhar. Em um sorteio justo, todos os números elegíveis possuem a mesma probabilidade. "
    "Os números são candidatos experimentais, não bilhetes disponíveis ou garantidos para compra."
)
CORES_MODELO = "#4FB3A9"

CSS = """
<style>
.fi-destaque {background:linear-gradient(135deg,#14262A,#161B26); border:1px solid #2F6F68;
              border-radius:16px; padding:20px 24px; margin:8px 0 14px 0;}
.fi-destaque .fi-numero {font-size:3.2rem;}
.fi-alerta {display:inline-block; font-size:0.78rem; padding:3px 10px; border-radius:999px;
            background:#3A2A12; color:#F2C27B; margin:6px 0;}
.fi-ok {display:inline-block; font-size:0.78rem; padding:3px 10px; border-radius:999px;
        background:#173A2A; color:#8FE0B5; margin:6px 0;}
.fi-alt .fi-numero {font-size:1.6rem;}
</style>
"""


def _aviso(texto: str) -> None:
    st.markdown(f'<div class="fi-aviso">{texto}</div>', unsafe_allow_html=True)


def _layout(fig: go.Figure, altura: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=altura, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def _num(valor: float, casas: int = 2) -> str:
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(valor: float | None, casas: int = 2) -> str:
    return "não calculado" if valor is None else _num(valor * 100, casas) + "%"


def _data_br(iso: str | None) -> str:
    if not iso:
        return "não informada"
    try:
        return date.fromisoformat(iso[:10]).strftime("%d/%m/%Y") + (iso[10:].replace("T", " às ") if len(iso) > 10 else "")
    except ValueError:
        return iso


def _ultimo_concurso() -> tuple[int, str] | None:
    with db.conectar() as conexao:
        linha = conexao.execute("SELECT concurso, data FROM contests ORDER BY concurso DESC LIMIT 1").fetchone()
    return (int(linha["concurso"]), linha["data"]) if linha else None


# ------------------------------------------------------------------ página

def pagina_previsao() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Previsão do próximo sorteio")
    st.caption("Motor experimental de análise estatística e aprendizado de máquina, "
               "avaliado em backtest cronológico contra o acaso.")
    _aviso(AVISO_GERAL)

    armazenamento.inicializar()
    conferidas = armazenamento.conferir_pendentes()
    if conferidas:
        st.success(f"{len(conferidas)} previsão(ões) comparada(s) automaticamente com o resultado oficial.")

    ultimo = _ultimo_concurso()
    if ultimo is None:
        st.warning("Os dados históricos precisam ser carregados antes de gerar previsões. "
                   "Importe o CSV na tela Histórico.")
        _secao_historico_previsoes()
        return

    alvo = _secao_alvo(*ultimo)
    _secao_botoes(alvo)

    previsoes = armazenamento.listar_previsoes()
    if previsoes.empty:
        st.info("Nenhuma previsão gerada ainda. Clique em GERAR PREVISÃO: o motor treina e testa os "
                "modelos com o histórico real e grava o resultado.")
        return
    atual = st.session_state.get("previsao_atual")
    if atual not in set(previsoes["id"]):
        atual = previsoes["id"].iloc[0]
    exibir_previsao(armazenamento.carregar_previsao(atual))
    _secao_historico_previsoes()


def _secao_alvo(ultimo: int, ultima_data: str) -> dict:
    st.subheader("1. Próximo concurso")
    chave = f"alvo_{ultimo}_{ultima_data}"
    if chave not in st.session_state:
        st.session_state[chave] = calendario.estimar_proximo(ultimo, ultima_data)
    alvo = st.session_state[chave]

    c1, c2, c3 = st.columns(3)
    c1.metric("Concurso", alvo["concurso"])
    c2.metric(f"Data ({alvo['status']})", _data_br(alvo["data"]))
    c3.metric("Último concurso no banco", f"{ultimo} · {_data_br(ultima_data)}")
    for texto in alvo["observacoes"]:
        st.caption(texto)
    st.caption(f"Fonte: {alvo['fonte']}")
    if st.button("Consultar calendário oficial da CAIXA", help=calendario.URL_SERVICO_OFICIAL):
        with st.spinner("Consultando o serviço oficial..."):
            st.session_state[chave] = calendario.consultar_oficial(calendario.estimar_proximo(ultimo, ultima_data))
        st.rerun()

    opcoes = [e.chave for e in extracoes_habilitadas()]
    padrao = opcoes.index(alvo["extracao"]) if alvo["extracao"] in opcoes else 0
    extracao = st.selectbox(
        f"Extração do concurso {alvo['concurso']} ({alvo.get('extracao_status', 'estimada')})",
        opcoes, index=padrao, format_func=lambda k: EXTRACOES[k].nome,
        help="Define o universo de números válidos dos candidatos. Ajuste se a CAIXA anunciar outra extração.",
    )
    e = EXTRACOES[extracao]
    st.caption(f"Universo de {e.nome}: {e.faixa_texto}. Probabilidade oficial do prêmio principal: {e.chance_texto}.")
    return {**alvo, "extracao_escolhida": extracao}


def _secao_botoes(alvo: dict) -> None:
    st.subheader("2. Executar o motor")
    st.caption(
        "GERAR PREVISÃO usa os modelos já treinados para esta versão do histórico (treina só se faltar). "
        "GERAR NOVA PREVISÃO treina de novo com outra amostra aleatória. As previsões anteriores são "
        "sempre preservadas. A primeira execução após importar dados roda o backtest e pode levar "
        "alguns minutos."
    )
    c1, c2 = st.columns(2)
    gerar = c1.button(motor.GERAR, type="primary", width="stretch")
    nova = c2.button(motor.GERAR_NOVA, width="stretch")
    if not (gerar or nova):
        return
    barra = st.progress(0.0, text="Iniciando...")
    alvo_gravado = {k: v for k, v in alvo.items() if k != "extracao_escolhida"}
    try:
        identificador = motor.gerar_previsao(
            alvo["extracao_escolhida"], alvo_gravado, nova=nova,
            progresso=lambda f, t: barra.progress(min(max(f, 0.0), 1.0), text=t),
        )
    except motor.ErroMotor as exc:
        barra.empty()
        st.error(str(exc))
        return
    barra.empty()
    st.session_state["previsao_atual"] = identificador
    st.success(f"Previsão {identificador} gerada e gravada.")


# ------------------------------------------------------------------ exibição

def exibir_previsao(p: dict) -> None:
    r = p["resumo"]
    st.subheader("3. Resultado")
    alvo = r.get("alvo", {})
    st.caption(
        f"Previsão **{p['id']}** · {p['tipo_execucao']} · gerada em {_data_br(p['criado_em'])} · "
        f"concurso alvo {p['concurso_alvo']} ({_data_br(p['data_alvo'])}, data {p['data_status']}) · "
        f"extração {r['extracao_nome']} ({r['universo']}) · histórico até o concurso {p['ultimo_concurso']}"
    )
    if not p["validado"]:
        _aviso(f"<b>{motor.AVISO_SEM_VANTAGEM}.</b> No backtest cronológico, o conjunto de modelos não "
               "mostrou vantagem estatisticamente significativa sobre uma escolha ao acaso. Por isso o "
               "número abaixo é apenas o 1º do ranking do modelo, não uma recomendação.")

    principal, *alternativos = p["itens"]
    selo = (f'<span class="fi-alerta">{motor.AVISO_SEM_VANTAGEM}</span>' if not p["validado"]
            else '<span class="fi-ok">Vantagem observada no backtest</span>')
    modelos_usados = ", ".join(NOMES_MODELOS[k] for k in r["modelos_no_conjunto"])
    st.markdown(
        f'<div class="fi-destaque"><div class="fi-rotulo">{html.escape(p["rotulo_principal"])}</div>'
        f'<div class="fi-numero">{principal["numero"]}</div>{selo}'
        f'<div class="fi-valor">Pontuação preditiva: <b>{_num(principal["pontuacao"])}</b> de 100 '
        f'(posição relativa da milhar {principal["detalhes"]["milhar"]} no ranking do modelo; não é probabilidade)</div>'
        f'<div class="fi-valor">Probabilidade matemática oficial: <b>{html.escape(r["probabilidade"]["chance_texto"])}</b>, '
        f'igual para todos os números de {html.escape(r["universo"])}</div>'
        f'<div class="fi-valor">Modelo responsável: conjunto de {len(r["modelos_no_conjunto"])} modelos com peso igual '
        f'({html.escape(modelos_usados)})</div>'
        f'{_conferencia_html(principal)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Alternativos (2º ao 10º do ranking)**")
    cartoes = "".join(
        f'<div class="fi-card fi-alt"><div class="fi-rotulo">{i["posicao"]}º do ranking</div>'
        f'<div class="fi-numero">{i["numero"]}</div>'
        f'<div class="fi-valor">Pontuação {_num(i["pontuacao"])}</div>{_conferencia_html(i)}</div>'
        for i in alternativos
    )
    st.markdown(f'<div class="fi-grid">{cartoes}</div>', unsafe_allow_html=True)
    if p.get("resultado"):
        _resultado_conferencia(p)

    abas = st.tabs(["Ranking", "A · Probabilidade teórica", "B · Pontuação preditiva",
                    "C · Frequência histórica", "D · Desempenho no backtest", "Modelos e indicadores"])
    with abas[0]:
        _aba_ranking(p)
    with abas[1]:
        _aba_probabilidade(p)
    with abas[2]:
        _aba_pontuacao(p)
    with abas[3]:
        _aba_frequencia(p)
    with abas[4]:
        _aba_backtest(r["backtest"])
    with abas[5]:
        _aba_modelos(p)


def _conferencia_html(item: dict) -> str:
    c = item.get("conferencia")
    if not c:
        return ""
    return f'<div class="fi-tag">{html.escape(c["resumo"])}</div>'


def _resultado_conferencia(p: dict) -> None:
    res = p["resultado"]
    st.markdown(f"**Comparação com o resultado oficial do concurso {res['concurso']} "
                f"({_data_br(res['data'])}):** {', '.join(res['premios'])}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acertos exatos (prêmio principal)", res["acertos_exatos"])
    c2.metric("Coincidências de milhar", res.get("coincidencias_milhar", 0))
    c3.metric("Coincidências de centena", res.get("coincidencias_centena", 0))
    c4.metric("Coincidências de dezena", res.get("coincidencias_dezena", 0))
    st.caption("Coincidências parciais não são acerto do prêmio principal. Valores e acumulação das faixas "
               "derivadas dependem do plano impresso no bilhete.")


def _aba_ranking(p: dict) -> None:
    linhas = []
    for i in p["itens"]:
        d = i["detalhes"]
        linhas.append({
            "Posição": i["posicao"],
            "Número": i["numero"],
            "Pontuação preditiva (0–100)": i["pontuacao"],
            "Milhar avaliada": d["milhar"],
            "Posição da milhar entre 10.000": d["posicao_ranking_milhar"],
            "Principais fatores": "; ".join(f["descricao"] for f in d["fatores"]),
            "Vezes que o número saiu": d["historico"]["vezes_exato"],
            "Conferência": (i["conferencia"] or {}).get("resumo", "aguardando resultado"),
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption("O modelo avalia só os 4 últimos algarismos (a milhar). O 1º algarismo é sorteado ao acaso, "
               "com a fonte segura do Python, entre os válidos no universo da extração, porque o universo "
               "mudou ao longo das décadas e o modelo não deve aprender mudanças de regra.")


def _aba_probabilidade(p: dict) -> None:
    r = p["resumo"]
    st.markdown("#### A · Probabilidade matemática teórica (regras oficiais)")
    st.metric("Chance de o número ser sorteado no prêmio principal", r["probabilidade"]["chance_texto"])
    st.write(explicacao_probabilidade(p["extracao"]))
    st.info("Esta probabilidade é a mesma para os 10 candidatos e para qualquer outro número do universo "
            f"{r['universo']}. A pontuação preditiva (bloco B) não altera esse valor.")


def _aba_pontuacao(p: dict) -> None:
    r = p["resumo"]
    st.markdown("#### B · Pontuação preditiva (ranking do modelo, não é probabilidade)")
    st.write("Cada modelo ordena as 10.000 milhares. A nota de um modelo é a posição relativa da milhar "
             "(100 = primeira). A pontuação final é a posição relativa da média das notas dos modelos, "
             "todos com o mesmo peso.")
    linhas = [{"Número": i["numero"], "Modelo": NOMES_MODELOS[m], "Nota (0–100)": v}
              for i in p["itens"] for m, v in i["detalhes"]["contribuicoes"].items()]
    df = pd.DataFrame(linhas)
    fig = px.bar(df, x="Número", y="Nota (0–100)", color="Modelo", barmode="group",
                 title="Contribuição de cada modelo para os 10 candidatos")
    fig.update_yaxes(range=[min(80, df["Nota (0–100)"].min() - 1), 100.5])
    st.plotly_chart(_layout(fig, 380))
    st.dataframe(df.pivot(index="Número", columns="Modelo", values="Nota (0–100)").reindex(
        [i["numero"] for i in p["itens"]]))
    st.markdown("**Por que cada candidato ficou no topo** (indicadores em que a milhar mais se afasta das demais)")
    for i in p["itens"]:
        fatores = "; ".join(
            f"{f['descricao']}: {_num(f['valor'], 3)} (maior que {_num(f['percentil'], 1)}% das milhares)"
            for f in i["detalhes"]["fatores"])
        st.caption(f"**{i['numero']}**: {fatores}")
    st.caption("Frequências aparecem como razão observado/esperado: 1,000 é exatamente o que o acaso produziria.")


def _aba_frequencia(p: dict) -> None:
    r = p["resumo"]
    st.markdown("#### C · Frequência histórica observada")
    linhas = []
    for i in p["itens"]:
        h = i["detalhes"]["historico"]
        linha = {"Número": i["numero"], "Vezes que saiu (exato)": h["vezes_exato"],
                 "Último concurso em que saiu": str(h["ultima_vez_concurso"] or "nunca")}
        for nome, t in h["terminacoes"].items():
            linha[f"{nome.capitalize()} {t['valor']}: vezes"] = t["vezes"]
            linha[f"{nome.capitalize()}: esperado"] = t["esperado"]
        linhas.append(linha)
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption("Frequência passada descreve o que aconteceu; não aumenta nem diminui a chance futura. "
               "Um número ausente há muito tempo não fica 'mais provável'.")

    est = r["estatisticas"]
    st.markdown(f"**Indicadores gerais** ({est['concursos']} concursos, até o {est['ultimo_concurso']}; "
                f"{est['numeros_repetidos']} números saíram mais de uma vez)")
    nivel = st.segmented_control("Terminação", ["dezena", "centena", "milhar"], default="centena",
                                 key=f"nivel_{p['id']}") or "centena"
    dados = est[nivel]
    c1, c2, c3 = st.columns(3)
    for coluna, titulo, chave in ((c1, "Mais frequentes", "mais_frequentes"),
                                  (c2, "Menos frequentes", "menos_frequentes"),
                                  (c3, "Maiores atrasos (concursos)", "maiores_atrasos")):
        coluna.markdown(f"**{titulo}**")
        coluna.dataframe(pd.DataFrame(dados[chave])[["valor", "total", "ultimos_100", "ultimos_1000", "atraso"]],
                         hide_index=True)
    st.caption(f"Esperado por {nivel} em todo o histórico: {_num(dados['esperado_total'])} vezes.")
    pos = est["algarismos_por_posicao"]
    df = pd.DataFrame([{"Posição": f"{k}ª", "Algarismo": str(a), "Últimos 500": v}
                       for k, d in pos.items() for a, v in enumerate(d["ultimos_500"])])
    fig = px.bar(df, x="Algarismo", y="Últimos 500", color="Posição", barmode="group",
                 title="Algarismos por posição (2ª a 5ª) nos últimos 500 concursos")
    fig.add_hline(y=250, line_dash="dot", annotation_text="esperado: 250")
    st.plotly_chart(_layout(fig))


def _aba_backtest(b: dict) -> None:
    st.markdown("#### D · Desempenho real dos modelos em testes históricos")
    if b.get("erro"):
        st.warning(f"O backtest não pôde ser calculado: {b['erro']}")
        return
    st.write(
        f"Teste walk-forward em {b['concursos_testados']} concursos ({b['primeiro_concurso']} a "
        f"{b['ultimo_concurso']}), em {b['blocos']} blocos. Antes de cada bloco os modelos foram treinados "
        "só com concursos anteriores; cada concurso foi avaliado só com dados anteriores a ele."
    )
    t = b["testes_posicao"]["conjunto"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Posição média das milhares sorteadas", _num(t["media"], 4), help="0,5 = acaso; menor = melhor")
    c2.metric("Valor-p (teste unilateral)", _num(t["p"], 4))
    c3.metric("Resultado", "Vantagem significativa" if b["validado"] else "Sem vantagem comprovada")
    estrutura = b.get("conjunto_contra_estrutura")
    if estrutura:
        criterios = b["criterios"]
        e1, e2 = st.columns(2)
        e1.metric("1. Abaixo de 0,5 (acaso)", "sim" if criterios["abaixo_do_acaso"] else "não")
        e2.metric("2. Melhor que a estrutura do universo", "sim" if criterios["melhor_que_estrutura"] else "não",
                  help=f"Diferença média {_num(estrutura['diferenca'], 5)} (negativa = conjunto melhor); "
                       f"valor-p {_num(estrutura['p'], 4)}")
        st.caption(
            f"Critério definido antes do teste: o conjunto só é validado se cumprir os dois itens, cada um com "
            f"p < {_num(b['alfa'], 2)}. Foram {t['n']} milhares sorteadas avaliadas. O item 2 existe porque, em "
            "universos como 96.000, algumas milhares têm 10 bilhetes válidos e outras 9: ficar abaixo de 0,5 só por "
            "isso não é previsão, já que o número inteiro continua com a mesma chance.")
    else:
        st.caption(f"Critério deste backtest (versão anterior): posição média abaixo de 0,5 com "
                   f"p < {_num(b['alfa'], 2)}. Foram {t['n']} milhares sorteadas avaliadas.")

    coinc = b["coincidencias"]
    nomes = {"exato": "Acerto exato (número inteiro)", "milhar": "Milhar", "centena": "Centena", "dezena": "Dezena"}
    st.markdown("**Top 10 contra o acaso**")
    st.dataframe(pd.DataFrame([{
        "Tipo": nomes[k],
        "Modelo (top 10)": int(v["modelo"]),
        "Esperado para a seleção do modelo": round(v["esperado_teorico"], 2),
        "Valor-p (modelo acima do esperado)": None if v["p_mais_que_acaso"] is None else round(v["p_mais_que_acaso"], 4),
        "Seleção aleatória (10 números)": int(v["acaso_simulado"]),
        "Esperado para a seleção aleatória": round(v["esperado_acaso"], 2) if "esperado_acaso" in v else "não registrado",
    } for k, v in coinc.items()]), hide_index=True)
    st.caption("Cada seleção deve ser comparada com o seu próprio esperado. O número esperado de coincidências "
               "depende de quantas terminações diferentes os 10 números cobrem: se o modelo concentra os "
               "candidatos em poucas dezenas ou centenas, ele cobre menos finais e o esperado cai. Isso não é "
               "nem vantagem nem desvantagem real.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acertos do número principal", b["acertos_principal"],
              help=f"Seleção aleatória: {b['acertos_principal_acaso']}; esperado: {_num(b['acertos_principal_esperado'], 3)}")
    c2.metric("Acertos exatos no top 10", b["acertos_top10"], help=f"Seleção aleatória: {b['acertos_top10_acaso']}")
    c3.metric("Precision@10", _pct(b["precision_10"], 3), help=f"Acaso: {_pct(b['precision_10_acaso'], 3)}")
    c4.metric("Recall@10", _pct(b["recall_10"], 3), help=f"Acaso: {_pct(b['recall_10_acaso'], 3)}")

    st.markdown("**Resultado por modelo**")
    st.dataframe(pd.DataFrame([{
        "Modelo": {"conjunto": "Conjunto (usado na previsão)",
                   "estrutura": "Referência: estrutura do universo"}.get(k, NOMES_MODELOS.get(k, k)),
        "Posição média": round(v["media"], 4), "z": round(v["z"], 2), "Valor-p": round(v["p"], 4),
        "Milhares avaliadas": v["n"],
    } for k, v in b["testes_posicao"].items()]), hide_index=True)
    st.caption("Com vários modelos, algum pode parecer bom por sorte. Só o conjunto, definido antes do teste, "
               "decide a validação.")

    st.markdown("**Resultado por período**")
    periodos = pd.DataFrame(b["por_periodo"])
    fig = go.Figure(go.Bar(x=[f"{a}–{z}" for a, z in zip(periodos["primeiro"], periodos["ultimo"])],
                           y=periodos["posicao_media"], marker_color=CORES_MODELO, name="Posição média"))
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="acaso: 0,5")
    fig.update_layout(title="Posição média das milhares sorteadas por bloco (menor = melhor)")
    st.plotly_chart(_layout(fig, 300))
    st.dataframe(periodos.rename(columns={
        "bloco": "Bloco", "primeiro": "De", "ultimo": "Até", "concursos": "Concursos",
        "milhar_modelo": "Milhares coincidentes", "milhar_esperado": "Milhares esperadas",
        "centena_modelo": "Centenas coincidentes", "centena_esperado": "Centenas esperadas",
        "exato_modelo": "Exatos (modelo)", "exato_acaso": "Exatos (acaso)", "posicao_media": "Posição média",
    }), hide_index=True)
    st.caption(f"Backtest calculado em {_data_br(b.get('criado_em'))}, em {_num(b.get('duracao_segundos', 0), 1)} s.")


def _aba_modelos(p: dict) -> None:
    r = p["resumo"]
    st.markdown("#### Modelos executados")
    st.dataframe(pd.DataFrame([{"Modelo": m["nome"], "Situação": m["status"]} for m in r["modelos"].values()]),
                 hide_index=True)
    t = r["treino"]
    st.caption(
        f"Treino com os concursos {t['primeiro_concurso']} a {t['ultimo_concurso']} ({t['exemplos']} exemplos, "
        f"{t['positivos']} milhares sorteadas e as demais aleatórias). Semente de amostragem {t['semente']}. "
        + ("Modelos treinados nesta execução." if r["treinou_agora"] else "Modelos reaproveitados do disco.")
    )
    if r["importancias"]:
        df = pd.DataFrame(r["importancias"]).sort_values("peso")
        fig = px.bar(df, x="peso", y="descricao", orientation="h",
                     title="Indicadores mais usados pelos modelos (média dos pesos)")
        fig.update_traces(marker_color=CORES_MODELO)
        st.plotly_chart(_layout(fig, 460))
        st.caption("Peso alto quer dizer que o modelo usou muito o indicador, não que ele preveja o sorteio.")
    st.markdown("**Indicadores calculados** (sempre só com concursos anteriores ao avaliado)")
    st.write(
        "Frequência das terminações (unidade, dezena, centena, milhar) no histórico inteiro e nos últimos "
        "50, 100, 500 e 1000 concursos; atraso desde a última saída; tendência (últimos 100 contra 1000); "
        "repetição no concurso anterior; frequência dos algarismos por posição; soma, paridade e algarismos "
        "distintos da milhar."
    )
    st.json(r["config"], expanded=False)


def _secao_historico_previsoes() -> None:
    previsoes = armazenamento.listar_previsoes()
    if previsoes.empty:
        return
    st.subheader("4. Histórico de previsões")
    st.caption("Todas as previsões ficam gravadas. Quando o resultado oficial do concurso alvo entra no "
               "banco (importação ou tela Conferência), a comparação é feita automaticamente.")
    tabela = previsoes.assign(
        Resultado=previsoes["resultado"].map(_resumo_resultado),
        Validado=previsoes["validado"].map({1: "sim", 0: "não"}),
    ).rename(columns={"id": "ID", "criado_em": "Gerada em", "tipo_execucao": "Execução",
                      "concurso_alvo": "Concurso alvo", "data_alvo": "Data alvo", "data_status": "Data",
                      "extracao": "Extração", "principal": "1º do ranking"})
    st.dataframe(tabela[["ID", "Gerada em", "Execução", "Concurso alvo", "Data alvo", "Data", "Extração",
                         "1º do ranking", "Validado", "Resultado"]], hide_index=True)
    escolha = st.selectbox("Abrir uma previsão anterior", previsoes["id"], key="abrir_previsao")
    if st.button("Abrir previsão selecionada"):
        st.session_state["previsao_atual"] = escolha
        st.rerun()


def _resumo_resultado(texto: str | None) -> str:
    if not texto:
        return "aguardando resultado"
    r = json.loads(texto)
    return (f"{r['acertos_exatos']} exato(s); milhar {r.get('coincidencias_milhar', 0)}, "
            f"centena {r.get('coincidencias_centena', 0)}, dezena {r.get('coincidencias_dezena', 0)}")
