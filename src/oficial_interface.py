"""Seção "Resultados oficiais (CAIXA)" do painel: buscar, conferir a prévia e confirmar.

Usada nas telas Histórico e Previsão do próximo sorteio. A consulta à internet
só acontece quando o usuário clica em "Buscar resultados oficiais", e a
gravação só depois de "Confirmar e gravar".
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import oficial

CHAVE_PREVIA = "oficial_previa"
CHAVE_MENSAGEM = "oficial_mensagem"


def _data_br(iso: str) -> str:
    return f"{iso[8:10]}/{iso[5:7]}/{iso[0:4]}" if iso and len(iso) >= 10 else (iso or "")


def secao_resultados_oficiais(chave: str) -> None:
    mensagem = st.session_state.pop(CHAVE_MENSAGEM, None)
    if mensagem:
        st.success(mensagem)

    st.caption("Consulta o serviço oficial da CAIXA, mostra o que foi encontrado e só grava depois da sua "
               "confirmação. Nada é sobrescrito: se um concurso já gravado vier diferente, nada é gravado.")
    if st.button("Buscar resultados oficiais", key=f"oficial_buscar_{chave}"):
        with st.spinner("Consultando o serviço oficial da CAIXA..."):
            st.session_state[CHAVE_PREVIA] = oficial.previsualizar()

    previa = st.session_state.get(CHAVE_PREVIA)
    if not previa:
        return
    if previa["erro"]:
        st.error(previa["erro"] + " Nada foi alterado.")
        if not previa["itens"]:
            return

    st.write(f"Último concurso no banco: **{previa['ultimo_local'] or 'nenhum'}** · último publicado pela CAIXA: "
             f"**{previa['ultimo_oficial'] or '—'}** · consulta em {previa['consultado_em'].replace('T', ' ')}")
    linhas = []
    for item in previa["itens"]:
        r = item["resultado"]
        linhas.append({
            "Concurso": item["concurso"],
            "Situação": oficial.ROTULOS[item["situacao"]],
            "Data": _data_br(r.data) if r else "",
            "Prêmios (1º ao 5º)": ", ".join(r.premios) if r else "",
            "Prêmio zero": ("sim" if r.premio_zero else "não") if r else "",
            "Observação": item["mensagem"] or (r.dados_originais.get("observacao") or "" if r else ""),
        })
    if linhas:
        st.dataframe(pd.DataFrame(linhas), hide_index=True)
    if previa["restantes"]:
        st.info(f"Faltam mais {previa['restantes']} concursos depois destes. Grave e busque de novo.")

    resumo = oficial.resumo_previa(previa)
    contagem = resumo["contagem"]
    if contagem[oficial.CONFLITO]:
        st.error("Há conflito entre o serviço e o banco. Nada será gravado; confira os concursos marcados.")
    if not contagem[oficial.NOVO] and not previa["erro"]:
        st.info("Nenhum resultado novo: o banco já está atualizado com o último concurso publicado.")
    c1, c2 = st.columns(2)
    if resumo["pode_gravar"]:
        if c1.button(f"Confirmar e gravar {contagem[oficial.NOVO]} concurso(s)", type="primary",
                     key=f"oficial_gravar_{chave}"):
            try:
                resultado = oficial.gravar(previa)
            except oficial.ErroValidacao as exc:
                st.error(str(exc))
                return
            st.cache_data.clear()  # Histórico, Taxa histórica e motor passam a ver os novos concursos
            st.session_state.pop(CHAVE_PREVIA, None)
            conferidas = len(resultado["previsoes_conferidas"])
            st.session_state[CHAVE_MENSAGEM] = (
                f"Gravado(s): concurso(s) {', '.join(map(str, resultado['gravados']))} (importação "
                f"{resultado['import_id']}). Previsões conferidas automaticamente: {conferidas}."
            )
            st.rerun()
    if c2.button("Descartar a prévia", key=f"oficial_descartar_{chave}"):
        st.session_state.pop(CHAVE_PREVIA, None)
        st.rerun()
