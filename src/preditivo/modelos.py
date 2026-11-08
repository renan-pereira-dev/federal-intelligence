"""Modelos do motor, treino com amostragem e gravação em disco.

Formulação: para cada concurso de treino, as 5 milhares sorteadas são
exemplos positivos e algumas milhares sorteadas ao acaso são negativos. Os
indicadores de cada exemplo usam só concursos anteriores. Cada modelo devolve
uma nota para as 10.000 milhares; notas são comparadas por posição (ranking),
nunca interpretadas como probabilidade.

Nenhum modelo é presumido melhor: o conjunto (ensemble) dá peso igual a todos
os que treinaram com sucesso, e o backtest mede cada um separadamente.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preditivo.atributos import UNIVERSO_MILHAR, IndiceHistorico, calcular_atributos


@dataclass(frozen=True)
class Configuracao:
    janela_treino: int = 2000      # concursos mais recentes usados no treino
    negativos: int = 15            # milhares aleatórias por concurso de treino
    concursos_teste: int = 500     # concursos avaliados no backtest
    bloco: int = 100               # re-treino a cada N concursos no backtest
    minimo_treino: int = 200       # menos que isso: o motor não treina
    arvores: int = 80
    iteracoes: int = 150
    semente: int = 20261107
    # Muda quando o conjunto de atributos muda; faz parte do hash, então backtests
    # e modelos gravados com outra versão não são reaproveitados (nem apagados).
    # Versão 2 (08/11/2026): sem posicao_5_*, que duplicavam freq_1_*.
    versao_atributos: int = 2

    def hash(self) -> str:
        texto = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(texto.encode()).hexdigest()[:12]


CONFIG_PADRAO = Configuracao()
# Usada nos testes automatizados e quando FI_MOTOR_CONFIG=rapido.
CONFIG_RAPIDA = Configuracao(janela_treino=300, negativos=8, concursos_teste=120, bloco=40,
                             minimo_treino=60, arvores=20, iteracoes=30)


def _xgboost_disponivel() -> tuple[bool, str]:
    try:
        import xgboost  # noqa: F401
        return True, xgboost.__version__
    except Exception as exc:  # pacote ausente ou incompatível
        return False, f"não disponível ({type(exc).__name__})"


class IndiceEstatistico:
    """Modelo sem aprendizado: média das frequências recentes das terminações."""

    COLUNAS = ["freq_1_500", "freq_2_500", "freq_3_500", "freq_4_total"]

    def fit(self, X, y):
        return self

    def nota(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.COLUNAS].mean(axis=1).to_numpy()


def _construtores(config: Configuracao, semente: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    construtores = {
        "estatistico": lambda: IndiceEstatistico(),
        "logistica": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "floresta": lambda: RandomForestClassifier(
            n_estimators=config.arvores, max_depth=8, min_samples_leaf=40,
            n_jobs=-1, random_state=semente),
        "gradiente": lambda: HistGradientBoostingClassifier(
            max_iter=config.iteracoes, learning_rate=0.05, max_depth=4, random_state=semente),
        "rede_neural": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
            max_iter=200, random_state=semente)),
    }
    ok, _ = _xgboost_disponivel()
    if ok:
        from xgboost import XGBClassifier
        construtores["xgboost"] = lambda: XGBClassifier(
            n_estimators=config.iteracoes, max_depth=4, learning_rate=0.05, subsample=0.8,
            n_jobs=-1, verbosity=0, random_state=semente)
    return construtores


NOMES_MODELOS = {
    "estatistico": "Índice estatístico (sem aprendizado)",
    "logistica": "Regressão Logística",
    "floresta": "Random Forest",
    "gradiente": "Gradient Boosting (HistGradientBoosting)",
    "xgboost": "XGBoost",
    "rede_neural": "Rede neural experimental (MLP)",
}
ORDEM_MODELOS = list(NOMES_MODELOS)


def amostra_treino(indice: IndiceHistorico, inicio: int, fim: int, config: Configuracao,
                   rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Exemplos dos concursos [inicio, fim): positivos sorteados e negativos aleatórios."""
    ts, milhares, rotulos = [], [], []
    for t in range(inicio, fim):
        sorteadas = set((indice.numeros[t] % UNIVERSO_MILHAR).tolist())
        negativas = [m for m in rng.integers(0, UNIVERSO_MILHAR, config.negativos).tolist()
                     if m not in sorteadas]
        for m in indice.numeros[t] % UNIVERSO_MILHAR:
            ts.append(t); milhares.append(int(m)); rotulos.append(1)
        for m in negativas:
            ts.append(t); milhares.append(m); rotulos.append(0)
    X = calcular_atributos(indice, np.array(milhares), np.array(ts))
    return X, np.array(rotulos)


def treinar(indice: IndiceHistorico, fim: int, config: Configuracao, semente: int | None = None) -> dict:
    """Treina todos os modelos com os concursos [fim - janela, fim). Usa só o passado de `fim`."""
    semente = config.semente if semente is None else semente
    inicio = max(1, fim - config.janela_treino)  # o concurso 0 não tem passado
    if fim - inicio < config.minimo_treino:
        raise ValueError(
            f"Histórico insuficiente para treinar: {fim - inicio} concursos (mínimo {config.minimo_treino})."
        )
    rng = np.random.default_rng(semente)
    X, y = amostra_treino(indice, inicio, fim, config, rng)
    modelos, status = {}, {}
    for chave, construir in _construtores(config, semente).items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelos[chave] = construir().fit(X, y)
            status[chave] = "treinado"
        except Exception as exc:  # um modelo com falha não derruba os outros
            status[chave] = f"falhou: {type(exc).__name__}: {exc}"[:200]
    ok, versao = _xgboost_disponivel()
    if not ok:
        status["xgboost"] = versao
    return {
        "modelos": modelos, "status": status, "colunas": list(X.columns),
        "treino": {"inicio_indice": inicio, "fim_indice": fim, "exemplos": int(len(y)),
                   "positivos": int(y.sum()), "semente": int(semente),
                   "primeiro_concurso": int(indice.concursos[inicio]),
                   "ultimo_concurso": int(indice.concursos[fim - 1])},
    }


def notas(pacote: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Nota bruta de cada modelo para cada linha de X."""
    saida = {}
    for chave, modelo in pacote["modelos"].items():
        if hasattr(modelo, "nota"):
            saida[chave] = modelo.nota(X)
        else:
            saida[chave] = modelo.predict_proba(X[pacote["colunas"]])[:, 1]
    return saida


def percentil(valores: np.ndarray) -> np.ndarray:
    """Posição relativa de 0 a 1 (1 = maior nota). Empates recebem a posição média."""
    return pd.Series(valores).rank(method="average", pct=True).to_numpy()


def _montar_tabela(brutas: dict[str, np.ndarray], milhares: np.ndarray) -> pd.DataFrame:
    tabela = pd.DataFrame({"milhar": milhares})
    for chave, valores in brutas.items():
        tabela[chave] = percentil(valores)
    chaves = list(brutas)
    media = tabela[chaves].mean(axis=1).to_numpy() if chaves else np.zeros(len(tabela))
    tabela["pontuacao"] = 100 * percentil(media)
    tabela["_media"] = media
    return tabela


def ranking_milhares(pacote: dict, indice: IndiceHistorico, t: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Avalia as 10.000 milhares no momento t (só passado) e combina os modelos.

    Devolve (tabela, atributos): a nota de cada modelo (percentil de 0 a 1), a
    pontuação do conjunto (0 a 100) e os indicadores, da maior para a menor nota.
    """
    milhares = np.arange(UNIVERSO_MILHAR)
    X = calcular_atributos(indice, milhares, t)
    tabela = _montar_tabela(notas(pacote, X), milhares)
    ordem = tabela.sort_values(["_media", "milhar"], ascending=[False, True]).index
    return tabela.loc[ordem].reset_index(drop=True), X.loc[ordem].reset_index(drop=True)


def ranking_em_lote(pacote: dict, indice: IndiceHistorico, ts: list[int]) -> list[pd.DataFrame]:
    """Mesmo que ranking_milhares para vários concursos, com uma predição por modelo (mais rápido).

    Cada linha usa só os concursos anteriores ao seu próprio t.
    """
    milhares = np.arange(UNIVERSO_MILHAR)
    X = calcular_atributos(indice, np.tile(milhares, len(ts)), np.repeat(np.asarray(ts), UNIVERSO_MILHAR))
    brutas = notas(pacote, X)
    tabelas = []
    for i in range(len(ts)):
        fatia = slice(i * UNIVERSO_MILHAR, (i + 1) * UNIVERSO_MILHAR)
        tabela = _montar_tabela({k: v[fatia] for k, v in brutas.items()}, milhares)
        ordem = tabela.sort_values(["_media", "milhar"], ascending=[False, True]).index
        tabelas.append(tabela.loc[ordem].reset_index(drop=True))
    return tabelas


def importancias(pacote: dict) -> pd.DataFrame:
    """Peso de cada indicador nos modelos que expõem essa informação."""
    linhas = []
    colunas = pacote["colunas"]
    for chave, modelo in pacote["modelos"].items():
        valores = None
        if hasattr(modelo, "feature_importances_"):
            valores = np.asarray(modelo.feature_importances_, dtype=float)
        elif hasattr(modelo, "steps") and hasattr(modelo.steps[-1][1], "coef_"):
            valores = np.abs(modelo.steps[-1][1].coef_.ravel())
        if valores is None or valores.sum() == 0:
            continue
        valores = valores / valores.sum()
        linhas += [{"modelo": chave, "atributo": c, "peso": float(v)} for c, v in zip(colunas, valores)]
    return pd.DataFrame(linhas, columns=["modelo", "atributo", "peso"])


# ------------------------------------------------------------------ gravação

def pasta_modelos(caminho_banco: Path | str) -> Path:
    pasta = Path(caminho_banco).resolve().parent / "modelos"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def arquivo_modelo(caminho_banco, versao: tuple, config: Configuracao, semente: int) -> Path:
    marca = hashlib.sha256(f"{versao}|{config.hash()}|{semente}".encode()).hexdigest()[:16]
    return pasta_modelos(caminho_banco) / f"motor_{marca}.joblib"


def carregar_ou_treinar(caminho_banco, versao: tuple, indice: IndiceHistorico, config: Configuracao,
                        semente: int | None = None) -> tuple[dict, bool]:
    """Usa os modelos gravados para esta versão do histórico; treina só se faltar."""
    semente = config.semente if semente is None else semente
    arquivo = arquivo_modelo(caminho_banco, versao, config, semente)
    if arquivo.exists():
        try:
            pacote = joblib.load(arquivo)
            if pacote.get("versao") == list(versao) and pacote.get("total") == indice.total:
                return pacote, False
        except Exception:
            pass  # arquivo corrompido ou de outra versão: treina de novo
    pacote = treinar(indice, indice.total, config, semente)
    pacote["versao"] = list(versao)
    pacote["total"] = indice.total
    # Grava num arquivo temporário e renomeia: se a execução for interrompida no meio
    # (ex.: novo clique no painel), não fica um arquivo de modelo pela metade.
    temporario = arquivo.with_name(f"{arquivo.stem}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        joblib.dump(pacote, temporario, compress=3)
        os.replace(temporario, arquivo)
    finally:
        if temporario.exists():
            temporario.unlink()
    return pacote, True
