"""Conferência de candidatos com os cinco números premiados.

Regras usadas (Manual de Produtos v23, itens 5.4.3 e 5.4.4):

- Prêmio principal: o número é igual a um dos cinco números sorteados.
- "Bilhetes cujos números contenham a milhar, a centena e a dezena de qualquer
  um dos números sorteados nos 05 (cinco) prêmios principais".
- "Bilhetes cujos números correspondam à aproximação imediatamente anterior e
  posterior ao número sorteado para o 1.º prêmio".
- "Bilhetes cujos números contenham a dezena final idêntica a umas das 03 (três)
  dezenas anteriores ou das 03 (três) dezenas posteriores à dezena do número
  sorteado para o 1.º prêmio, excetuando-se os premiados pela aproximação
  anterior e posterior".
- "Bilhetes cujos números contenham a unidade do 1.º prêmio".

Cuidados:
- Só o acerto exato é classificado como prêmio principal.
- Milhar, centena, dezena, aproximação, dezenas finais e unidade são mostradas
  como COINCIDÊNCIAS compatíveis com as faixas derivadas do Manual. Os valores,
  e se as faixas se acumulam (item 5.4.9), dependem do plano de sorteio
  impresso no verso de cada bilhete, que não está disponível aqui.
- Casos que o Manual não define (aproximação além do primeiro ou do último
  bilhete; dezenas finais que passariam de 99 para 00) não são avaliados.
- Um candidato gerado não é um bilhete comprado; a conferência é só educativa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.probability import DIGITOS_BILHETE, formatar_numero, numero_no_universo, obter_extracao

TERMINACOES = (
    (4, "milhar"),
    (3, "centena"),
    (2, "dezena"),
)

ROTULOS = {
    "exato": "Acerto exato (prêmio principal)",
    "milhar": "Coincidência de milhar (4 últimos dígitos)",
    "centena": "Coincidência de centena (3 últimos dígitos)",
    "dezena": "Coincidência de dezena (2 últimos dígitos)",
    "aproximacao": "Coincidência de aproximação do 1º prêmio",
    "dezenas_finais": "Coincidência de dezena final vizinha do 1º prêmio",
    "unidade": "Coincidência de unidade do 1º prêmio",
}


@dataclass
class ResultadoConferencia:
    candidato: str
    extracao: str
    premio_principal: list[int] = field(default_factory=list)
    terminacoes: dict[int, str] = field(default_factory=dict)
    aproximacao: str | None = None
    dezena_final_vizinha: bool = False
    unidade_primeiro: bool = False
    nao_avaliado: list[str] = field(default_factory=list)
    observacoes: list[str] = field(default_factory=list)

    @property
    def acertou_principal(self) -> bool:
        return bool(self.premio_principal)

    @property
    def tem_coincidencia(self) -> bool:
        return bool(
            self.terminacoes or self.aproximacao or self.dezena_final_vizinha or self.unidade_primeiro
        )

    def resumo(self) -> str:
        if self.acertou_principal:
            posicoes = ", ".join(f"{p}º" for p in self.premio_principal)
            return f"Acerto exato do {posicoes} prêmio"
        if self.tem_coincidencia:
            return "Coincidências parciais (não é prêmio principal)"
        return "Nenhuma coincidência"

    def itens(self) -> list[str]:
        """Lista legível de tudo que foi encontrado."""
        linhas = []
        for posicao in self.premio_principal:
            linhas.append(f"{ROTULOS['exato']}: {posicao}º prêmio")
        for posicao, tipo in sorted(self.terminacoes.items()):
            linhas.append(f"{ROTULOS[tipo]} com o {posicao}º prêmio")
        if self.aproximacao:
            linhas.append(f"{ROTULOS['aproximacao']} ({self.aproximacao})")
        if self.dezena_final_vizinha:
            linhas.append(ROTULOS["dezenas_finais"])
        if self.unidade_primeiro:
            linhas.append(ROTULOS["unidade"])
        return linhas

    def para_dict(self) -> dict:
        dados = asdict(self)
        dados["terminacoes"] = {str(k): v for k, v in self.terminacoes.items()}
        dados["resumo"] = self.resumo()
        dados["itens"] = self.itens()
        return dados


def _validar_numero(numero: str, rotulo: str) -> str:
    texto = str(numero).strip()
    if len(texto) != DIGITOS_BILHETE or not texto.isdigit():
        raise ValueError(f"{rotulo} deve ter exatamente 5 dígitos: {numero!r}")
    if int(texto) == 0:
        raise ValueError(f"{rotulo}: 00000 não é um bilhete válido.")
    return texto


def conferir(candidato: str, premios: list[str], chave_extracao: str) -> ResultadoConferencia:
    """Compara um candidato com os cinco números premiados de uma extração."""
    candidato = _validar_numero(candidato, "Candidato")
    if len(premios) != 5:
        raise ValueError("Informe exatamente cinco números premiados.")
    premios = [_validar_numero(p, f"{i}º prêmio") for i, p in enumerate(premios, 1)]
    extracao = obter_extracao(chave_extracao)
    resultado = ResultadoConferencia(candidato=candidato, extracao=extracao.chave)

    if not numero_no_universo(candidato, extracao.chave):
        resultado.observacoes.append(
            f"O candidato está fora do universo da extração {extracao.nome} ({extracao.faixa_texto})."
        )
    if extracao.series > 1:
        resultado.observacoes.append(
            "Esta extração tem mais de uma série. Um prêmio real também depende da série do "
            "bilhete, que o candidato não possui."
        )

    # 1) Prêmio principal e terminações, para cada um dos cinco prêmios.
    for posicao, premio in enumerate(premios, start=1):
        if candidato == premio:
            resultado.premio_principal.append(posicao)
            continue
        for tamanho, tipo in TERMINACOES:
            if candidato[-tamanho:] == premio[-tamanho:]:
                resultado.terminacoes[posicao] = tipo  # guarda só a mais longa
                break

    # 2) Regras derivadas que dependem apenas do 1º prêmio.
    primeiro = int(premios[0])
    valor = int(candidato)
    anterior, posterior = primeiro - 1, primeiro + 1
    if anterior < extracao.numero_minimo:
        resultado.nao_avaliado.append(
            "Aproximação anterior: o 1º prêmio é o primeiro bilhete da faixa; o Manual não "
            "define este caso."
        )
    elif valor == anterior:
        resultado.aproximacao = f"anterior, {formatar_numero(anterior)}"
    if posterior > extracao.numero_maximo:
        resultado.nao_avaliado.append(
            "Aproximação posterior: o 1º prêmio é o último bilhete da faixa; o Manual não "
            "define este caso."
        )
    elif valor == posterior:
        resultado.aproximacao = f"posterior, {formatar_numero(posterior)}"

    dezena_primeiro = primeiro % 100
    dezena_candidato = valor % 100
    vizinhas = [dezena_primeiro + d for d in (-3, -2, -1, 1, 2, 3)]
    if any(v < 0 or v > 99 for v in vizinhas):
        resultado.nao_avaliado.append(
            "Dezenas finais: a dezena do 1º prêmio está perto de 00 ou 99, e o Manual não diz "
            "se a contagem continua do outro lado. Só as dezenas entre 00 e 99 foram avaliadas."
        )
    vizinhas_validas = {v for v in vizinhas if 0 <= v <= 99}
    if dezena_candidato in vizinhas_validas and resultado.aproximacao is None:
        resultado.dezena_final_vizinha = True

    resultado.unidade_primeiro = (
        candidato[-1] == premios[0][-1] and 1 not in resultado.premio_principal
    )
    return resultado


def conferir_lista(candidatos: list[str], premios: list[str], chave_extracao: str) -> list[ResultadoConferencia]:
    return [conferir(c, premios, chave_extracao) for c in candidatos]
