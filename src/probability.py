"""Regras oficiais das extrações e probabilidades da Loteria Federal.

Todos os valores abaixo foram copiados de documentos oficiais da CAIXA,
consultados em 07/11/2026. Nada aqui é estimado a partir de histórico.

Fontes:
- Manual de Produtos Loterias CAIXA, versão 23 (Circular CAIXA n.º 1.128,
  de 14/10/2026), itens 5.3.4 a 5.3.7 (bilhetes por extração),
  5.4.4 (prêmios derivados) e 5.5.1 / 5.5.2 (chance de acerto).
- Regulamento Loterias - Loteria Federal (Regras dos sorteios), itens 2 a 4
  e 11.
- Página oficial: https://loterias.caixa.gov.br/Paginas/Federal.aspx
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

FONTE_PAGINA = "https://loterias.caixa.gov.br/Paginas/Federal.aspx"
FONTE_MANUAL = (
    "Manual de Produtos Loterias CAIXA, versão 23 "
    "(Circular CAIXA n.º 1.128, de 14/10/2026)"
)
FONTE_MANUAL_URL = (
    "https://www.caixa.gov.br/Downloads/loterias-manual-de-produtos/"
    "manual-de-produtos-v23.pdf"
)
FONTE_REGULAMENTO = "Regulamento Loterias - Loteria Federal (Regras dos sorteios)"
FONTE_REGULAMENTO_URL = (
    "https://www.caixa.gov.br/downloads/loterias-comunicados-importantes/"
    "Regras_dos_sorteios.pdf"
)
DATA_CONSULTA = "07/11/2026"

DIGITOS_BILHETE = 5

# Regulamento, item 4: "Sendo sorteado um número superior à quantidade de
# bilhetes prevista para a extração ou 0 em todos os globos, procede-se novo
# sorteio." Logo, o número 00000 nunca é válido e o maior número válido é a
# própria quantidade de bilhetes emitidos.
NUMERO_MINIMO = 1
REGRA_UNIVERSO = (
    "Regulamento, item 4: número superior à quantidade de bilhetes prevista "
    "para a extração, ou 0 em todos os globos, gera novo sorteio. "
    "Universo válido: de 00001 até a quantidade de bilhetes emitidos."
)

# Manual 5.5.2: chance oficial de ganhar qualquer prêmio derivado. Este valor é
# exibido como informado pela CAIXA, nunca recalculado pela aplicação.
CHANCE_DERIVADOS_OFICIAL = "1 em 4,78"
CHANCE_DERIVADOS_TEXTO = (
    "O apostador concorre também a prêmios derivados dos principais; neste "
    "caso, a chance de ser premiado em qualquer uma das modalidades de "
    "extração é de 1 em 4,78 (Manual, item 5.5.2). Esse número é da CAIXA e "
    "não é somado nem combinado com a chance do prêmio principal."
)


@dataclass(frozen=True)
class Extracao:
    """Uma modalidade de extração e suas regras oficiais."""

    chave: str
    nome: str
    descricao: str
    bilhetes_por_serie: int
    series: int
    chance_um_em: int
    chance_por_serie: bool
    fonte_quantidade: str
    fonte_chance: str
    habilitada: bool = True

    @property
    def numero_minimo(self) -> int:
        return NUMERO_MINIMO

    @property
    def numero_maximo(self) -> int:
        return self.bilhetes_por_serie

    @property
    def tamanho_universo(self) -> int:
        return self.numero_maximo - self.numero_minimo + 1

    @property
    def faixa_texto(self) -> str:
        return f"{formatar_numero(self.numero_minimo)} a {formatar_numero(self.numero_maximo)}"

    @property
    def chance_texto(self) -> str:
        texto = f"1 em {formatar_milhar(self.chance_um_em)}"
        return texto + " (por série)" if self.chance_por_serie else texto

    @property
    def probabilidade(self) -> Fraction:
        """Probabilidade oficial de um bilhete acertar um prêmio principal."""
        return Fraction(1, self.chance_um_em)


EXTRACOES: dict[str, Extracao] = {
    "quartou": Extracao(
        chave="quartou",
        nome="QUARTOU!",
        descricao="Extração regular de quarta-feira.",
        bilhetes_por_serie=92_000,
        series=1,
        chance_um_em=92_000,
        chance_por_serie=False,
        fonte_quantidade="Manual v23, item 5.3.4: série única de 92 mil bilhetes.",
        fonte_chance="Manual v23, item 5.5.1: QUARTOU! 1 em 92.000.",
    ),
    "domingo": Extracao(
        chave="domingo",
        nome="Domingo",
        descricao="Extração regular de domingo (sem nome comercial próprio).",
        bilhetes_por_serie=96_000,
        series=1,
        chance_um_em=96_000,
        chance_por_serie=False,
        fonte_quantidade="Manual v23, item 5.3.5: série única de 96 mil bilhetes.",
        fonte_chance="Manual v23, item 5.5.1: DOMINGO 1 em 96.000.",
    ),
    "enricou": Extracao(
        chave="enricou",
        nome="ENRICOU!",
        descricao="Extração especial mensal (Milionária Federal).",
        bilhetes_por_serie=84_000,
        series=1,
        chance_um_em=84_000,
        chance_por_serie=False,
        fonte_quantidade="Manual v23, item 5.3.6: série única de 84 mil bilhetes.",
        fonte_chance="Manual v23, item 5.5.1: ENRICOU! 1 em 84.000.",
    ),
    "natal": Extracao(
        chave="natal",
        nome="Especial de Natal",
        descricao="Extração especial anual, em dezembro, com duas séries.",
        bilhetes_por_serie=85_000,
        series=2,
        chance_um_em=85_000,
        chance_por_serie=True,
        fonte_quantidade="Manual v23, item 5.3.7: duas séries de 85 mil bilhetes.",
        fonte_chance="Manual v23, item 5.5.1: ESPECIAL DE NATAL 85.000 (por série).",
    ),
}


def formatar_numero(numero: int) -> str:
    """Formata um número de bilhete com 5 dígitos, mantendo zeros à esquerda."""
    if numero < 0 or numero >= 10**DIGITOS_BILHETE:
        raise ValueError(f"Número fora do formato de {DIGITOS_BILHETE} dígitos: {numero}")
    return f"{numero:0{DIGITOS_BILHETE}d}"


def formatar_milhar(valor: int) -> str:
    """Formata inteiros no padrão brasileiro: 92000 -> '92.000'."""
    return f"{valor:,}".replace(",", ".")


def obter_extracao(chave: str) -> Extracao:
    try:
        return EXTRACOES[chave]
    except KeyError as exc:
        validas = ", ".join(EXTRACOES)
        raise ValueError(f"Extração desconhecida: {chave!r}. Use uma de: {validas}.") from exc


def extracoes_habilitadas() -> list[Extracao]:
    return [e for e in EXTRACOES.values() if e.habilitada]


def numero_no_universo(numero: str, chave: str) -> bool:
    """Verifica se um número (texto de 5 dígitos) pertence ao universo da extração."""
    if not (isinstance(numero, str) and len(numero) == DIGITOS_BILHETE and numero.isdigit()):
        return False
    extracao = obter_extracao(chave)
    return extracao.numero_minimo <= int(numero) <= extracao.numero_maximo


# ------------------------------------------------------------- justificativas
#
# Por que cada número válido tem a mesma chance (Regulamento, itens 2 a 4):
# cada globo sorteia um algarismo com bolas iguais; combinações acima da
# quantidade de bilhetes, ou 00000, são descartadas e sorteadas de novo.
# Descartar e repetir mantém todas as combinações válidas com a mesma chance
# (amostragem por rejeição). Logo, P(um número) = 1 / N.
#
# Por que os cinco prêmios são independentes (Regulamento, itens 11 e 12):
# as bolas voltam aos globos e "é válida a repetição de número sorteado na
# mesma extração". Cada prêmio é um sorteio novo, com a mesma distribuição.

JUSTIFICATIVA_PRINCIPAL = (
    "Por quê: cada globo sorteia um algarismo, e combinações acima da quantidade de "
    "bilhetes, ou 00000, são descartadas e sorteadas de novo (Regulamento, itens 2 a 4). "
    "Descartar e repetir não favorece nenhum número válido, então cada um dos N bilhetes "
    "tem chance 1/N. A CAIXA publica exatamente esse valor (Manual, item 5.5.1)."
)
JUSTIFICATIVA_INDEPENDENCIA = (
    "As bolas voltam aos globos e o mesmo número pode sair em mais de um prêmio "
    "(Regulamento, itens 11 e 12). Por isso, cada um dos cinco prêmios é um sorteio "
    "independente, e a chance de coincidir com pelo menos um deles é 1 - (1 - p)^5."
)

TERMINACOES_NOMES = {1: "unidade", 2: "dezena", 3: "centena", 4: "milhar"}


def contagem_terminacao(maximo: int, digitos: int, terminacao: int) -> int:
    """Quantos números de 1 a `maximo` terminam em `terminacao` (com `digitos` dígitos)."""
    modulo = 10**digitos
    if not 0 <= terminacao < modulo:
        raise ValueError("terminação fora do intervalo")
    if terminacao == 0:
        return maximo // modulo
    return maximo // modulo + (1 if terminacao <= maximo % modulo else 0)


def probabilidade_terminacao(maximo: int, digitos: int, terminacao: int | None = None) -> Fraction | None:
    """Chance exata de um prêmio terminar numa terminação, num universo de 1 a `maximo`.

    Sem `terminacao`, devolve a chance comum a todas as terminações, ou None se
    ela variar (por exemplo, milhar com 92.000 bilhetes).
    """
    modulo = 10**digitos
    if terminacao is not None:
        return Fraction(contagem_terminacao(maximo, digitos, terminacao), maximo)
    if maximo % modulo == 0:
        return Fraction(1, modulo)
    return None


def probabilidade_algum_premio(p: Fraction, premios: int = 5) -> Fraction:
    """Chance de coincidir com pelo menos um de `premios` sorteios independentes."""
    return 1 - (1 - p) ** premios


def justificativa_terminacao(maximo: int, digitos: int) -> str:
    """Explica a chance de uma terminação para o universo de 1 a `maximo`."""
    nome = TERMINACOES_NOMES[digitos]
    modulo = 10**digitos
    p = probabilidade_terminacao(maximo, digitos)
    if p is not None:
        return (
            f"De 00001 a {formatar_numero(maximo)} há {formatar_milhar(maximo)} números. Como "
            f"{formatar_milhar(maximo)} é múltiplo de {formatar_milhar(modulo)}, cada {nome} final "
            f"aparece exatamente {formatar_milhar(maximo // modulo)} vezes. Logo, a chance de um "
            f"prêmio terminar numa {nome} específica é 1 em {formatar_milhar(modulo)}."
        )
    menor = Fraction(maximo // modulo, maximo)
    maior = Fraction(maximo // modulo + 1, maximo)
    return (
        f"De 00001 a {formatar_numero(maximo)}, {formatar_milhar(maximo)} não é múltiplo de "
        f"{formatar_milhar(modulo)}. Algumas terminações de {digitos} dígitos aparecem "
        f"{maximo // modulo} vezes e outras {maximo // modulo + 1}, então a chance da {nome} "
        f"varia entre {float(menor):.5%} e {float(maior):.5%}, conforme a terminação."
    )


def explicacao_probabilidade(chave: str) -> str:
    """Texto simples explicando a probabilidade oficial de uma extração."""
    e = obter_extracao(chave)
    texto = (
        f"Na extração {e.nome}, a CAIXA emite {formatar_milhar(e.bilhetes_por_serie)} "
        f"bilhetes por série, numerados de {e.faixa_texto}. A chance oficial de um "
        f"bilhete acertar um prêmio principal é de {e.chance_texto}. "
        "Todos os números dessa faixa têm exatamente a mesma chance teórica: "
        "o histórico não aumenta nem diminui essa chance."
    )
    if e.series > 1:
        texto += (
            f" Esta extração tem {e.series} séries; a chance vale dentro de cada série, "
            "e o número candidato não indica série."
        )
    return texto
