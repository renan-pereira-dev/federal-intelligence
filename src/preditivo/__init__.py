"""Motor experimental de previsão (independente do restante da aplicação).

Módulos:
- atributos: indicadores calculados só com concursos ANTERIORES ao analisado.
- modelos: modelos estatísticos e de aprendizado de máquina, treino e gravação.
- backtest: avaliação walk-forward (ordem cronológica) contra o acaso.
- calendario: identificação do próximo concurso e da data (oficial ou estimada).
- armazenamento: tabelas próprias no SQLite (previsões, itens e backtests).
- motor: orquestra tudo quando o usuário clica em GERAR PREVISÃO.

A pontuação preditiva NÃO é probabilidade. Em um sorteio justo, todos os
números elegíveis têm a mesma chance; o backtest mede se o modelo teve alguma
vantagem real sobre o acaso no passado.
"""
