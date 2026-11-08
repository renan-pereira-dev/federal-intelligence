
# CONTEXTO
O Federal Intelligence é uma aplicação Python/Streamlit para análise 
estatística da Loteria Federal, com histórico de resultados, geração 
de candidatos, Machine Learning e backtesting. O projeto já está 
implementado e em evolução. A prioridade é melhorar o sistema 
existente, nunca reconstruí-lo desnecessariamente.

# ARQUITETURA
app.py: interface principal.
src/database.py: banco SQLite.
src/statistics.py: análises estatísticas.
src/preditivo/: motor de Machine Learning e previsão.
tests/: testes automatizados.
requirements.txt: dependências.
iniciar.bat: inicialização no Windows.
Tecnologias: Python, Streamlit, SQLite, Pandas, NumPy, 
Plotly, scikit-learn, XGBoost e SciPy.

# ESTADO DO PROJETO
Última auditoria registrada:

6.072 concursos.
30.360 números premiados.
Histórico desde 1962.
43 lacunas documentadas.
203 testes aprovados.

O motor utiliza Random Forest, Gradient Boosting, Regressão Logística, 
XGBoost, MLP e Ensemble. A validação utiliza Walk-Forward Backtesting. 
Até a última avaliação, nenhum modelo demonstrou vantagem estatística 
sobre o acaso. Confira sempre o estado atual antes de assumir que 
esses números continuam válidos.

# REGRAS OBRIGATÓRIAS
Inspecionar o código antes de modificar qualquer arquivo.
Apresentar um plano breve antes de implementar mudanças.
Preservar o banco SQLite definitivo.
Fazer backup antes de alterações estruturais.
Nunca apagar previsões ou candidatos existentes.
Nunca inventar ou corrigir automaticamente resultados históricos.
Impedir vazamento de dados futuros no treinamento.
Não confundir pontuação do modelo com probabilidade matemática.
Manter ativa a trava de validação estatística.
Preservar funcionalidades e cache existentes.
Não encerrar processos indiscriminadamente.
Executar os testes após cada alteração relevante.

# MOTOR PREDITIVO
Os modelos trabalham com atributos históricos e devem respeitar 
rigorosamente a ordem cronológica.Toda melhoria precisa ser comparada 
com uma linha de base aleatória. Não declarar superioridade sem 
evidência estatística independente.

Se não houver vantagem comprovada, manter o aviso:

"Sem vantagem comprovada sobre o acaso."

# EVOLUÇÃO
Prioridades:

Estabilidade do iniciar.bat.
Aperfeiçoamento do motor preditivo.
Avaliação contínua do backtesting.
Melhoria dos relatórios e gráficos.
Otimização de desempenho.
Histórico e rastreabilidade das previsões.
DIRETRIZ FINAL
Trabalhar incrementalmente, preservar os dados e manter a confiabilidade
estatística. Antes de concluir qualquer implementação, executar os testes,
verificar a integridade do banco e confirmar que o painel abre corretamente.
