FEDERAL INTELLIGENCE
Laboratório experimental de análise estatística da Loteria Federal

Versão: Beta 0.1 — evolução experimental
Tecnologia: Python, Streamlit, SQLite e Machine Learning
Execução: Aplicação local para Windows

1. Sobre o projeto

O Federal Intelligence é uma aplicação desenvolvida em Python para análise estatística dos resultados históricos da Loteria Federal brasileira.

O sistema combina análise de dados, estatística, geração de números candidatos e modelos experimentais de Machine Learning.

Seu objetivo é investigar distribuições, frequências, padrões históricos e avaliar se modelos computacionais conseguem identificar algum sinal estatístico útil.

A aplicação possui finalidade educacional e experimental. Não garante previsões nem ganhos financeiros.

2. Funcionalidades
Gerador de candidatos
Geração aleatória de números.
Seleção do tipo de extração.
Geração de diferentes quantidades de candidatos.
Preservação de zeros à esquerda.
Ausência de duplicidade na mesma lista.
Apresentação das probabilidades matemáticas.
Histórico estatístico
Frequência dos algarismos.
Frequência por posição.
Distribuição de terminações.
Repetições.
Distribuições de dezenas e centenas.
Análise por décadas e épocas.
Gráficos estatísticos.
Taxa histórica
Análise de coincidências.
Testes de uniformidade.
Testes qui-quadrado.
Comparação entre frequências observadas e esperadas.
Previsão do próximo sorteio
Número principal do ranking.
Nove alternativas.
Pontuação individual.
Combinação de modelos de Machine Learning.
Indicadores históricos.
Backtesting.
Histórico persistente das previsões.
Conferência de resultados
Registro dos resultados oficiais.
Comparação com candidatos gerados.
Identificação de coincidências exatas e parciais.
Acompanhamento do desempenho das previsões.
3. Base histórica

Última conferência registrada:

Indicador	Quantidade
Concursos	6.072
Números premiados	30.360
Primeiro concurso	1
Último concurso	6115
Período	1962–2026
Lacunas documentadas	43
Duplicidades	0

Os números são armazenados como texto para preservar os cinco dígitos.

Existem lacunas e inconsistências de datas provenientes da fonte original. Elas estão documentadas e não devem ser interpretadas como dados corrigidos.

4. Tecnologias
Tecnologia	Finalidade
Python	Linguagem principal
Streamlit	Interface
SQLite	Banco de dados
Pandas	Manipulação de dados
NumPy	Cálculos numéricos
Plotly	Gráficos
scikit-learn	Machine Learning
XGBoost	Modelo experimental
SciPy	Estatística
Pytest	Testes
5. Arquitetura
text
federal-intelligence/
│
├── app.py
├── iniciar.bat
├── requirements.txt
├── README.md
├── CLAUDE.md
│
├── src/
│   ├── database.py
│   ├── statistics.py
│   ├── oficial.py
│   ├── oficial_interface.py
│   ├── monitoramento.py
│   ├── monitoramento_interface.py
│   └── preditivo/
│       ├── atributos.py
│       ├── modelos.py
│       ├── backtest.py
│       ├── calendario.py
│       ├── armazenamento.py
│       ├── motor.py
│       ├── experimentos.py
│       ├── estrategias.py
│       ├── laboratorio.py
│       ├── auditoria_final.py
│       └── interface.py
│
├── tests/
├── data/
├── models/
└── backup/

Principais módulos
app.py: interface e navegação.
database.py: acesso ao SQLite.
statistics.py: cálculos estatísticos.
oficial.py: importação do resultado oficial pelo serviço da CAIXA.
oficial_interface.py: tela de busca, prévia e confirmação do resultado oficial.
monitoramento.py: testes de justiça do sorteio por bloco de concursos.
monitoramento_interface.py: tela "Monitoramento do sorteio".
atributos.py: engenharia de atributos.
modelos.py: treinamento dos modelos.
backtest.py: validação cronológica.
calendario.py: calendário dos concursos.
armazenamento.py: persistência das previsões.
motor.py: execução e combinação dos modelos.
experimentos.py: auditoria e experimentos controlados (não altera o motor).
estrategias.py: laboratório de estratégias de seleção dos 10 candidatos.
laboratorio.py: tela "Laboratório de estratégias".
auditoria_final.py: auditoria final de desempenho contra o acaso.
interface.py: apresentação dos resultados.
6. Motor de Machine Learning

Modelos implementados:

Random Forest.
Gradient Boosting.
Regressão Logística.
XGBoost.
MLP.
Ensemble.

O treinamento utiliza atributos históricos, como frequências, terminações, posições e intervalos de ocorrência.

Os atributos de cada previsão devem ser calculados exclusivamente com informações disponíveis até aquele momento.

O ensemble combina os rankings dos modelos e apresenta um candidato principal e nove alternativas.

A pontuação é uma classificação interna, não uma probabilidade de sorteio.

7. Backtesting

A validação utiliza Walk-Forward Backtesting.

O treinamento ocorre com concursos anteriores, e o modelo é avaliado em concursos posteriores.

Indicadores avaliados:

Acertos exatos.
Coincidências de milhar.
Coincidências de centena.
Coincidências de dezena.
Precision@10.
Recall@10, quando aplicável.
Comparação com seleção aleatória.
Resultados por período e modelo.

Na última avaliação registrada, nenhum modelo demonstrou vantagem estatística sobre o acaso.

O sistema mantém a proteção:

Sem vantagem comprovada sobre o acaso.

Auditoria de 08/11/2026 (experimentos.py):

Ajuste da configuração: concursos 4598–5097.
Teste final fora da amostra: concursos 5098–6115.
Comparadas 8 variantes (janelas, atributos, parâmetros) e 63 combinações de modelos.
Referências: seleção aleatória, frequência histórica, frequência recente, intervalo de ausência e estrutura do universo.
Resultado: nenhuma configuração aprovada; a configuração padrão foi mantida.
Mudanças aprovadas e aplicadas: (1) a trava agora exige também superar a estrutura do universo
(milhares com 10 ou 9 bilhetes válidos), e não só ficar abaixo de 0,5; (2) removidos os atributos
posicao_5_500 e posicao_5_total, idênticos a freq_1_500 e freq_1_total. Backtest refeito: sem vantagem.
Relatório: data/experimentos/auditoria_20261108-001746.md

Laboratório de estratégias de 08/11/2026 (estrategias.py):

92 estratégias: 13 rankings (cada modelo, conjunto igual e ponderado, frequência histórica e recente, combinação estatística, ausência, controle estrutural) × 7 seleções (tradicional, diversificada por unidade, dezena e centena, ranking + diversidade com λ 0,25/0,5/0,75), mais a seleção aleatória.
Desenvolvimento: concursos 4098–5097. Amostra final: 5098–6115. Pré-registro de 3 candidatas feito só com o desenvolvimento.
Resultado: nenhuma estratégia aprovada; o motor padrão continua em uso.
Diversificar aumenta as coincidências parciais só pela cobertura de terminações, igualando a seleção aleatória; não há ganho além do esperado pela cobertura.
Relatório: data/experimentos/estrategias_20261108-012614.md
Para rodar de novo (cerca de 20 minutos): .venv\Scripts\python.exe -m src.preditivo.estrategias
Ativar uma estratégia só é possível se ela for aprovada; a tela permite voltar à configuração anterior.

Auditoria final de 08/11/2026 (auditoria_final.py):

Backtest cronológico do motor padrão em 5.072 concursos (1004–6115). Análise principal nos 1.572 concursos nunca usados antes (1004–2577).
Conclusão: não há vantagem sobre o acaso. Nenhum resultado sobreviveu à correção de Holm nem se repetiu no trecho secundário.
Meta de 70% com 10 candidatos: impossível no acerto exato, na milhar e na centena; na dezena, o acaso dá 41% e 70% exigiria pelo menos 22 números; na unidade, 100% sai só por cobertura, sem previsão.
Relatório: data/experimentos/auditoria_final_20261108-021052.md
Regra de "sinal promissor" (atualizada): o trecho inédito é dividido em duas janelas cronológicas, e o sinal precisa ter p < 0,05, na direção favorável, nas duas, cada uma analisada sozinha. Com essa regra, o resultado real é "sem vantagem": os sinais da janela 1 (1004–1789) não se repetem na janela 2 (1790–2577).
Relatório com a nova regra: data/experimentos/auditoria_final_20261108-023053_replicacao.md

8. Probabilidades matemáticas

O sistema separa:

Probabilidade matemática oficial.
Frequência histórica.
Pontuação do modelo.
Desempenho no backtesting.

Em um sorteio justo, todos os números elegíveis possuem a mesma probabilidade teórica.

Frequências históricas não alteram a probabilidade de um próximo sorteio independente.

Fonte oficial:

https://loterias.caixa.gov.br/Paginas/Federal.aspx

9. Instalação e execução
Requisitos
Windows 10 ou superior.
Python compatível com as dependências instaladas.
Ambiente virtual Python.
Inicialização pelo Windows
Abra a pasta do projeto.
Localize iniciar.bat.
Dê dois cliques.
Aguarde o Streamlit iniciar.
Acesse o endereço local apresentado.
Inicialização manual

Criar ambiente virtual:

bash
python -m venv .venv


Ativar:

bash
.venv\Scripts\activate


Instalar dependências:

bash
pip install -r requirements.txt


Executar:

bash
streamlit run app.py

10. Resultado oficial automático

Nas telas Histórico e Previsão do próximo sorteio, o botão "Buscar resultados oficiais" consulta o serviço da CAIXA.
Antes de gravar, o sistema mostra a prévia: concursos novos, já gravados e conflitos.
A gravação só acontece depois de "Confirmar e gravar".
Nada é sobrescrito: se um concurso já gravado vier diferente, nada é gravado.
Depois de gravar, as previsões do concurso são conferidas automaticamente.
Sem internet, nada muda e o restante do aplicativo continua funcionando.
O serviço não informa o tipo de extração (QUARTOU, Domingo, ENRICOU); o campo fica vazio, como no restante do histórico.

11. Monitoramento do sorteio

A tela "Monitoramento do sorteio" confere se o sorteio continua uniforme e independente. Não prevê resultados.
O histórico é dividido em blocos de 250 concursos, contados a partir do mais recente.
Em cada bloco são feitos 6 testes: unidade, dezena, algarismos da 3ª e 4ª posição, unidade igual nos 5 prêmios e independência entre concursos seguidos, com correção de Holm.
Normal: nada significativo no bloco mais recente.
Atenção: algum teste significativo no bloco mais recente, sem repetição (acontece em cerca de 5% dos blocos por acaso).
Alerta: o mesmo teste significativo em dois blocos seguidos.
As posições 1 e 2 ficam de fora porque dependem da quantidade de bilhetes de cada época.
Situação em 08/11/2026: normal; 24 blocos, 9 de 144 testes com p < 0,05 (7,2 esperados por acaso), nenhum alerta em todo o histórico.

12. Banco de dados

O projeto utiliza SQLite para armazenar histórico, candidatos, importações e previsões.

Tabelas principais do módulo preditivo:

predictions
prediction_items
prediction_backtests

O banco definitivo deve ser preservado.

Alterações estruturais devem ser acompanhadas de backup e verificação de integridade.

13. Testes

O projeto utiliza Pytest.

Última validação registrada: 297 testes aprovados.

Para executar:

bash
pytest -v


Os testes cobrem geração, estatística, modelos, backtesting, persistência, integridade e interface.

14. Ética e limitações

O Federal Intelligence é um laboratório experimental.

Não promete ganhos.
Não apresenta previsões como certezas.
Não confunde pontuação com probabilidade.
Não manipula resultados de backtesting.
Não inventa dados históricos.
Não declara vantagem sem evidência estatística.
Não considera coincidências parciais como prêmio principal.
15. Próximas implementações
Melhorar a estabilidade do inicializador.
Monitorar o desempenho real das previsões.
Aperfeiçoar os relatórios estatísticos.
Avaliar novos atributos e estratégias.
Melhorar o histórico de previsões.
Otimizar treinamento e cache.
Ampliar os testes de validação fora da amostra.

Novas estratégias só devem ser incorporadas após avaliação estatística independente.

FEDERAL INTELLIGENCE

Análise de dados, estatística e Machine Learning aplicados ao estudo dos resultados históricos da Loteria Federal.

## 🚀 Acesso ao Aplicativo
Você pode testar a versão online do Federal Intelligence clicando no link abaixo:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://federal-intelligence-uajnhukcptexo8w5rnfu4v.streamlit.app/)
