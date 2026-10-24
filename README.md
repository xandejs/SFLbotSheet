# 🤖 SFL Trading Bot: Inteligência Artificial Aplicada ao Mercado de Sunflower Land

> Descrição geral: sistema automatizado de inteligência artificial quantitativa para o ecossistema de Sunflower Land. O projeto combina modelagem preditiva, aprendizado por reforço, gestão de risco e integração com a API do marketplace para transformar dados de mercado em decisões automatizadas de compra, venda e manutenção de posição.

---

## 🧠 Visão Estratégica

O bot foi concebido para operar em um ambiente de mercado com ruído, liquidez variável, comportamento de rede e fricção de execução. Em vez de depender apenas de regras heurísticas, ele usa uma arquitetura híbrida em camadas:

- coleta e normalização de dados;
- engenharia de features de mercado;
- previsão de tendência e reversão;
- política de decisão com aprendizado por reforço;
- validação financeira antes da execução;
- controle de risco baseado em custo médio e alocação ponderada.

A ideia central é transformar sinais de curto prazo em decisões financeiramente consistentes, respeitando regras de custo médio, taxas reais e exposição máxima.

---

## 🏗️ Arquitetura Técnica

### 1. Motor de IA híbrido

O núcleo do sistema não depende apenas de médias móveis ou regras fixas. Ele aplica um pipeline de Machine Learning em múltiplas camadas:

- XGBoost para predição de curto e médio prazo sobre séries temporais, nível de momentum, volatilidade, volume e liquidez;
- aprendizado por reforço para definir ações em ambiente dinâmico (comprar, vender ou aguardar);
- análise de regime do mercado para distinguir contexto de tendência, consolidação e reversão;
- filtros estatísticos para reduzir ruído e evitar entradas impulsivas em condições frágeis.

Em termos práticos, o modelo tenta responder três perguntas em sequência:

1. O ativo está em tendência ou em regime de baixa qualidade?
2. Qual é a probabilidade de movimento favorável nos próximos intervalos?
3. Valeria a pena abrir/fechar posição considerando custo, risco e prêmio esperado?

### 2. Gestão financeira e custo médio

A lógica de saída é reforçada por regras quantitativas para manter consistência e robustez:

- cálculo do preço médio ponderado de entrada por item em carteira;
- margem mínima de ganho definida sobre o custo médio mais taxas nativas;
- validação da viabilidade da operação antes da execução;
- controle de exposição por ativo e por portfolio;
- proteção contra entradas em condições de baixa confiança ou liquidez insuficiente.

O objetivo é evitar que a decisão seja guiada apenas por preço spot ou impulso momentâneo. O bot opera com uma lógica de custo médio e prêmio real, em vez de assumir que qualquer tendência é automaticamente lucrativa.

### 3. Ingestão de dados e histórico

O sistema coleta e processa dados em múltiplas fontes e formatos:

- logs de preços do mercado;
- histórico do marketplace e transações;
- variação de liquidez e volume;
- eventos temporais e de regime;
- séries históricas alinhadas por timestamp.

Antes da modelagem, os dados passam por:

- limpeza e padronização;
- remoção de ruído e outliers relevantes;
- cálculo de indicadores estendidos;
- transformações logarítmicas e normalizações de escala;
- empacotamento em vetores de features para treino e inferência.

Isso reduz distortões causadas por volatilidade extrema e melhora a qualidade dos sinais gerados pelo modelo.

### 4. Integração com o ecossistema do marketplace

A arquitetura também considera a camada operacional de execução:

- leitura de dados via API externa e planilhas estruturadas;
- persistência em base de dados/células organizadas por item e período;
- rastreio de sinais em aberto;
- conferência de taxas, custos e margem mínima antes da execução;
- escrita atômica ou em lote para evitar inconsistência de estado.

Esse nível operacional é crucial, porque o ganho teórico do modelo só se traduz em benefício real quando a execução respeita custos reais de transação e latência do ambiente.

---

## 📊 Pipeline Operacional

| Etapa | Processo interno | Objetivo |
| --- | --- | --- |
| 1. Ingestão | Coleta de logs, histórico e dados de mercado | Alimentar a base com séries temporais consistentes |
| 2. Engenharia de Features | Indicadores, momentum, volatilidade, liquidez, regime | Extrair informação útil para decisão |
| 3. Predição | XGBoost + análise comparativa de contexto | Identificar pontos de virada e assimetria de preço |
| 4. Política de Decisão | RL + regras de custo médio | Definir buy / sell / hold com base em recompensa real |
| 5. Validação Financeira | Taxas, margem, risco e limites de exposição | Filtrar sinais antes da execução |
| 6. Execução | Operação e rastreio do evento | Transformar decisão em ação controlada |

---

## 🔧 Stack Técnico e Implementação

O projeto foi estruturado para permitir experimentação e evolução incremental do modelo:

- Python para prototipagem, dados, treino e orchestration;
- XGBoost como modelo principal de previsão;
- pandas/numpy para manipulação de séries temporais e engenharia de features;
- técnicas de RL para tomada de decisão em ambiente discreto;
- integração com planilhas / API do marketplace para persistência e execução;
- módulo experimental em C para benchmark de performance e validação de gargalos.

A parte em C foi desenvolvida como experimento de performance para verificar se a linguagem oferece ganho real em código crítico. O benchmark indicou que o ganho relevante aparece em loops manuais de indicadores; o gargalo principal continua sendo I/O de rede, leitura de planilhas e retrabalho redundante de treino e recalculação.

### Resultados do benchmark experimental (C vs Python)

| Etapa | C | Python | Resultado |
| --- | --- | --- | --- |
| Leitura + parse do CSV | ~0,179s | ~0,172s | empate |
| Cálculo de indicadores | ~0,043s | ~0,224s | C ~5x mais rápido |
| Treino + predição XGBoost | ~0,025s | ~0,030s | empate |

Conclusão prática: o ganho real em C aparece no cálculo manual de indicadores com janela deslizante. Em XGBoost, a diferença é pequena porque a biblioteca já é nativa e compilada em C++; o overhead fica em conversão de dados e orchestration. Em produção, a maior parte do tempo continua sendo consumo de API, sincronização e operação de rede.

---

## 🛡️ Estratégia de Risco

A governança financeira do bot é um diferencial importante:

- não entra em posições sem filtro de contexto;
- não aceita sinais de baixa confiança sem validação complementar;
- acompanha posição por custo médio e margem efetiva;
- evita operação em períodos de volatilidade extrema sem confirmação amplificada;
- prioriza consistência sobre frequência exagerada de trades.

Em outras palavras, a estratégia não busca “alavancar qualquer ruído” do mercado. Ela busca um conjunto de sinais que tenham maior probabilidade de sobreviver ao custo de execução e à incerteza do ambiente.

---

## 🎯 Objetivos Principais do Projeto

- maximização de retorno ajustado ao risco;
- mitigação de exposição em cenários voláteis;
- uso de IA para tomada de decisão quantitativa e automatizada;
- criação de um sistema replicável, verificável e monitorável;
- evolução contínua do modelo e da política de execução.

---

## 🔄 Roadmap e evolução

O projeto foi pensado para evoluir em etapas:

1. consolidar ingestão e normalização de dados;
2. aumentar qualidade das features de mercado;
3. reduzir retrabalho de treino e computação incremental;
4. implementar persistência de estado e Q-table;
5. melhorar a deduplicação de sinais e a validação de posições em aberto;
6. expandir o mecanismo de custo médio, FIFO e proteção de risco;
7. balancear precisão, custo operacional e confiabilidade do sistema.

A direção geral é manter a arquitetura modular, promovendo melhoria contínua sem virar uma dependência rígida de uma única regra ou um único modelo.

---

## 📌 Resumo Executivo

O SFL Trading Bot é um sistema de trading inteligente, orientado por dados e regras financeiras rigorosas, que combina:

- previsão estatística e machine learning;
- decisão por reforço;
- controle matemático de risco;
- execução financeira baseada em custo médio;
- integração com o ecossistema de Sunflower Land em escala operacional.

Assim, o projeto vai além de um simples “bot de compra e venda”: ele busca uma arquitetura automatizada, monitorável e financeiramente coerente para operar em um mercado de alta volatilidade e baixa previsibilidade.

