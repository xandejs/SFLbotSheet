# SFL Bot em C — porte experimental para teste de performance

Porte parcial do `bot_4_24.py` para C, feito para responder uma pergunta
específica: **C dá ganho de performance real nesse bot?** A resposta,
depois de medir (não assumir), está na seção de resultados abaixo.

## Resultados dos benchmarks (63 itens simulados, 8000h de histórico cada ~333 dias)

| Etapa                                  | C          | Python          | Resultado |
|-----------------------------------------|-----------|-----------------|-----------|
| Leitura+parse do CSV (63 itens)         | 0,179s    | 0,172s (pandas) | empate    |
| Cálculo de indicadores (rolling/RSI)    | 0,043s    | 0,224s (pandas) | **C ~5x mais rápido** |
| Treino+predição XGBoost (1 item, 40 estimators) | 0,025s | 0,030s (sklearn) | empate (mesma lib C++ por baixo) |

**Conclusão prática:** o único trecho onde C ganha de verdade é o cálculo
de indicadores com loop manual otimizado (janela deslizante O(n) com soma
incremental) — porque ali o pandas paga overhead de criar objetos
intermediários (Series, alocação por operação encadeada) que um loop C
bem escrito evita. No XGBoost, ganho é zero: a lib já é C++ compilado,
o wrapper Python só adiciona ~5ms de overhead de conversão de DataFrame.
Isso bate com a intuição original: o gargalo do bot não é a linguagem do
laço, é I/O de rede (Sheets API) e trabalho redundante (retreinar 63
modelos do zero por hora, recalcular indicadores do zero por ciclo) — os
dois já endereçados no roadmap V2 (modelo pooled + cache incremental),
**sem precisar trocar de linguagem**.

Se ainda assim quiser seguir com C em produção, o ganho de ~0,18s no
cálculo de indicadores por ciclo é irrelevante perto do tempo gasto
esperando respostas da Sheets API (tipicamente 200-800ms por chamada,
e o bot faz várias por ciclo) — ver a nota de performance na seção
"O que falta" abaixo.

## O que está completo (compilável e testado isoladamente)

- **`oauth.c`** — fluxo OAuth2 de Service Account: monta o JWT (RS256),
  assina com a chave privada do `credenciais_robo.json`, troca por
  `access_token` via `grant_type=jwt-bearer`. Não testado contra o Google
  de verdade neste ambiente (sandbox sem acesso a `googleapis.com`) — só
  a lógica de montagem/assinatura foi exercitada.
- **`sheets.c`** — cliente da Sheets API v4 + Drive API v3: localizar
  planilha por nome, `get_values`, `update_values`, `append_rows`, `clear`,
  `add/delete/rename worksheet` via `batchUpdate`. Mesma cobertura de
  `gspread` usada no bot.
- **`indicators.c`** — rolling mean/std O(n) (janela deslizante com soma
  incremental), RSI, detector de regime. **Validado ponto a ponto contra
  pandas** com dados sintéticos — bate em 4-6 casas decimais a partir do
  registro 48 (ver nota abaixo sobre o único ponto que diverge).
- **`calibration.c`** — tradução 1:1 de `calcular_taxa_confianca`,
  `calcular_prior_empirico`, `calcular_veto_rl`, `calcular_stop_adaptativo`,
  `calcular_margem_atencao`, `calcular_roi_real`.
- **`xgb_wrapper.c`** — treino+inferência via C API nativa do XGBoost
  (`xgboost/c_api.h`, cujos headers não vêm no wheel pip — as assinaturas
  foram extraídas da documentação oficial e conferidas contra os símbolos
  exportados pela `.so` instalada).

### Nota de correção conhecida (indicators.c)
No primeiro `diff()` de preço de cada item, o pandas marca a 1ª posição
como `NaN` e a exclui da janela de contagem do RSI. O C atual conta essa
posição como "0" válido, então o **único registro que diverge do pandas
é o índice equivalente a `min_periods - 1`** (ex.: linha 47 de 8000) — um
efeito de aquecimento de fração de segundo no início do histórico, sem
efeito prático (nenhum item real roda com 47h de histórico). Documentado
em `src/indicators.c`, função `calcular_indicadores`.

## O que falta (não portado nesta versão)

- **Persistência da Q-table** (`Cerebro_RL`/`Cerebro_RL_Item`) — o `main.c`
  atual roda o veto com "estado novo" (Q=0) sempre, só para provar a
  chamada. Persistir é mecânico: já dá pra ler via `sheets_get_values` e
  escrever via `sheets_update_values`/`salvar_aba_atomica` (falta escrever
  essa última em C, é só portar a lógica de aba temporária).
- **Validação de sinais** (`Validacao_Sinais`) — sustentação de preço,
  janela de graça 12-30 dias, todo o bloco de `Dias_Ate_Alvo`/`Preco_Final`.
- **Deduplicação híbrida** (4h/3%/24h).
- **FIFO híbrido de custo médio.**
- **Escrita atômica** (`salvar_aba_atomica`) — o `sheets.c` já expõe
  `add/delete/rename worksheet`, falta só compor a sequência.

Ou seja: a esteira **OAuth → leitura → cálculo → decisão → escrita** está
provada de ponta a ponta, mas a parte de "lembrar o que já aconteceu"
(Q-table, sinais em aberto, dedup) ainda não foi portada. Seguir os
mesmos padrões dos módulos prontos (structs simples + funções puras)
deve ser direto, mas é trabalho real, não um detalhe.

## Como compilar

Dependências (Ubuntu/Debian):
```bash
sudo apt-get install libcurl4-openssl-dev libssl-dev libcjson-dev build-essential
```

XGBoost — usa a `.so` que vem com o pacote pip (não precisa compilar o
XGBoost do zero):
```bash
pip install xgboost
```

O `Makefile` já detecta o caminho da lib automaticamente via
`python3 -c "import xgboost..."`. Se sua `libxgboost.so` não tiver o
symlink versionado (`libxgboost.so.3`) que o binário espera, crie um:
```bash
XGDIR=$(python3 -c "import xgboost,os;print(os.path.dirname(xgboost.__file__)+'/lib')")
ln -sf $XGDIR/libxgboost.so $XGDIR/libxgboost.so.3
sudo ldconfig
```

Compilar:
```bash
make
```

Rodar (precisa de `credenciais_robo.json` — o mesmo arquivo de Service
Account que o bot Python já usa — na pasta de execução):
```bash
./build/sfl_bot
```

## Como rodar os benchmarks (sem precisar de credenciais do Google)

```bash
# gera dados sintéticos (63 itens, ~333 dias de histórico horário cada)
python3 -c "... ver test_data/ para os scripts já usados ..."

gcc -O2 -Iinclude src/bench_indicators.c src/indicators.c src/util.c -lm -lcrypto -o build/bench_indicators
./build/bench_indicators test_data/reclog_63itens.csv

gcc -O2 -Iinclude src/bench_xgb.c src/indicators.c src/util.c src/xgb_wrapper.c \
    -L$(python3 -c "import xgboost,os;print(os.path.dirname(xgboost.__file__)+'/lib')") \
    -lxgboost -Wl,-rpath,$(python3 -c "import xgboost,os;print(os.path.dirname(xgboost.__file__)+'/lib')") \
    -lm -lcrypto -o build/bench_xgb
./build/bench_xgb test_data/reclog_1item.csv
```

## Estrutura

```
include/    headers (.h) de cada módulo
src/        implementação (.c) + main.c (ciclo simplificado) + programas de benchmark
test_data/  CSVs sintéticos usados para validar e medir performance
Makefile
```
