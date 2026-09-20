/*
 * sfl_bot.c — porte experimental do bot_4_24.py para C, para teste de
 * performance (ver conversa: motivação foi comparar tempo de execução,
 * não reescrever definitivamente).
 *
 * ESCOPO DESTA VERSÃO — o que está completo vs. simplificado:
 *
 *   COMPLETO (tradução 1:1, testável isoladamente):
 *     - OAuth2 Service Account (JWT RS256 + troca por access_token)
 *     - Cliente Sheets API v4 (get/update/append/clear/add/del/rename worksheet)
 *     - Indicadores técnicos (rolling mean/std O(n), RSI) — o hot path
 *       que motivou a pergunta de performance
 *     - Funções de calibração adaptativa (veto RL, taxa de confiança
 *       bayesiana, stop adaptativo, margem de atenção, ROI real)
 *     - Treino+inferência XGBoost via C API nativa
 *
 *   SIMPLIFICADO NESTA VERSÃO (não portado ainda — ver bot_4_24.py):
 *     - Persistência da Q-table (Cerebro_RL/Cerebro_RL_Item) — o ciclo
 *       abaixo roda o veto com Q=0 (estado novo) para todo item, só para
 *       provar a chamada; persistir precisa do parser incremental de CSV
 *       que já existe em sheets_get_values(), é só mapear.
 *     - Validação de sinais (Validacao_Sinais) e todo o cálculo de
 *       sustentação de preço / janela de graça de 12-30 dias
 *     - Deduplicação híbrida (4h/3%/24h)
 *     - Escrita atômica (o sheets.c já expõe add/del/rename — falta só
 *       replicar salvar_aba_atomica() por cima, é bem mecânico)
 *     - FIFO híbrido de custo médio
 *
 * Ou seja: isto SERVE para medir tempo de ciclo (I/O de rede + cálculo de
 * indicadores + treino XGBoost, que é o que domina o runtime), mas NÃO
 * está pronto para gerar sinais reais — a parte de decisão final e
 * persistência de estado ainda precisa ser portada seguindo os mesmos
 * padrões dos módulos já prontos.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curl/curl.h>

#include "util.h"
#include "oauth.h"
#include "sheets.h"
#include "indicators.h"
#include "calibration.h"
#include "xgb_wrapper.h"

#define NOME_PLANILHA "SFL 2025_10"
#define TAXA_MARKETPLACE 0.10
#define REGISTROS_POR_DIA 24
#define DIAS_SWING 12
#define SHIFT_ALVO (DIAS_SWING * REGISTROS_POR_DIA)  /* 288, estrutural — igual ao Python */

/* Parse "%d/%m/%Y %H:%M" (formato do RecLog) para time_t UTC.
 * Simplificação: trata o timestamp como UTC (o Python usa BRT sem tz e
 * compara datetime "naive" — para o teste de performance isso não afeta
 * o resultado, mas anote se for comparar valores ponto-a-ponto). */
static time_t parse_timestamp_brl(const char *s) {
    struct tm t = {0};
    if (sscanf(s, "%d/%d/%d %d:%d", &t.tm_mday, &t.tm_mon, &t.tm_year, &t.tm_hour, &t.tm_min) != 5)
        return (time_t)-1;
    t.tm_mon -= 1;
    t.tm_year -= 1900;
    return timegm(&t);
}

int main(void) {
    /* CRÍTICO para depurar no log do Actions: por padrão, stdout é
     * "fully buffered" quando não é um terminal (é sempre o caso de logs
     * de CI) — os printf() ficam represados até o buffer encher ou o
     * processo terminar, então progresso real não aparece no log ao
     * vivo. Isso é a causa mais provável de "parece travado" quando na
     * verdade só não está mostrando nada ainda. */
    setvbuf(stdout, NULL, _IONBF, 0);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    clock_t t_inicio = clock();

    printf("🚀 Iniciando Bot (C) — porte experimental de performance...\n");

    /* ---------- 1. OAuth2 ---------- */
    ServiceAccount sa;
    if (carregar_service_account("credenciais_robo.json", &sa) != 0) return 1;

    clock_t t_step = clock();
    char access_token[2048];
    if (obter_access_token(&sa, access_token, sizeof(access_token)) != 0) {
        fprintf(stderr, "❌ Falha ao obter access_token — abortando.\n");
        return 1;
    }
    printf("🔑 Access token obtido em %.2fs.\n", (double)(clock() - t_step) / CLOCKS_PER_SEC);

    /* ---------- 2. Localizar planilha ---------- */
    t_step = clock();
    SheetsClient cli;
    snprintf(cli.access_token, sizeof(cli.access_token), "%s", access_token);
    if (drive_localizar_planilha(access_token, NOME_PLANILHA, cli.spreadsheet_id, sizeof(cli.spreadsheet_id)) != 0)
        return 1;
    printf("📄 Planilha localizada em %.2fs: %s\n",
           (double)(clock() - t_step) / CLOCKS_PER_SEC, cli.spreadsheet_id);

    /* ---------- 3. Ler RecLog ---------- */
    t_step = clock();
    printf("📥 Lendo RecLog (pode demorar se a aba tiver muitas linhas acumuladas — "
           "mesma leitura completa que o Python faz via get_all_values())...\n");
    SheetValues rec;
    if (sheets_get_values(&cli, "RecLog", &rec) != 0) return 1;
    printf("📥 RecLog: %d linhas lidas em %.2fs.\n",
           rec.n_rows, (double)(clock() - t_step) / CLOCKS_PER_SEC);

    if (rec.n_rows < 2) {
        fprintf(stderr, "⚠️ RecLog vazio — nada a processar.\n");
        sheet_values_free(&rec);
        return 0;
    }

    /* Agrupa por Item (colunas: Timestamp, Item, Price, Season) — mesma
     * estrutura de df_bruto no Python. Aqui, simplificação deliberada:
     * processa só o PRIMEIRO item único encontrado, como prova de conceito
     * do ciclo completo. Para os 63 itens, envolva o bloco abaixo (a partir
     * de "processar 1 item") num loop sobre os nomes únicos de Item. */
    char item_alvo[128] = "";
    for (int i = 1; i < rec.n_rows && item_alvo[0] == '\0'; i++) {
        if (rec.n_cols[i] >= 2) snprintf(item_alvo, sizeof(item_alvo), "%s", rec.rows[i][1]);
    }
    if (item_alvo[0] == '\0') {
        fprintf(stderr, "❌ Nenhum item encontrado no RecLog.\n");
        sheet_values_free(&rec);
        return 1;
    }
    printf("🎯 Processando item de exemplo: %s\n", item_alvo);

    int cap = rec.n_rows;
    PriceRecord *hist = malloc(sizeof(PriceRecord) * (size_t)cap);
    int n_hist = 0;
    for (int i = 1; i < rec.n_rows; i++) {
        if (rec.n_cols[i] < 3) continue;
        if (strcmp(rec.rows[i][1], item_alvo) != 0) continue;
        time_t ts = parse_timestamp_brl(rec.rows[i][0]);
        if (ts == (time_t)-1) continue;
        hist[n_hist].timestamp = ts;
        hist[n_hist].price = limpar_numero(rec.rows[i][2]);
        hist[n_hist].season_num = 0; /* mapeamento Season->num omitido nesta versão de teste */
        n_hist++;
    }
    printf("📊 %d registros históricos para '%s'.\n", n_hist, item_alvo);
    sheet_values_free(&rec);

    if (n_hist < 50 + SHIFT_ALVO) {
        fprintf(stderr, "⚠️ Histórico insuficiente para treino (precisa >= %d, tem %d).\n",
                50 + SHIFT_ALVO, n_hist);
        free(hist);
        return 0;
    }

    /* ---------- 4. Indicadores (o hot path do teste de performance) ---------- */
    clock_t t_ind_ini = clock();
    Indicadores ind;
    calcular_indicadores(hist, n_hist, &ind);
    clock_t t_ind_fim = clock();
    printf("📈 Indicadores calculados em %.4fs (%d registros).\n",
           (double)(t_ind_fim - t_ind_ini) / CLOCKS_PER_SEC, n_hist);

    double em_crash_var;
    int em_crash = detectar_regime(hist, n_hist, 24, -0.12, &em_crash_var);
    printf("%s Detector de regime: variação 24h = %.2f%%\n", em_crash ? "⚠️" : "✅", em_crash_var);

    /* ---------- 5. Montar dataset de treino (Preco_Futuro = shift(-288)) ---------- */
    /* 10 features, igual ao X_cols do Python: Price, Lag_1/2/3, RSI_Micro,
     * Media_Macro, Desvio_Macro, Season_Num, Hora_Dia, Dia_Semana */
    #define N_FEATURES 10
    int n_treino_max = n_hist - SHIFT_ALVO;
    float *X = malloc(sizeof(float) * (size_t)n_treino_max * N_FEATURES);
    float *y = malloc(sizeof(float) * (size_t)n_treino_max);
    float *peso = malloc(sizeof(float) * (size_t)n_treino_max);
    int n_amostras = 0;

    time_t max_ts = hist[n_hist - 1].timestamp;
    for (int i = 0; i < n_treino_max; i++) {
        if (isnan(ind.media_macro[i]) || isnan(ind.desvio_macro[i]) || isnan(ind.rsi_micro[i]) ||
            isnan(ind.lag1[i]) || isnan(ind.lag2[i]) || isnan(ind.lag3[i]))
            continue; /* dropna(), igual ao Python */

        int row = n_amostras * N_FEATURES;
        X[row + 0] = (float)hist[i].price;
        X[row + 1] = (float)ind.lag1[i];
        X[row + 2] = (float)ind.lag2[i];
        X[row + 3] = (float)ind.lag3[i];
        X[row + 4] = (float)ind.rsi_micro[i];
        X[row + 5] = (float)ind.media_macro[i];
        X[row + 6] = (float)ind.desvio_macro[i];
        X[row + 7] = (float)hist[i].season_num;
        X[row + 8] = (float)ind.hora_dia[i];
        X[row + 9] = (float)ind.dia_semana[i];

        y[n_amostras] = (float)hist[i + SHIFT_ALVO].price;
        double dias_atras = difftime(max_ts, hist[i].timestamp) / 86400.0;
        peso[n_amostras] = (float)exp(-0.01 * dias_atras); /* Peso_Temporal */
        n_amostras++;
    }
    printf("🧮 Dataset de treino: %d amostras válidas (de %d possíveis).\n", n_amostras, n_treino_max);

    if (n_amostras < 50) {
        fprintf(stderr, "⚠️ Menos de 50 amostras de treino — item ignorado neste ciclo (igual ao Python).\n");
        goto limpeza;
    }

    /* ---------- 6. Treino + inferência XGBoost ---------- */
    float x_pred[N_FEATURES] = {
        (float)hist[n_hist - 1].price, (float)ind.lag1[n_hist - 1], (float)ind.lag2[n_hist - 1],
        (float)ind.lag3[n_hist - 1], (float)ind.rsi_micro[n_hist - 1], (float)ind.media_macro[n_hist - 1],
        (float)ind.desvio_macro[n_hist - 1], (float)hist[n_hist - 1].season_num,
        (float)ind.hora_dia[n_hist - 1], (float)ind.dia_semana[n_hist - 1]
    };
    float pred = 0.0f;

    clock_t t_xgb_ini = clock();
    int xgb_ok = xgb_treinar_e_prever(X, y, peso, n_amostras, N_FEATURES, 40, 3, x_pred, &pred);
    clock_t t_xgb_fim = clock();

    if (xgb_ok != 0) {
        fprintf(stderr, "❌ Treino/inferência XGBoost falhou.\n");
        goto limpeza;
    }
    printf("🌲 XGBoost treinado+predito em %.4fs.\n", (double)(t_xgb_fim - t_xgb_ini) / CLOCKS_PER_SEC);

    double preco_atual = hist[n_hist - 1].price;
    double roi = ((pred - preco_atual) / preco_atual) * 100.0;
    printf("💡 Preço atual: %.4f | Previsão (12d): %.4f | ROI previsto: %.2f%%\n",
           preco_atual, pred, roi);

    /* ---------- 7. Calibração adaptativa (demo com dados neutros) ---------- */
    /* Numa integração completa, estes valores viriam de Validacao_Sinais e
     * Cerebro_RL já carregados — aqui simulamos "sem histórico ainda" para
     * provar que a chamada e a fórmula batem com o Python. */
    VetoRL veto = calcular_veto_rl(0.0, 0, 0, 0.0, 0, 0);
    double margem = calcular_margem_atencao(
        ind.media_macro[n_hist - 1] - 2.0 * ind.desvio_macro[n_hist - 1],
        ind.media_macro[n_hist - 1] + 2.0 * ind.desvio_macro[n_hist - 1],
        0.15, 0.0001, 0.10);
    printf("🧠 Veto RL (estado novo): vetado=%s | Margem de atenção: %.6f\n",
           veto.vetado ? "sim" : "não", margem);

    /* ---------- 8. Escrever resultado em Sinais_Hoje (demo, 1 linha) ---------- */
    if (!sheets_worksheet_existe(&cli, "Sinais_Hoje_C_Teste")) {
        sheets_add_worksheet(&cli, "Sinais_Hoje_C_Teste", 200, 9);
    }
    char roi_str[32], alvo_in_str[32], alvo_out_str[32];
    snprintf(roi_str, sizeof(roi_str), "%.2f", roi);
    snprintf(alvo_in_str, sizeof(alvo_in_str), "%.4f",
             ind.media_macro[n_hist - 1] - 2.0 * ind.desvio_macro[n_hist - 1]);
    snprintf(alvo_out_str, sizeof(alvo_out_str), "%.4f",
             ind.media_macro[n_hist - 1] + 2.0 * ind.desvio_macro[n_hist - 1]);

    char *linha_valores[1][9] = {{
        item_alvo, roi_str, (char *)"TESTE_C", alvo_in_str, alvo_out_str,
        (char *)"-", (char *)"-", (char *)"-", (char *)"gerado_por_C"
    }};
    sheets_append_rows(&cli, "Sinais_Hoje_C_Teste", (char ***)linha_valores, 1, 9);
    printf("📌 Linha de teste gravada em 'Sinais_Hoje_C_Teste'.\n");

limpeza:
    free(X); free(y); free(peso);
    indicadores_free(&ind);
    free(hist);

    clock_t t_fim = clock();
    printf("🏁 Ciclo (C) finalizado em %.4fs de CPU.\n", (double)(t_fim - t_inicio) / CLOCKS_PER_SEC);

    curl_global_cleanup();
    return 0;
}
