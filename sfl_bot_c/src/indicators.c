#include "indicators.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Rolling mean/std por janela deslizante O(n) — mantém soma e soma dos
 * quadrados incrementalmente, em vez de recalcular a janela inteira a
 * cada posição (que seria O(n*window), o equivalente ingênuo do que um
 * loop Python puro faria sem numpy). O pandas .rolling().mean()/.std()
 * já faz otimização parecida internamente (é C compilado) — esta função
 * é o que permite uma comparação justa de performance contra o pandas,
 * em vez de comparar C ingênuo com numpy vetorizado.
 *
 * ddof=1 (desvio padrão amostral), igual ao default do pandas .std(). */
static void rolling_mean_std(const double *x, int n, int window, int min_periods,
                              double *out_mean, double *out_std) {
    double soma = 0.0, soma_sq = 0.0;
    for (int i = 0; i < n; i++) {
        soma += x[i];
        soma_sq += x[i] * x[i];
        if (i >= window) {
            soma -= x[i - window];
            soma_sq -= x[i - window] * x[i - window];
        }
        int count = (i + 1 < window) ? (i + 1) : window;

        if (count < min_periods) {
            out_mean[i] = NAN;
            out_std[i] = NAN;
        } else {
            double media = soma / count;
            out_mean[i] = media;
            if (count > 1) {
                double var = (soma_sq - (soma * soma) / count) / (count - 1);
                out_std[i] = (var > 0.0) ? sqrt(var) : 0.0;
            } else {
                out_std[i] = NAN;
            }
        }
    }
}

void calcular_indicadores(const PriceRecord *hist, int n, Indicadores *out) {
    out->n = n;
    out->media_macro  = malloc(sizeof(double) * (size_t)n);
    out->desvio_macro = malloc(sizeof(double) * (size_t)n);
    out->rsi_micro    = malloc(sizeof(double) * (size_t)n);
    out->lag1 = malloc(sizeof(double) * (size_t)n);
    out->lag2 = malloc(sizeof(double) * (size_t)n);
    out->lag3 = malloc(sizeof(double) * (size_t)n);
    out->hora_dia   = malloc(sizeof(int) * (size_t)n);
    out->dia_semana = malloc(sizeof(int) * (size_t)n);

    if (n == 0) return;

    double *precos = calloc((size_t)n, sizeof(double));
    for (int i = 0; i < n; i++) precos[i] = hist[i].price;

    /* Janela macro: 30 dias / 720 registros, min_periods=120 (5 dias) —
     * mesmos parâmetros do Python (comentário lá: "min_periods=240 exige
     * 10 dias completos" foi revisado para 120 na versão atual, ver
     * bot_4_24.py preparar_indicadores). */
    int janela_macro = 30 * REGISTROS_POR_DIA;
    rolling_mean_std(precos, n, janela_macro, 120, out->media_macro, out->desvio_macro);

    /* RSI — janela micro 10 dias / 240 registros, min_periods=48 */
    int janela_micro = 10 * REGISTROS_POR_DIA;
    double *ganho = calloc((size_t)n, sizeof(double));
    double *perda = calloc((size_t)n, sizeof(double));
    for (int i = 1; i < n; i++) {
        double delta = precos[i] - precos[i - 1];
        ganho[i] = delta > 0 ? delta : 0.0;
        perda[i] = delta < 0 ? -delta : 0.0;
    }
    double *ganho_medio = malloc(sizeof(double) * (size_t)n);
    double *perda_medio = malloc(sizeof(double) * (size_t)n);
    double *dummy_std   = malloc(sizeof(double) * (size_t)n);
    rolling_mean_std(ganho, n, janela_micro, 48, ganho_medio, dummy_std);
    rolling_mean_std(perda, n, janela_micro, 48, perda_medio, dummy_std);

    for (int i = 0; i < n; i++) {
        if (isnan(ganho_medio[i]) || isnan(perda_medio[i]) || perda_medio[i] == 0.0) {
            /* Replica rsi_raw.fillna(100) do Python: tanto o período de
             * warmup (< min_periods, ainda sem ganho/perda_medio) quanto
             * perda_medio==0 (mercado só subiu) viram NaN em rsi_raw antes
             * do fillna — os dois casos colapsam para 100 no final. */
            out->rsi_micro[i] = 100.0;
        } else {
            double rs = ganho_medio[i] / perda_medio[i];
            out->rsi_micro[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    free(ganho); free(perda); free(ganho_medio); free(perda_medio); free(dummy_std);

    /* Lags + hora/dia da semana */
    for (int i = 0; i < n; i++) {
        out->lag1[i] = (i >= 1) ? precos[i - 1] : NAN;
        out->lag2[i] = (i >= 2) ? precos[i - 2] : NAN;
        out->lag3[i] = (i >= 3) ? precos[i - 3] : NAN;

        struct tm t;
        gmtime_r(&hist[i].timestamp, &t);
        out->hora_dia[i]   = t.tm_hour;
        out->dia_semana[i] = t.tm_wday; /* 0=domingo, pandas usa 0=segunda — ajuste se for comparar valor a valor */
    }

    free(precos);
}

void indicadores_free(Indicadores *ind) {
    free(ind->media_macro); free(ind->desvio_macro); free(ind->rsi_micro);
    free(ind->lag1); free(ind->lag2); free(ind->lag3);
    free(ind->hora_dia); free(ind->dia_semana);
    memset(ind, 0, sizeof(*ind));
}

int detectar_regime(const PriceRecord *hist, int n, int janela_crash_h, double limiar_queda, double *out_variacao_24h) {
    if (n < janela_crash_h) { *out_variacao_24h = 0.0; return 0; }
    double preco_atual = hist[n - 1].price;
    double preco_24h = hist[n - janela_crash_h].price;
    if (preco_24h == 0.0) { *out_variacao_24h = 0.0; return 0; }
    double variacao = (preco_atual - preco_24h) / preco_24h;
    *out_variacao_24h = round(variacao * 10000.0) / 100.0; /* 2 casas, igual ao round(...,2) do Python */
    return variacao < limiar_queda;
}
