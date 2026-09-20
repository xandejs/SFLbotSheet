#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "indicators.h"
#include "xgb_wrapper.h"
#include "util.h"

static time_t parse_ts(const char *s) {
    struct tm t = {0};
    sscanf(s, "%d/%d/%d %d:%d", &t.tm_mday, &t.tm_mon, &t.tm_year, &t.tm_hour, &t.tm_min);
    t.tm_mon -= 1; t.tm_year -= 1900;
    return timegm(&t);
}

#define N_FEATURES 10
#define SHIFT_ALVO 288

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s arquivo.csv (1 item só)\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[512];
    if (!fgets(line, sizeof(line), f)) return 1;

    int cap = 20000, n = 0;
    PriceRecord *hist = malloc(sizeof(PriceRecord) * (size_t)cap);
    while (fgets(line, sizeof(line), f)) {
        char ts[64], item[64], price[32], season[32];
        if (sscanf(line, "%63[^,],%63[^,],%31[^,],%31[^\n]", ts, item, price, season) == 4) {
            hist[n].timestamp = parse_ts(ts);
            hist[n].price = limpar_numero(price);
            hist[n].season_num = 0;
            n++;
        }
    }
    fclose(f);

    Indicadores ind;
    calcular_indicadores(hist, n, &ind);

    int n_treino_max = n - SHIFT_ALVO;
    float *X = malloc(sizeof(float) * (size_t)n_treino_max * N_FEATURES);
    float *y = malloc(sizeof(float) * (size_t)n_treino_max);
    float *peso = malloc(sizeof(float) * (size_t)n_treino_max);
    int n_amostras = 0;
    time_t max_ts = hist[n - 1].timestamp;

    for (int i = 0; i < n_treino_max; i++) {
        if (isnan(ind.media_macro[i]) || isnan(ind.desvio_macro[i]) || isnan(ind.rsi_micro[i]) ||
            isnan(ind.lag1[i]) || isnan(ind.lag2[i]) || isnan(ind.lag3[i])) continue;
        int row = n_amostras * N_FEATURES;
        X[row+0]=(float)hist[i].price; X[row+1]=(float)ind.lag1[i]; X[row+2]=(float)ind.lag2[i];
        X[row+3]=(float)ind.lag3[i]; X[row+4]=(float)ind.rsi_micro[i]; X[row+5]=(float)ind.media_macro[i];
        X[row+6]=(float)ind.desvio_macro[i]; X[row+7]=(float)hist[i].season_num;
        X[row+8]=(float)ind.hora_dia[i]; X[row+9]=(float)ind.dia_semana[i];
        y[n_amostras] = (float)hist[i + SHIFT_ALVO].price;
        double dias_atras = difftime(max_ts, hist[i].timestamp) / 86400.0;
        peso[n_amostras] = (float)exp(-0.01 * dias_atras);
        n_amostras++;
    }
    printf("n_amostras=%d\n", n_amostras);

    float x_pred[N_FEATURES] = {
        (float)hist[n-1].price, (float)ind.lag1[n-1], (float)ind.lag2[n-1], (float)ind.lag3[n-1],
        (float)ind.rsi_micro[n-1], (float)ind.media_macro[n-1], (float)ind.desvio_macro[n-1],
        (float)hist[n-1].season_num, (float)ind.hora_dia[n-1], (float)ind.dia_semana[n-1]
    };
    float pred;

    clock_t t0 = clock();
    int ok = xgb_treinar_e_prever(X, y, peso, n_amostras, N_FEATURES, 40, 3, x_pred, &pred);
    clock_t t1 = clock();

    if (ok == 0) {
        printf("pred=%.6f\n", pred);
        printf("tempo_treino_predicao=%.4fs\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
    } else {
        printf("ERRO no treino\n");
    }

    free(X); free(y); free(peso);
    indicadores_free(&ind);
    free(hist);
    return 0;
}
