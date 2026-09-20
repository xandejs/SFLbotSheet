#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "indicators.h"
#include "util.h"

static time_t parse_ts(const char *s) {
    struct tm t = {0};
    sscanf(s, "%d/%d/%d %d:%d", &t.tm_mday, &t.tm_mon, &t.tm_year, &t.tm_hour, &t.tm_min);
    t.tm_mon -= 1; t.tm_year -= 1900;
    return timegm(&t);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s arquivo.csv\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[512];
    fgets(line, sizeof(line), f); /* pula header */

    int cap = 20000, n = 0;
    PriceRecord *hist = malloc(sizeof(PriceRecord) * (size_t)cap);

    clock_t t0 = clock();
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
    clock_t t1 = clock();

    printf("n=%d tempo_total=%.6fs (leitura+calculo)\n", n, (double)(t1 - t0) / CLOCKS_PER_SEC);

    int checkpoints[] = {0, 47, 48, 119, 120, 500, 719, 720, 1000, 3000, 7999};
    for (size_t k = 0; k < sizeof(checkpoints) / sizeof(int); k++) {
        int i = checkpoints[k];
        if (i >= n) continue;
        printf("i=%5d price=%.4f media_macro=%.6f desvio_macro=%.6f rsi=%.4f lag1=%.4f hora=%d dow=%d\n",
               i, hist[i].price,
               ind.media_macro[i], ind.desvio_macro[i], ind.rsi_micro[i],
               ind.lag1[i], ind.hora_dia[i], ind.dia_semana[i]);
    }

    indicadores_free(&ind);
    free(hist);
    return 0;
}
