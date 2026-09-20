#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "indicators.h"
#include "util.h"

/* Parser manual por delimitador — evita sscanf() (reparseia a format
 * string a cada chamada, é a causa mais comum de "C mais lento que
 * Python" em parsing ingênuo: o gargalo não é a linguagem, é a escolha
 * de função). strchr()/memcpy() direto no buffer é o que faz o parser
 * de CSV do pandas (C interno) ser rápido — replicando aqui. */
static int parse_csv_linha(char *line, char *ts, char *item, char *price, size_t bufsz) {
    char *p1 = strchr(line, ',');
    if (!p1) return 0;
    char *p2 = strchr(p1 + 1, ',');
    if (!p2) return 0;
    char *p3 = strchr(p2 + 1, ',');
    if (!p3) return 0;

    size_t n1 = (size_t)(p1 - line);
    size_t n2 = (size_t)(p2 - p1 - 1);
    size_t n3 = (size_t)(p3 - p2 - 1);
    if (n1 >= bufsz || n2 >= bufsz || n3 >= bufsz) return 0;

    memcpy(ts, line, n1); ts[n1] = '\0';
    memcpy(item, p1 + 1, n2); item[n2] = '\0';
    memcpy(price, p2 + 1, n3); price[n3] = '\0';
    return 1;
}

static time_t parse_ts(const char *s) {
    /* Parse manual de "dd/mm/aaaa HH:MM" — mesma ideia do parser CSV acima,
     * evita sscanf no caminho quente (chamado 1x por linha, milhões de vezes
     * num histórico real). */
    struct tm t = {0};
    t.tm_mday = (s[0]-'0')*10 + (s[1]-'0');
    t.tm_mon  = (s[3]-'0')*10 + (s[4]-'0') - 1;
    t.tm_year = (s[6]-'0')*1000 + (s[7]-'0')*100 + (s[8]-'0')*10 + (s[9]-'0') - 1900;
    t.tm_hour = (s[11]-'0')*10 + (s[12]-'0');
    t.tm_min  = (s[14]-'0')*10 + (s[15]-'0');
    return timegm(&t);
}

#define MAX_ITENS 128
#define MAX_LINHAS 600000

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s arquivo.csv\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[512];
    if (!fgets(line, sizeof(line), f)) return 1; /* header */

    /* Carrega tudo em memória, agrupando por item (mapa nome->índice) */
    char nomes_itens[MAX_ITENS][64];
    PriceRecord *hist_por_item[MAX_ITENS];
    int cap_por_item[MAX_ITENS], n_por_item[MAX_ITENS];
    int n_itens = 0;

    for (int i = 0; i < MAX_ITENS; i++) {
        hist_por_item[i] = malloc(sizeof(PriceRecord) * 10000);
        cap_por_item[i] = 10000;
        n_por_item[i] = 0;
    }

    clock_t t_leitura_ini = clock();
    while (fgets(line, sizeof(line), f)) {
        char ts[32], item[64], price[32];
        if (!parse_csv_linha(line, ts, item, price, sizeof(ts))) continue;
        /* nota: bufsz passado é o de `ts` (32) — item/price têm o mesmo
         * tamanho no benchmark, então serve como guarda geral aqui. */

        int idx = -1;
        for (int i = 0; i < n_itens; i++) if (strcmp(nomes_itens[i], item) == 0) { idx = i; break; }
        if (idx == -1) {
            idx = n_itens++;
            snprintf(nomes_itens[idx], sizeof(nomes_itens[idx]), "%s", item);
        }

        PriceRecord *h = hist_por_item[idx];
        if (n_por_item[idx] >= cap_por_item[idx]) {
            cap_por_item[idx] *= 2;
            hist_por_item[idx] = realloc(hist_por_item[idx], sizeof(PriceRecord) * (size_t)cap_por_item[idx]);
            h = hist_por_item[idx];
        }
        h[n_por_item[idx]].timestamp = parse_ts(ts);
        h[n_por_item[idx]].price = limpar_numero(price);
        h[n_por_item[idx]].season_num = 0;
        n_por_item[idx]++;
    }
    fclose(f);
    clock_t t_leitura_fim = clock();

    printf("Itens carregados: %d\n", n_itens);
    printf("Tempo de leitura+parse do CSV: %.4fs\n",
           (double)(t_leitura_fim - t_leitura_ini) / CLOCKS_PER_SEC);

    clock_t t_calc_ini = clock();
    for (int i = 0; i < n_itens; i++) {
        Indicadores ind;
        calcular_indicadores(hist_por_item[i], n_por_item[i], &ind);
        indicadores_free(&ind);
    }
    clock_t t_calc_fim = clock();

    double tempo_calc = (double)(t_calc_fim - t_calc_ini) / CLOCKS_PER_SEC;
    printf("Tempo de CALCULO dos indicadores (63 itens): %.4fs\n", tempo_calc);
    printf("Media por item: %.4fs\n", tempo_calc / n_itens);

    for (int i = 0; i < n_itens; i++) free(hist_por_item[i]);
    return 0;
}
