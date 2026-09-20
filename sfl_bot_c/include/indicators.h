#ifndef SFL_INDICATORS_H
#define SFL_INDICATORS_H

#include <stddef.h>
#include <time.h>

#define REGISTROS_POR_DIA 24

typedef struct {
    time_t timestamp;
    double price;
    int season_num;   /* 0=PRIMAVERA 1=VERÃO 2=OUTONO 3=INVERNO, igual ao MAP_ESTACOES */
} PriceRecord;

/* Espelha as colunas geradas por preparar_indicadores() do bot_4_24.py
 * para UM item. NAN (de <math.h>) representa o mesmo papel que NaN do
 * pandas — "sem dado suficiente ainda" (equivalente ao min_periods). */
typedef struct {
    double *media_macro;   /* rolling mean, janela 720, min_periods 120 */
    double *desvio_macro;  /* rolling std,  janela 720, min_periods 120 */
    double *rsi_micro;     /* RSI,          janela 240, min_periods 48  */
    double *lag1, *lag2, *lag3;
    int *hora_dia, *dia_semana;
    int n;
} Indicadores;

/* Calcula os indicadores para o histórico completo de um item (já
 * ordenado por timestamp). Aloca os arrays internamente — chame
 * indicadores_free() depois. Equivalente ao núcleo de
 * preparar_indicadores()/​_calcular_indicadores_do_zero() do Python,
 * exceto pela parte de holdout (tratada à parte, ver calibration.h). */
void calcular_indicadores(const PriceRecord *hist, int n, Indicadores *out);
void indicadores_free(Indicadores *ind);

/* Detector de regime (crash 24h) — cópia direta de detectar_regime(). */
int detectar_regime(const PriceRecord *hist, int n, int janela_crash_h, double limiar_queda, double *out_variacao_24h);

#endif
