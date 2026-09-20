#include "calibration.h"
#include <math.h>

TaxaConfianca calcular_taxa_confianca(int n_total, int n_acertos, double soma_recompensas,
                                       double prior_a, double prior_b, int n_minimo) {
    TaxaConfianca r = {0.5, 0, 0.0};
    if (n_total == 0) return r;

    double a = prior_a + n_acertos;
    double b = prior_b + (n_total - n_acertos);

    r.taxa_efetiva = a / (a + b);
    r.ev_esperado = soma_recompensas / n_total;

    double variancia = (a * b) / ((a + b) * (a + b) * (a + b + 1));
    double ic_inferior = r.taxa_efetiva - 1.96 * sqrt(variancia);
    if (ic_inferior < 0) ic_inferior = 0;
    r.confiante = (n_total >= n_minimo) && (ic_inferior > 0.35);

    return r;
}

void calcular_prior_empirico(const double *pts, int n, double forca_prior,
                              double *out_prior_a, double *out_prior_b) {
    if (n == 0) { *out_prior_a = 1.0; *out_prior_b = 1.0; return; }
    int acertos = 0;
    for (int i = 0; i < n; i++) if (pts[i] >= 0.5) acertos++;
    double taxa_global = (double)acertos / n;
    double a = forca_prior * taxa_global;
    double b = forca_prior * (1.0 - taxa_global);
    *out_prior_a = (a > 0.5) ? a : 0.5;
    *out_prior_b = (b > 0.5) ? b : 0.5;
}

VetoRL calcular_veto_rl(double q_compra_global, int estado_existe_na_tabela,
                         int n_obs_estado, double q_compra_item, int tem_item, int n_obs_item) {
    VetoRL r = {0, 0.0};

    double q_compra;
    if (tem_item) {
        double peso_item = n_obs_item / 10.0;
        if (peso_item > 1.0) peso_item = 1.0;
        q_compra = peso_item * q_compra_item + (1 - peso_item) * q_compra_global;
    } else {
        q_compra = q_compra_global;
    }

    if (!estado_existe_na_tabela && !tem_item) return r; /* sem dado nenhum -> não veta */

    double threshold;
    if (n_obs_estado < 5) threshold = -0.5;
    else if (n_obs_estado < 20) threshold = -0.5 + (0.05 / 15.0) * (n_obs_estado - 5);
    else threshold = -0.45;

    r.vetado = q_compra < threshold;
    if (q_compra > 0.7) r.bonus_roi = 10.0;
    else if (q_compra > 0.5) r.bonus_roi = 5.0;

    return r;
}

StopAdaptativo calcular_stop_adaptativo(double preco_atual, double roi_pred, double mae_relativo,
                                         double n_sigma, double stop_minimo, double stop_maximo) {
    StopAdaptativo r;
    double threshold;
    (void)preco_atual;

    if (mae_relativo > 0.0) {
        double threshold_bruto = -n_sigma * mae_relativo;
        /* clamp: max(stop_maximo, min(stop_minimo, threshold_bruto)) — mesma
         * ordem do Python, cuidado ao trocar (stop_maximo é o mais negativo) */
        double v = threshold_bruto < stop_minimo ? threshold_bruto : stop_minimo;
        threshold = v > stop_maximo ? v : stop_maximo;
    } else {
        threshold = -0.20; /* fallback sem dados de MAE */
    }

    r.threshold = threshold;
    r.deve_stopar = roi_pred < threshold;
    return r;
}

double calcular_margem_atencao(double alvo_in, double alvo_out, double fracao,
                                double minimo_abs, double maximo_pct) {
    double banda = alvo_out - alvo_in;
    if (banda < 0) banda = 0;
    double margem = banda * fracao;
    if (margem < minimo_abs) margem = minimo_abs;
    if (alvo_in > 0) {
        double teto = alvo_in * maximo_pct;
        if (margem > teto) margem = teto;
    }
    return round(margem * 1000000.0) / 1000000.0; /* round(...,6) */
}

double calcular_roi_real(int tipo_compra, double preco_sinal, double preco_final, double taxa_marketplace) {
    if (preco_sinal <= 0) return 0.0;
    if (tipo_compra) {
        double venda_liquida = preco_final * (1 - taxa_marketplace);
        return ((venda_liquida - preco_sinal) / preco_sinal) * 100.0;
    }
    return ((preco_sinal - preco_final) / preco_sinal) * 100.0;
}
