#ifndef SFL_CALIBRATION_H
#define SFL_CALIBRATION_H

/* calcular_taxa_confianca(): posterior Beta + EV — tradução 1:1 */
typedef struct {
    double taxa_efetiva;
    int confiante;
    double ev_esperado;
} TaxaConfianca;

TaxaConfianca calcular_taxa_confianca(int n_total, int n_acertos, double soma_recompensas,
                                       double prior_a, double prior_b, int n_minimo);

/* calcular_prior_empirico(): retorna (prior_a, prior_b) via ponteiros */
void calcular_prior_empirico(const double *pts, int n, double forca_prior,
                              double *out_prior_a, double *out_prior_b);

/* calcular_veto_rl(): veto adaptativo + Q-table híbrida (global+item) */
typedef struct {
    int vetado;
    double bonus_roi;
} VetoRL;

VetoRL calcular_veto_rl(double q_compra_global, int estado_existe_na_tabela,
                         int n_obs_estado, double q_compra_item, int tem_item, int n_obs_item);

/* calcular_stop_adaptativo(): stop normalizado pelo MAE do item */
typedef struct {
    int deve_stopar;
    double threshold;
} StopAdaptativo;

StopAdaptativo calcular_stop_adaptativo(double preco_atual, double roi_pred, double mae_relativo,
                                         double n_sigma, double stop_minimo, double stop_maximo);

/* calcular_margem_atencao(): margem proporcional à largura da banda Bollinger */
double calcular_margem_atencao(double alvo_in, double alvo_out, double fracao,
                                double minimo_abs, double maximo_pct);

/* calcular_roi_real(): ROI real observado — usado tanto no RL quanto na Alocação */
double calcular_roi_real(int tipo_compra /* 1=COMPRA, 0=VENDA/STOP */,
                          double preco_sinal, double preco_final, double taxa_marketplace);

#endif
