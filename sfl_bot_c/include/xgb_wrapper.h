#ifndef SFL_XGB_WRAPPER_H
#define SFL_XGB_WRAPPER_H

/* Treina 1 modelo XGBoost (equivalente a xgb.XGBRegressor(n_estimators=40,
 * max_depth=3).fit(X, y, sample_weight=peso) do bot_4_24.py) e prediz o
 * valor para `x_pred` (1 linha, n_features colunas).
 *
 * X: matriz row-major (n_amostras x n_features), floats.
 * y: vetor alvo (Preco_Futuro), tamanho n_amostras.
 * peso: vetor de peso temporal, tamanho n_amostras (pode ser NULL -> peso 1.0).
 * x_pred: 1 linha de n_features valores, a predição é escrita em *out_pred.
 *
 * Retorna 0 em sucesso, -1 em erro (mensagem em XGBGetLastError()). */
int xgb_treinar_e_prever(const float *X, const float *y, const float *peso,
                          int n_amostras, int n_features,
                          int n_estimators, int max_depth,
                          const float *x_pred, float *out_pred);

#endif
