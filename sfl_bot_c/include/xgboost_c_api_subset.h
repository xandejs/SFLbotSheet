#ifndef XGBOOST_C_API_SUBSET_H
#define XGBOOST_C_API_SUBSET_H
/* Subconjunto da C API pública do XGBoost (https://xgboost.readthedocs.io/en/stable/c.html).
 * O wheel do pip só empacota libxgboost.so (usado via ctypes pelo binding
 * Python) e não inclui os headers C — as assinaturas abaixo são as
 * declaradas na documentação oficial do projeto, conferidas contra os
 * símbolos exportados pela .so instalada (ver `nm -D libxgboost.so`). */

typedef void *DMatrixHandle;
typedef void *BoosterHandle;

#ifdef __cplusplus
extern "C" {
#endif

const char *XGBGetLastError(void);

int XGDMatrixCreateFromMat(const float *data, unsigned long nrow, unsigned long ncol,
                            float missing, DMatrixHandle *out);
int XGDMatrixFree(DMatrixHandle handle);
/* field = "label" (alvo de regressão, Preco_Futuro) ou "weight" (Peso_Temporal) */
int XGDMatrixSetFloatInfo(DMatrixHandle handle, const char *field, const float *info, unsigned long len);

int XGBoosterCreate(const DMatrixHandle dmats[], unsigned long len, BoosterHandle *out);
int XGBoosterFree(BoosterHandle handle);
int XGBoosterSetParam(BoosterHandle handle, const char *name, const char *value);
int XGBoosterUpdateOneIter(BoosterHandle handle, int iter, DMatrixHandle dtrain);

int XGBoosterPredict(BoosterHandle handle, DMatrixHandle dmat, int option_mask,
                      unsigned ntree_limit, int training,
                      unsigned long *out_len, const float **out_result);

#ifdef __cplusplus
}
#endif

#endif
