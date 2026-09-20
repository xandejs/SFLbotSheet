#include "xgb_wrapper.h"
#include "xgboost_c_api_subset.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(call) do { \
    if ((call) != 0) { \
        fprintf(stderr, "❌ XGBoost erro em '%s': %s\n", #call, XGBGetLastError()); \
        return -1; \
    } \
} while (0)

int xgb_treinar_e_prever(const float *X, const float *y, const float *peso,
                          int n_amostras, int n_features,
                          int n_estimators, int max_depth,
                          const float *x_pred, float *out_pred) {
    DMatrixHandle dtrain = NULL, dpred = NULL;
    BoosterHandle booster = NULL;
    int ret = -1;

    if (XGDMatrixCreateFromMat(X, (unsigned long)n_amostras, (unsigned long)n_features,
                                NAN, &dtrain) != 0) {
        fprintf(stderr, "❌ XGDMatrixCreateFromMat(treino): %s\n", XGBGetLastError());
        goto fim;
    }
    if (XGDMatrixSetFloatInfo(dtrain, "label", y, (unsigned long)n_amostras) != 0) {
        fprintf(stderr, "❌ XGDMatrixSetFloatInfo(label): %s\n", XGBGetLastError());
        goto fim;
    }
    if (peso != NULL) {
        if (XGDMatrixSetFloatInfo(dtrain, "weight", peso, (unsigned long)n_amostras) != 0) {
            fprintf(stderr, "❌ XGDMatrixSetFloatInfo(weight): %s\n", XGBGetLastError());
            goto fim;
        }
    }

    if (XGBoosterCreate(&dtrain, 1, &booster) != 0) {
        fprintf(stderr, "❌ XGBoosterCreate: %s\n", XGBGetLastError());
        goto fim;
    }

    /* Parâmetros equivalentes ao XGBRegressor(n_estimators=40, max_depth=3)
     * do sklearn wrapper — 'eta'/'objective' usam os defaults do sklearn
     * wrapper (0.3 / reg:squarederror); se quiser bater 100% com o Python,
     * ajuste aqui para o que veio de model.get_params(). */
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", max_depth);
    XGBoosterSetParam(booster, "max_depth", buf);
    XGBoosterSetParam(booster, "objective", "reg:squarederror");
    XGBoosterSetParam(booster, "eta", "0.3");
    XGBoosterSetParam(booster, "verbosity", "0");

    for (int iter = 0; iter < n_estimators; iter++) {
        if (XGBoosterUpdateOneIter(booster, iter, dtrain) != 0) {
            fprintf(stderr, "❌ XGBoosterUpdateOneIter(iter=%d): %s\n", iter, XGBGetLastError());
            goto fim;
        }
    }

    if (XGDMatrixCreateFromMat(x_pred, 1, (unsigned long)n_features, NAN, &dpred) != 0) {
        fprintf(stderr, "❌ XGDMatrixCreateFromMat(pred): %s\n", XGBGetLastError());
        goto fim;
    }

    unsigned long out_len = 0;
    const float *out_result = NULL;
    if (XGBoosterPredict(booster, dpred, 0, 0, 0, &out_len, &out_result) != 0) {
        fprintf(stderr, "❌ XGBoosterPredict: %s\n", XGBGetLastError());
        goto fim;
    }
    if (out_len < 1) {
        fprintf(stderr, "❌ XGBoosterPredict retornou vetor vazio\n");
        goto fim;
    }
    *out_pred = out_result[0];
    ret = 0;

fim:
    if (dtrain) XGDMatrixFree(dtrain);
    if (dpred) XGDMatrixFree(dpred);
    if (booster) XGBoosterFree(booster);
    return ret;
}
