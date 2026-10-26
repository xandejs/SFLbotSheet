"""Testes de fumaça do sfl_core com dados sintéticos. Rode: python test_core.py"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sfl_core import *   # noqa

rng = np.random.default_rng(7)
AGORA = datetime(2026, 10, 23, 12, 0)
N_H = 120 * 24


def fmt(x):  # imita pt-BR do get_all_values
    return f"{x:.6f}".replace(".", ",")


def serie(base, vol=0.01):
    return base * np.exp(np.cumsum(rng.normal(0, vol, N_H)))


def gerar_rec():
    itens = {"Wheat": serie(0.02), "Corn": serie(0.05), "Barley": serie(0.03),
             "Egg": serie(0.04), "Wood": serie(0.01), "Stone": serie(0.2), "Iron": serie(1.0),
             "Gold": serie(10.0), "Zeroso": serie(0.5)}
    ts0 = AGORA - timedelta(hours=N_H)
    linhas = [["Timestamp", "Item", "Price", "Season"]]   # cabeçalho no meio do caminho: deve ser descartado
    for h in range(N_H):
        if 500 < h < 510:         # buraco de coleta (RecLog adiado)
            continue
        t = (ts0 + timedelta(hours=h)).strftime("%d/%m/%Y %H:%M:%S")
        for it, s in itens.items():
            p = s[h]
            if it == "Zeroso" and h % 3 == 0:
                p = 0.0           # "sem oferta"
            linhas.append([t, it, fmt(p), ["PRIMAVERA", "VERÃO", "OUTONO", "INVERNO"][(h // 168) % 4]])
    linhas.append(linhas[-1])     # duplicata
    return linhas


def main():
    # ---- 1. dados -----------------------------------------------------------
    rec = gerar_rec()
    df, st = preparar_df_bruto(rec)
    print("stats:", st)
    assert (df["Price"] > 0).all(), "preço 0 não foi descartado"
    assert st["preco_zero"] > 0 and st["duplicadas"] >= 1 and st["data_invalida"] == 1
    assert not df.duplicated(["Timestamp", "Item"]).any()

    # ---- 2. alvo por TEMPO (gap + zeros não deslocam o horizonte) -------------
    cache, idx = {}, construir_indices_insumos(df)
    d = construir_features_item(df, "Zeroso", cache, idx)
    lin = d[d["Preco_Futuro"].notna()].iloc[100]
    alvo_t = lin["Timestamp"] + timedelta(days=DIAS_SWING)
    real = df[(df["Item"] == "Zeroso")].set_index("Timestamp")["Price"]
    prox = real.iloc[real.index.get_indexer([alvo_t], method="nearest")[0]]
    assert abs(lin["Preco_Futuro"] - prox) < 1e-12, "Preco_Futuro não é o preço de ~12 dias depois"
    print("✓ alvo por tempo")

    # ---- 3. pipeline + modelo pooled relativo ---------------------------------
    cache_ind = {}
    for it in df["Item"].unique():
        cache_ind[it] = construir_features_item(df, it, cache, idx)
    assert cache_ind["Egg"]["Indice_Dist"].notna().sum() > 100, "índice de insumo vazio"
    assert cache_ind["Gold"]["Margem_Dist"].notna().sum() > 100

    tr, te, cats = montar_datasets_pooled(cache_ind)
    assert len(cats) == 9 and not te.empty
    modelo = treinar_modelo_pooled(tr, cats)
    rois = prever_rois(modelo, cats, cache_ind)
    assert set(rois) == set(cats) and all(abs(v) < 200 for v in rois.values())
    aud = calcular_audit_linhas(modelo, cats, tr, te, "23/10/2026 12:00")
    assert aud and all(l[2] > 0 for l in aud), "MAE zerado (arredondamento?)"
    print(f"✓ modelo pooled: {len(rois)} previsões, {len(aud)} linhas de audit, ex.: {aud[0]}")

    # ---- 4. cache incremental == recálculo do zero -----------------------------
    corte = AGORA - timedelta(hours=6)
    df_antigo = df[df["Timestamp"] <= corte]
    c2 = {}
    preparar_indicadores_incremental(df_antigo, "Egg", c2)
    inc = preparar_indicadores_incremental(df, "Egg", c2)
    zero = preparar_indicadores_incremental(df, "Egg", {})
    for col in ["Media_Macro", "Desvio_Macro", "RSI_Micro", "Preco_Futuro"]:
        assert np.allclose(inc[col].fillna(-1).values, zero[col].fillna(-1).values, rtol=1e-9), f"{col} diverge"
    print("✓ cache incremental idêntico ao recálculo")

    # ---- 5. EV pode ser negativo e vetar (antes era inalcançável) -------------
    vals = []
    for i in range(10):    # 5 dias entre sinais => 10 episódios distintos
        vals.append([i + 1, "COMPRA", (datetime(2026, 7, 1, 10, 0) + timedelta(days=5 * i)).strftime("%d/%m/%Y %H:%M"),
                     "Egg", 1.0, 1.2, 1.0, "❌ FALHOU", 30,
                     "'1011100", f"k{i}", 30.0, 0.80, ""])    # preço final -20%: perda real
    dfv = pd.DataFrame(vals, columns=["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin",
                                       "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica",
                                       "Dias_Ate_Alvo", "Preco_Final", "Estado_RL_Continuo"])
    m, v, r, _ = calibrar_hibrido(dfv, None, df)
    assert "Egg" in m and v["Egg"] is False     # só falhas, mas taxa baixa => não 'confiante' => sem veto manual
    conc = preparar_concluidos(dfv)
    assert conc["Recompensa"].mean() < -1.0
    mapa_m, mapa_v, mapa_r = {}, {}, {}
    # com 10 acertos/falhas mistos: taxa baixa => não 'confiante' => sem veto; força caso confiante+EV<0:
    ac = dfv.copy()
    ac["Status"] = "🏆 ACERTOU"
    ac["Preco_Final"] = 0.90          # status "acertou" mas fechou -10%: EV negativo com taxa alta
    calibrar_por_taxa_confianca(preparar_concluidos(ac), mapa_m, mapa_r, mapa_v)
    assert mapa_v["Egg"] is True, f"veto manual deveria ativar (EV<0 com taxa alta): {mapa_m, mapa_v}"
    print("✓ EV contínuo habilita veto manual")

    # ---- 6. veto RL lê N_COMPRA da Q-table -------------------------------------
    q = {"1011100": {"COMPRA": -0.47, "VENDA": 0.0, "N_COMPRA": 40, "N_VENDA": 0}}
    vet, _ = calcular_veto_rl("1011100", q)
    assert vet is True        # n=40 -> threshold -0.45; -0.47 < -0.45 => vetado
    q2 = {"1011100": {"COMPRA": -0.47, "VENDA": 0.0, "N_COMPRA": 2, "N_VENDA": 0}}
    vet2, _ = calcular_veto_rl("1011100", q2)
    assert vet2 is False      # n=2 -> threshold -0.5 -> não veta
    print("✓ veto RL por N_COMPRA (n=40: vetado =", vet, "| n=2: vetado =", vet2, ")")

    # ---- 7. item em carteira sem previsão ainda é decidido ---------------------
    row = cache_ind["Egg"].iloc[-1].copy()
    row["Price"] = row["Media_Macro"] * 5            # bem acima da banda de saída
    res = decidir_item("Egg", row, 0.0, 1000, 1000, row["Media_Macro"], 5, 2.0, False, {}, {},
                       df_audit_mem=None, d_item_hist=cache_ind["Egg"])
    assert res and "VENDER" in res["acao"] and res["gera_sinal"], res["acao"]
    print("✓ item em carteira sem previsão -> ", res["acao"])

    # ---- 8. dedup: STOP tem chave própria --------------------------------------
    dfv2 = dfv.iloc[:1].copy()
    dfv2["Tipo"], dfv2["Chave_Unica"], dfv2["Data_Sinal"] = "VENDA", "Egg_VENDA_23102026_1100_STOP", "23/10/2026 11:00"
    mapa, ign = montar_mapa_ultimo_sinal(dfv2)
    assert "Egg_STOP" in mapa and "Egg_VENDA" not in mapa and ign == 0
    assert pode_gerar_sinal(None, 1, AGORA, False)
    assert not pode_gerar_sinal(mapa["Egg_STOP"], 1.0, AGORA, True)           # <4h
    print("✓ dedup com chave de STOP")

    # ---- 9. validação de VENDA: zeros não contam como 'preço no alvo' -----------
    sinal = pd.DataFrame([[1, "VENDA", (AGORA - timedelta(days=13)).strftime("%d/%m/%Y %H:%M"), "Zeroso", 0.5, 0.01, 0.5,
                           "Aguardando", 0, "'1011100", "Zeroso_VENDA_x", "", "", ""]], columns=dfv.columns)
    conc_idx = atualizar_sinais_abertos(sinal, df, AGORA)
    assert not conc_idx or "ACERTOU" not in sinal.iloc[0]["Status"], "zeros viraram falso 'ACERTOU (FUNDO)'"
    print("✓ validação ignora preço 0 ->", sinal.iloc[0]["Status"])

    # ---- 10. FIFO ordena por data e devolve data do lote -----------------------
    tl = pd.DataFrame([["Compra (x)", "Egg", "10", "-1,0", "p", "02/10/2026 10:00"],
                       ["Compra (x)", "Egg", "10", "-2,0", "p", "01/10/2026 10:00"]],
                      columns=["Tipo", "Nome do Item", "Qtd", "SFL", "Parceiro", "Data"])   # fora de ordem
    custo, datas = calcular_fifo_hibrido(tl, {"Egg": 15})
    assert abs(custo["Egg"] - (10 * 0.1 + 5 * 0.2) / 15) < 1e-9, custo        # 10 do lote novo + 5 do antigo
    assert datas["Egg"] == pd.Timestamp("2026-10-01 10:00")
    print("✓ FIFO por data:", custo, datas)
    # ---- 11. episódios: 10 sinais de uma condição persistente = 1 episódio ------
    seq = dfv.copy()
    seq["Data_Sinal"] = [(datetime(2026, 8, 1, 10, 0) + timedelta(days=i)).strftime("%d/%m/%Y %H:%M") for i in range(10)]
    seq["Status"], seq["Preco_Final"] = "🏆 ACERTOU", 0.90          # 24h entre sinais, mesmo item e tipo
    conc_seq = preparar_concluidos(seq)
    assert conc_seq["Episodio"].nunique() == 1, conc_seq["Episodio"].tolist()
    mm, mr, mv = {}, {}, {}
    usados = calibrar_por_taxa_confianca(conc_seq, mm, mr, mv)
    assert usados["Egg"] == 1, usados
    assert mv["Egg"] is False and mm["Egg"] == 2.2, "1 episódio não pode virar 'confiante' (N>=8)"
    # os mesmos 10 sinais espaçados em 5 dias = 10 episódios => confiante, EV<0 => veto (teste 5)
    seq5 = dfv.copy()
    seq5["Status"], seq5["Preco_Final"] = "🏆 ACERTOU", 0.90
    assert preparar_concluidos(seq5)["Episodio"].nunique() == 10

    # limite do gap (48h) e separação por tipo/STOP/item
    tri = dfv.iloc[:5].copy()
    tri["Data_Sinal"] = ["01/08/2026 10:00", "02/08/2026 10:00",   # +24h: mesmo episódio
                         "03/08/2026 11:00",                        # +25h: mesmo episódio
                         "06/08/2026 12:00",                        # +73h (> 48h): novo episódio
                         "01/08/2026 10:00"]                        # STOP no mesmo instante do 1º: episódio próprio
    tri["Chave_Unica"] = ["a", "b", "c", "d", "e_STOP"]
    tri["Status"] = "🏆 ACERTOU"
    ep = atribuir_episodios(tri)["Episodio"].tolist()
    assert ep[0] == ep[1] == ep[2] and ep[3] != ep[0] and ep[4] not in ep[:4], ep
    outro_item = tri.iloc[:1].copy(); outro_item["Item"] = "Wheat"
    assert atribuir_episodios(pd.concat([tri.iloc[:1], outro_item]))["Episodio"].nunique() == 2
    # alocação: item com apenas 1 episódio (mesmo com 10 sinais) fica de fora
    assert calcular_alocacao_recursos(conc_seq, min_sinais=3).empty
    print("✓ episódios: 10 sinais persistentes = 1 episódio; gap 48h; STOP e item separados")

    # ---- 12. item congelado (evento): banda degenerada não pode gerar COMPRA -----
    d = cache_ind["Egg"].copy()
    d["Price"], d["Media_Macro"], d["Desvio_Macro"] = 0.05, 0.05, 0.0
    res = decidir_item("Egg", d.iloc[-1], -7.5, 0, 0, 0.05, None, 2.0, False, {}, {}, d_item_hist=d)
    assert res and not res["gera_sinal"] and "COMPRAR" not in res["acao"], res["acao"]
    assert "CONGELADO" in res["acao"], res["acao"]
    print("✓ item congelado não gera COMPRA ->", res["acao"])
    # banda estreita mas preço vivo (não congelado): compra sem margem após a taxa também é barrada
    d2 = cache_ind["Egg"].copy()
    m = float(d2["Media_Macro"].iloc[-1])
    d2.loc[d2.index[-1], ["Desvio_Macro", "Price"]] = [m * 0.01, m * 0.97]     # alvo_in = 0.98m, alvo_out = 1.02m
    res2 = decidir_item("Egg", d2.iloc[-1], 0.0, 0, 0, m, None, 2.0, False, {}, {}, d_item_hist=d2)
    assert res2 and "BANDA" in res2["acao"] and not res2["gera_sinal"], res2 and res2["acao"]
    print("✓ banda < taxa não gera COMPRA ->", res2["acao"])
    # sanidade: banda larga e preço vivo continua gerando COMPRA
    d3 = cache_ind["Egg"].copy()
    d3.loc[d3.index[-1], ["Desvio_Macro", "Price"]] = [m * 0.3, m * 0.35]      # alvo_in = 0.4m, alvo_out = 1.6m
    res3 = decidir_item("Egg", d3.iloc[-1], 5.0, 0, 0, m, None, 2.0, False, {}, {}, d_item_hist=d3)
    assert res3 and "COMPRAR" in res3["acao"] and res3["gera_sinal"], res3 and res3["acao"]
    print("✓ banda larga continua gerando COMPRA ->", res3["acao"])
    print("\nTODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
