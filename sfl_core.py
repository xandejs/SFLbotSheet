"""
sfl_core.py — Lógica PURA do bot SFL V2 (sem gspread, sem rede, sem arquivos).

Tudo aqui recebe e devolve DataFrames/dicts, então pode ser testado
isoladamente (test_core.py) e reaproveitado pelo harness de backtest.
O bot_v2.py cuida de todo o I/O (planilhas, trava, cache em disco, Telegram).

MUDANÇAS em relação ao bot_v2.py anterior:
  1. Preço <= 0 é descartado na entrada (preparar_df_bruto) + dedup por
     (Timestamp, Item). O alvo de 12 dias agora é por TEMPO (merge_asof),
     não por 288 linhas, então buracos/zeros no RecLog não deslocam o horizonte.
  2. Modelo pooled prevê o RETORNO (log(Preco_Futuro/Price)) com features
     relativas (retornos dos lags, z-score, volatilidade relativa, distância
     à média, índice/margem de insumo relativos à própria média).
  3. EV e recompensa são a mesma coisa: recompensa contínua pelo ROI real
     (recompensa_sinal). Calibração, Alocação e RL usam a mesma função.
  4. Veto do RL usa N_COMPRA da própria Q-table (não a contagem em
     Validacao_Sinais).
  5. decidir_item() / pode_gerar_sinal() / atualizar_sinais_abertos() foram
     extraídos como funções puras (antes eram um script de ~400 linhas).
  6. FIFO ordena por data e devolve também a data do lote mais antigo em
     custódia (dias em custódia corretos).
  7. Dedup do STOP tem chave própria (item_STOP).
"""

import math
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta, timezone

# ==============================================================================
# CONSTANTES
# ==============================================================================
REGISTROS_POR_DIA = 24
TAXA_MARKETPLACE = 0.125          # ⚠️ ARQUITETURA_1_1.md cita 10% — confirmar o valor real do jogo
LUCRO_REAL_DESEJADO = 0.07
DIAS_SWING = 12
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA   # só usado p/ posicionar o holdout; o alvo é por tempo
TOLERANCIA_ALVO_FUTURO = pd.Timedelta("3h")
DIAS_MAX_MONITORAMENTO = 30

TOLERANCIA_SUJEIRA = 0.15
POSICAO_MINIMA_SFL = 5.0
ALPHA_MINIMO = 0.05

CALIBRAR_SO_COMPRA = True          # mapas de banda/veto governam compras: calibra só com sinais de COMPRA
EPISODIO_GAP_H = 48                # sinais do mesmo item+tipo com <= 48h entre si = mesmo episódio (condição persistente)
MAX_IDADE_PRECO_ITEM_H = 6.0       # item cujo último preço > 6h atrás não gera sinal (ex.: sem oferta/preço 0)
LUCRO_MIN_BANDA = 0.0              # COMPRAR exige que vender em alvo_out (já sem taxa) ao menos empate; suba até LUCRO_REAL_DESEJADO p/ ser mais estrito
HORAS_PRECO_CONGELADO = 72         # sem nenhuma variação nesse período = item sem mercado (ex.: item de evento)

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

HEADERS_AUDIT = [
    "Timestamp", "Item", "MAE", "RMSE",
    "Acerto_Direcao_%", "N_Treino", "N_Teste", "ROI_Previsto_%"
]

# Mapa de receita explícito (item -> insumos). Validar cada vínculo com
# df.corr() com lag sobre o RecLog antes de expandir.
MAPA_RECEITA = {
    "Egg":         ["Wheat", "Corn", "Barley"],
    "Feather":     ["Wheat", "Corn", "Barley"],
    "Milk":        ["Wheat", "Corn", "Barley"],
    "Leather":     ["Wheat", "Corn", "Barley"],
    "Wool":        ["Wheat", "Corn", "Barley"],
    "Merino Wool": ["Wheat", "Corn", "Barley"],
    "Stone":       ["Wood"],
    "Iron":        ["Wood", "Stone"],
    "Gold":        ["Wood", "Iron"],
    "Crimstone":   ["Wood", "Gold"],
}

# Features do modelo pooled — TODAS relativas (comparáveis entre itens de 0,001 a 50 SFL)
FEATURES_MODELO = [
    "Ret_L1", "Ret_L2", "Ret_L3", "Z_Macro", "Volat_Rel", "Dist_Media",
    "RSI_Micro", "Season_Num", "Hora_Dia", "Dia_Semana",
    "Indice_Dist", "Margem_Dist",
]
COLS_MODELO = FEATURES_MODELO + ["Item_Cat"]
FEATURES_NUCLEO = ["RSI_Micro", "Media_Macro", "Desvio_Macro", "Lag_3"]   # linha só é usável com isto
ACOES_SINAL = ["🟢 COMPRAR", "🔴 VENDER", "🔴 VENDER (Lucro Disponível)", "⚠️ VENDER (Stop Loss)"]


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================
def limpar_numero(texto):
    if texto is None or texto == "":
        return 0.0
    s = str(texto).strip()
    if not s:
        return 0.0
    if "." in s and "," in s:
        s = s.replace(".", "")
    s = s.replace(",", ".")
    if s.count(".") > 1:
        partes = s.split(".")
        s = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return float(s)
    except Exception:
        return 0.0


def obter_agora_brt():
    """Datetime atual em BRT, sem tzinfo (compat pandas)."""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).replace(tzinfo=None)


def parse_data_sinal_robusta(texto):
    if texto is None or str(texto).strip() == "":
        return pd.NaT
    texto = str(texto).strip()
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y"):
        dt = pd.to_datetime(texto, format=fmt, errors="coerce")
        if pd.notnull(dt):
            return dt
    return pd.to_datetime(texto, dayfirst=True, errors="coerce")


def _parse_timestamps(serie):
    """
    Parse rápido com formato fixo (dd/mm/aaaa HH:MM:SS); só o que falhar cai no
    parser genérico (outros formatos / cabeçalho viram NaT). Evita o fallback
    elemento-a-elemento do pandas em ~180 mil linhas.
    """
    ts = pd.to_datetime(serie, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    falhou = ts.isna() & serie.astype(str).str.strip().ne("")
    if falhou.any():
        ts.loc[falhou] = pd.to_datetime(serie[falhou], dayfirst=True, errors="coerce", format="mixed")
    return ts


def preparar_df_bruto(linhas):
    """
    RecLog cru (lista de listas) -> DataFrame limpo [Timestamp, Item, Price, Season].
      - Não depende de haver cabeçalho (linha com data inválida é descartada).
      - Preço <= 0 (= "sem oferta") é descartado: contaminava média, RSI,
        detector de regime e a validação de VENDA ("preço <= alvo").
      - Duplicatas (Timestamp, Item) removidas (fica a última).
    Retorna (df, stats).
    """
    pad = lambda l: (list(l) + ["", "", "", ""])[:4]
    df = pd.DataFrame([pad(l) for l in linhas], columns=["Timestamp", "Item", "Price", "Season"])
    n_total = len(df)
    df["Timestamp"] = _parse_timestamps(df["Timestamp"])
    n_nat = int(df["Timestamp"].isna().sum())
    df = df.dropna(subset=["Timestamp"])
    df["Item"] = df["Item"].astype(str).str.strip()
    df["Price"] = df["Price"].apply(limpar_numero)
    n_zero = int((df["Price"] <= 0).sum())
    df = df[(df["Price"] > 0) & (df["Item"] != "")]
    n_antes = len(df)
    df = df.drop_duplicates(["Timestamp", "Item"], keep="last")
    n_dup = n_antes - len(df)
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    return df, {"total": n_total, "data_invalida": n_nat, "preco_zero": n_zero, "duplicadas": n_dup, "uteis": len(df)}


# ==============================================================================
# FIFO
# ==============================================================================
def _datas_trade(df):
    col = next((c for c in ("Timestamp", "Data", "Data/Hora", "Date") if c in df.columns), None)
    if col is None:
        return None
    return pd.to_datetime(df[col], dayfirst=True, errors="coerce")


def calcular_fifo_hibrido(df_trade_raw, inventario_dict):
    """
    Retorna (custo_medio_map, data_lote_mais_antigo_map).
    Ordena as compras por data (o TradeLog não é estritamente cronológico) e
    consome do lote mais novo para o mais antigo até cobrir o saldo real.
    """
    try:
        df_log = df_trade_raw.copy()
        df_log["Qtd"] = df_log["Qtd"].apply(limpar_numero)
        df_log["SFL_Total"] = df_log["SFL"].apply(limpar_numero).abs()
        datas = _datas_trade(df_log)
        tem_data = datas is not None
        if tem_data:
            df_log["_dt"] = datas
        df_compras = df_log[df_log["Tipo"].astype(str).str.contains("Compra", case=False, na=False)].copy()
        if tem_data:
            df_compras = df_compras.sort_values("_dt", kind="stable", na_position="first")

        fifo_map, data_map = {}, {}
        for item, saldo_real in inventario_dict.items():
            if saldo_real <= 0:
                continue
            compras_item = df_compras[df_compras["Nome do Item"] == item].iloc[::-1]
            qtd_acumulada, custo_total, mais_antiga = 0.0, 0.0, None
            for _, row in compras_item.iterrows():
                falta = saldo_real - qtd_acumulada
                if falta <= 0:
                    break
                if row["Qtd"] <= 0:
                    continue
                usar = min(row["Qtd"], falta)
                custo_total += usar * (row["SFL_Total"] / row["Qtd"])
                qtd_acumulada += usar
                if tem_data and pd.notnull(row["_dt"]):
                    mais_antiga = row["_dt"]
            fifo_map[item] = custo_total / qtd_acumulada if qtd_acumulada > 0 else 0.0
            if mais_antiga is not None:
                data_map[item] = mais_antiga
        return fifo_map, data_map
    except Exception as e:
        print(f"Erro FIFO: {e}")
        return {}, {}


# ==============================================================================
# RECOMPENSA / EV / CALIBRAÇÃO
# ==============================================================================
def calcular_roi_real(tipo, preco_sinal, preco_final):
    if preco_sinal <= 0:
        return 0.0
    if tipo == "COMPRA":
        venda_liquida = preco_final * (1 - TAXA_MARKETPLACE)
        return ((venda_liquida - preco_sinal) / preco_sinal) * 100
    return ((preco_sinal - preco_final) / preco_sinal) * 100


def pontos_status(st):
    st = str(st)
    if "🏆" in st:
        return 1.0
    if "💰" in st:
        return 0.5
    return 0.0


def recompensa_sinal(tipo, status, preco_sinal, preco_final):
    """
    Recompensa contínua em [-2, 2]: 1.0 = ROI líquido igual ao lucro desejado.
    Usada pelo Q-learning E pelo EV da calibração/alocação (antes o EV usava
    pontos 1/0.5/0, que nunca é negativo — os ramos de veto manual eram
    inalcançáveis).
    """
    if preco_sinal > 0 and preco_final > 0:
        roi = calcular_roi_real(tipo, preco_sinal, preco_final)
        return max(-2.0, min(2.0, roi / (LUCRO_REAL_DESEJADO * 100)))
    st = str(status)
    if "🏆" in st:
        return 1.0
    if "💰" in st:
        return 0.5
    return -1.0


def atribuir_episodios(df_val):
    """
    Adiciona a coluna 'Episodio' (int). O dedup libera um novo sinal a cada ~24h
    enquanto a condição persiste; 8 sinais assim são UM episódio (as janelas de
    12 dias se sobrepõem, então os resultados não são independentes). Regra:
    mesmo Item + mesmo tipo (COMPRA / VENDA / STOP) com no máximo EPISODIO_GAP_H
    horas entre sinais consecutivos. Data inválida vira episódio próprio.
    Deve receber TODOS os sinais (inclusive os abertos), para o episódio não
    "quebrar" por falta de vizinhos.
    """
    df = df_val.copy()
    if df.empty:
        df["Episodio"] = pd.Series(dtype=int)
        return df
    item = df["Item"].astype(str).str.strip()
    chave = df["Chave_Unica"].astype(str) if "Chave_Unica" in df.columns else pd.Series("", index=df.index)
    tipo = pd.Series(np.where(chave.str.contains("_STOP", regex=False), "STOP",
                              df["Tipo"].astype(str).str.strip().str.upper()), index=df.index)
    dt = pd.to_datetime(df["Data_Sinal"].apply(parse_data_sinal_robusta), errors="coerce")
    tmp = pd.DataFrame({"g": item + "|" + tipo, "dt": dt}).sort_values(["g", "dt"], kind="stable")
    novo = ((tmp["g"] != tmp["g"].shift()) | tmp["dt"].isna() | tmp["dt"].shift().isna()
            | ((tmp["dt"] - tmp["dt"].shift()) > pd.Timedelta(hours=EPISODIO_GAP_H)))
    tmp["ep"] = novo.cumsum()
    df["Episodio"] = tmp["ep"].reindex(df.index).astype(int)
    return df


def _serie_episodio(df):
    return df["Episodio"] if "Episodio" in df.columns else pd.Series(range(len(df)), index=df.index)


def resumo_episodios(grupo):
    """(n_episodios, n_acertos, soma_recompensas): cada episódio pesa 1 (média dos seus sinais)."""
    ep = _serie_episodio(grupo)
    pts = grupo["Pts"].groupby(ep).mean()
    rec = grupo["Recompensa"].groupby(ep).mean()
    return len(pts), int((pts >= 0.5).sum()), float(rec.sum())


def preparar_concluidos(df_val):
    """
    Sinais concluídos com Pts, Recompensa, ROI_Realizado e Episodio.
    DataFrame vazio se não houver. Os episódios são calculados sobre TODOS os sinais.
    """
    if df_val is None or df_val.empty:
        return pd.DataFrame()
    df_all = atribuir_episodios(df_val)
    df = df_all[df_all["Status"].astype(str).str.contains("🏆|❌|💰", na=False)].copy()
    if df.empty:
        return pd.DataFrame()
    df["Pts"] = df["Status"].apply(pontos_status)
    df["Preco_Sinal_N"] = df["Preco_Sinal"].apply(limpar_numero)
    df["Preco_Final_N"] = df["Preco_Final"].apply(limpar_numero)
    df["Dias_Ate_Alvo_N"] = pd.to_numeric(df["Dias_Ate_Alvo"], errors="coerce")
    df["Recompensa"] = [
        recompensa_sinal(t, s, ps, pf)
        for t, s, ps, pf in zip(df["Tipo"], df["Status"], df["Preco_Sinal_N"], df["Preco_Final_N"])
    ]

    def roi_realizado(r):
        if r["Preco_Final_N"] > 0:
            return calcular_roi_real(r["Tipo"], r["Preco_Sinal_N"], r["Preco_Final_N"])
        if r["Preco_Sinal_N"] <= 0:
            return 0.0
        ref_n = limpar_numero(r["Referencia_Fin"])
        if r["Tipo"] == "COMPRA":
            return (((ref_n * (1 - TAXA_MARKETPLACE)) - r["Preco_Sinal_N"]) / r["Preco_Sinal_N"]) * 100 * r["Pts"]
        return ((r["Preco_Sinal_N"] - ref_n) / r["Preco_Sinal_N"]) * 100 * r["Pts"]

    df["ROI_Realizado"] = df.apply(roi_realizado, axis=1)
    return df


def calcular_taxa_confianca(n_total, n_acertos, soma_recompensas,
                            prior_a=1.0, prior_b=1.0, n_minimo=8):
    """taxa_efetiva (posterior Beta), confiante (IC inferior), EV = recompensa média por sinal."""
    if n_total == 0:
        return 0.5, False, 0.0
    a = prior_a + n_acertos
    b = prior_b + (n_total - n_acertos)
    taxa_efetiva = a / (a + b)
    ev_esperado = soma_recompensas / n_total
    variancia = (a * b) / ((a + b) ** 2 * (a + b + 1))
    ic_inferior = max(0, taxa_efetiva - 1.96 * math.sqrt(variancia))
    confiante = (n_total >= n_minimo) and (ic_inferior > 0.35)
    return taxa_efetiva, confiante, ev_esperado


def calcular_prior_empirico(df_conc, forca_prior=4.0):
    if df_conc is None or df_conc.empty:
        return 1.0, 1.0
    taxa_global = (df_conc["Pts"].groupby(_serie_episodio(df_conc)).mean() >= 0.5).mean()   # por episódio
    return max(forca_prior * taxa_global, 0.5), max(forca_prior * (1 - taxa_global), 0.5)


def calibrar_por_taxa_confianca(df_conc, mapa_mult, mapa_roi, mapa_veto, so_compra=CALIBRAR_SO_COMPRA):
    """Preenche os mapas. Retorna {item: n_sinais_usados}."""
    usados = {}
    if df_conc is None or df_conc.empty:
        return usados
    prior_a, prior_b = calcular_prior_empirico(df_conc)
    base = df_conc[df_conc["Tipo"] == "COMPRA"] if so_compra else df_conc

    for item_c, grupo in base.groupby("Item"):
        n, n_acertos, soma_rec = resumo_episodios(grupo)      # N = episódios, não sinais
        _, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_rec, prior_a=prior_a, prior_b=prior_b)
        usados[item_c] = n

        if not confiante:
            mapa_mult[item_c], mapa_roi[item_c], mapa_veto[item_c] = 2.2, -2.0, False
        elif ev >= 0.4:
            mapa_mult[item_c], mapa_roi[item_c], mapa_veto[item_c] = 2.0, 0.0, False
        elif ev >= 0.1:
            mapa_mult[item_c], mapa_roi[item_c], mapa_veto[item_c] = 2.4, -3.0, False
        elif ev >= -0.1:
            mapa_mult[item_c], mapa_roi[item_c], mapa_veto[item_c] = 2.4, -5.0, True
        else:
            mapa_mult[item_c], mapa_roi[item_c], mapa_veto[item_c] = 2.8, -10.0, True
    return usados


def normalizar_audit(df_audit):
    if df_audit is None or df_audit.empty:
        return pd.DataFrame(columns=HEADERS_AUDIT)
    df = df_audit.copy()
    df["MAE"] = df["MAE"].apply(limpar_numero)
    df["Acerto_Direcao_%"] = df["Acerto_Direcao_%"].apply(limpar_numero)
    return df


def calibrar_hibrido(df_val, df_audit, df_bruto):
    """
    FONTE 1: sinais reais (EV por recompensa contínua + taxa Bayesiana).
    FONTE 2: MAE do XGB_Audit como proxy para itens sem sinais concluídos.
    Retorna (mapa_mult, mapa_veto, mapa_roi, n_proxy).
    """
    mapa_mult, mapa_veto, mapa_roi, confianca = {}, {}, {}, {}
    df_conc = preparar_concluidos(df_val)
    if not df_conc.empty:
        usados = calibrar_por_taxa_confianca(df_conc, mapa_mult, mapa_roi, mapa_veto)
        for it, n in usados.items():
            confianca[it] = min(1.0, n / 10.0)
        n_ep = int(df_conc["Episodio"].nunique()) if "Episodio" in df_conc.columns else len(df_conc)
        print(f"📈 Calibração Bayesiana: {len(usados)} itens | {len(df_conc)} sinais concluídos em {n_ep} episódios.")
    else:
        print("ℹ️  Calibração: nenhum sinal concluído ainda — fase bootstrap.")

    n_proxy = 0
    df_a = normalizar_audit(df_audit)
    df_a = df_a[df_a["MAE"] > 0]
    if not df_a.empty:
        medias = df_bruto.groupby("Item")["Price"].apply(lambda s: s.tail(240).mean()).to_dict()
        for item_a, grupo in df_a.groupby("Item"):
            ultimos = grupo.tail(72)
            mae_medio = ultimos["MAE"].mean()
            acerto_medio = ultimos["Acerto_Direcao_%"].mean()
            preco_medio = medias.get(item_a, 0.0)
            mae_rel = (mae_medio / preco_medio) if preco_medio and preco_medio > 0 else 0.15

            if item_a not in mapa_mult:
                if mae_rel < 0.05 and acerto_medio >= 60:
                    mapa_mult[item_a], mapa_roi[item_a] = 2.0, 0.0
                elif mae_rel < 0.12:
                    mapa_mult[item_a], mapa_roi[item_a] = 2.4, -3.0
                else:
                    mapa_mult[item_a], mapa_roi[item_a] = 2.8, -6.0
                mapa_veto[item_a] = False
                confianca[item_a] = 0.0
                n_proxy += 1
            else:
                peso_xgb = 1.0 - confianca.get(item_a, 1.0)
                if mae_rel > 0.12 and peso_xgb > 0:
                    mapa_mult[item_a] = round(min(mapa_mult[item_a] + 0.3 * peso_xgb, 3.0), 2)
        print(f"📐 Calibração por XGB_Audit: {n_proxy} itens via proxy MAE.")
    return mapa_mult, mapa_veto, mapa_roi, n_proxy


def calcular_alocacao_recursos(df_conc, dias_swing=12, min_sinais=3):
    colunas = ["Item", "Taxa_Acerto_%", "EV_Sinal", "ROI_Medio_%",
               "Dias_Medio", "Retorno_Anualizado_%", "Score", "Peso_Alocacao_%"]
    if df_conc is None or df_conc.empty:
        return pd.DataFrame(columns=colunas)

    prior_a, prior_b = calcular_prior_empirico(df_conc)
    linhas = []
    for item_a, grupo in df_conc.groupby("Item"):
        n, n_acertos, soma_rec = resumo_episodios(grupo)      # min_sinais agora conta EPISÓDIOS
        if n < min_sinais:
            continue
        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_rec, prior_a=prior_a, prior_b=prior_b)

        ep = _serie_episodio(grupo)
        dias_validos = grupo["Dias_Ate_Alvo_N"].groupby(ep).mean().dropna()
        dias_medio = max(dias_validos.mean() if not dias_validos.empty else float(dias_swing), 1.0)
        roi_medio_pct = grupo["ROI_Realizado"].groupby(ep).mean().mean()
        try:
            ret_anual = ((1 + roi_medio_pct / 100) ** (365 / dias_medio) - 1) * 100
        except (OverflowError, ZeroDivisionError, ValueError):
            ret_anual = 0.0
        ret_anual = min(ret_anual, 2000.0)
        score = ret_anual * (taxa_ef if confiante else taxa_ef * 0.5) if ev > 0 else 0.0

        linhas.append({
            "Item": item_a, "Taxa_Acerto_%": round(taxa_ef * 100, 1), "EV_Sinal": round(ev, 3),
            "ROI_Medio_%": round(roi_medio_pct, 2), "Dias_Medio": round(dias_medio, 2),
            "Retorno_Anualizado_%": round(ret_anual, 1), "Score": round(score, 2),
        })

    df_score = pd.DataFrame(linhas)
    if df_score.empty:
        return pd.DataFrame(columns=colunas)
    soma = df_score[df_score["Score"] > 0]["Score"].sum()
    df_score["Peso_Alocacao_%"] = df_score["Score"].apply(
        lambda s: round((s / soma) * 100, 2) if (soma > 0 and s > 0) else 0.0)
    return df_score.sort_values("Peso_Alocacao_%", ascending=False).reset_index(drop=True)


# ==============================================================================
# VALIDAÇÃO DE SINAIS
# ==============================================================================
def atualizar_sinais_abertos(df_val, df_bruto, agora):
    """
    Atualiza IN-PLACE os sinais 'Aguardando' de df_val. df_bruto precisa ter
    Timestamp datetime e Price > 0. Retorna a lista de índices concluídos.
    """
    concluidos = []
    if df_val is None or df_val.empty:
        return concluidos
    # pandas >= 3: colunas lidas do Sheets são dtype 'str' e rejeitam números em .at[].
    # Converte IN-PLACE para object (o chamador segue usando o mesmo DataFrame).
    for c in df_val.columns:
        df_val[c] = df_val[c].astype(object)
    hist = {k: g for k, g in df_bruto.groupby("Item")}
    vazio = df_bruto.iloc[0:0]

    for idx, row_val in df_val.iterrows():
        if row_val["Status"] not in ("Aguardando", ""):
            continue
        item_v, tipo_v = row_val["Item"], row_val["Tipo"]
        data_sinal_v = parse_data_sinal_robusta(row_val["Data_Sinal"])
        preco_sinal_v = limpar_numero(row_val["Preco_Sinal"])
        ref_fin_v = limpar_numero(row_val["Referencia_Fin"])

        if pd.isnull(data_sinal_v):
            df_val.at[idx, "Dias_Passados"] = 0
            df_val.at[idx, "Status"] = "⚠️ ERRO_DATA (Verificar Manualmente)"
            continue

        dias_p = (agora - data_sinal_v).days
        d_hist = hist.get(item_v, vazio)
        df_pos = d_hist[d_hist["Timestamp"] >= data_sinal_v]
        is_stop = "_STOP" in str(row_val["Chave_Unica"])

        if tipo_v == "COMPRA" or is_stop:
            preco_mon = df_pos["Price"].max() if not df_pos.empty else preco_sinal_v
        else:
            preco_mon = df_pos["Price"].min() if not df_pos.empty else preco_sinal_v
        max_pos = df_pos["Price"].max() if not df_pos.empty else preco_sinal_v
        horas_no_alvo = int((df_pos["Price"] >= ref_fin_v).sum()) if not df_pos.empty else 0
        horas_venda = int((df_pos["Price"] <= ref_fin_v).sum()) if not df_pos.empty else 0

        dias_ate_alvo = None
        if not df_pos.empty:
            if tipo_v == "COMPRA":
                qualifica = df_pos["Price"] >= ref_fin_v
            elif tipo_v == "VENDA" and not is_stop:
                qualifica = df_pos["Price"] <= ref_fin_v
            else:
                qualifica = None
            if qualifica is not None:
                cum = qualifica.cumsum()
                atingiu = cum[cum >= 3]
                if not atingiu.empty:
                    ts = df_pos.loc[atingiu.index[0], "Timestamp"]
                    dias_ate_alvo = round((ts - data_sinal_v).total_seconds() / 86400, 2)

        status_v = "Aguardando"
        if tipo_v == "COMPRA":
            if dias_p >= DIAS_SWING:
                piso = preco_sinal_v * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
                if horas_no_alvo >= 3:
                    status_v = "🏆 ACERTOU"
                elif horas_no_alvo >= 1 or max_pos >= piso:
                    if dias_p >= DIAS_MAX_MONITORAMENTO:
                        status_v = "💰 LUCRO PARCIAL"
                elif dias_p >= DIAS_MAX_MONITORAMENTO:
                    status_v = "❌ FALHOU"
        elif tipo_v == "VENDA":
            if is_stop:
                if dias_p >= DIAS_SWING:
                    status_v = "🏆 ACERTOU (STOP PROTETIVO)" if max_pos <= ref_fin_v else "❌ FALHOU (STOP PREMATURO)"
            elif dias_p >= DIAS_SWING:
                if horas_venda >= 3:
                    status_v = "🏆 ACERTOU (FUNDO)"
                elif horas_venda >= 1:
                    if dias_p >= DIAS_MAX_MONITORAMENTO:
                        status_v = "💰 LUCRO PARCIAL (SPIKE)"
                elif dias_p >= DIAS_MAX_MONITORAMENTO:
                    status_v = "❌ FALHOU"

        df_val.at[idx, "Dias_Passados"] = int(dias_p)
        df_val.at[idx, "Monitoramento_12d"] = round(preco_mon, 4)
        df_val.at[idx, "Status"] = status_v
        if dias_ate_alvo is None and status_v != "Aguardando":
            dias_ate_alvo = float(dias_p)
        df_val.at[idx, "Dias_Ate_Alvo"] = dias_ate_alvo if dias_ate_alvo is not None else ""
        if status_v != "Aguardando":
            df_val.at[idx, "Preco_Final"] = round(df_pos["Price"].iloc[-1], 4) if not df_pos.empty else round(preco_sinal_v, 4)
        if any(s in status_v for s in ("🏆", "❌", "💰")):
            concluidos.append(idx)
    return concluidos


def normalizar_df_val(df_val):
    df = df_val.copy()
    df["ID_Sinal"] = pd.to_numeric(df["ID_Sinal"].apply(limpar_numero), errors="coerce").fillna(0).astype(int)
    df["Preco_Sinal"] = df["Preco_Sinal"].apply(limpar_numero)
    df["Referencia_Fin"] = df["Referencia_Fin"].apply(limpar_numero)
    df["Monitoramento_12d"] = df["Monitoramento_12d"].apply(limpar_numero)
    df["Dias_Passados"] = pd.to_numeric(df["Dias_Passados"].apply(limpar_numero), errors="coerce").fillna(0).astype(int)
    vazio = ("", "nan", "None")
    df["Dias_Ate_Alvo"] = df["Dias_Ate_Alvo"].apply(lambda x: round(limpar_numero(x), 2) if str(x).strip() not in vazio else "")
    df["Preco_Final"] = df["Preco_Final"].apply(lambda x: round(limpar_numero(x), 4) if str(x).strip() not in vazio else "")
    df["Estado_RL"] = df["Estado_RL"].astype(str).apply(lambda x: x.strip().replace("'", ""))
    df["Estado_RL"] = df["Estado_RL"].apply(lambda x: f"'{x}" if x != "" else "")
    return df


# ==============================================================================
# Q-LEARNING
# ==============================================================================
def aplicar_reforco(df_val, indices, q_table, q_table_item):
    """Atualiza q_table/q_table_item in-place com os sinais concluídos. Retorna nº processado."""
    if not indices:
        return 0
    df_r = df_val.loc[indices].copy()
    df_r = df_r[df_r["Estado_RL"].astype(str).str.strip() != ""]
    for _, r in df_r.iterrows():
        st_rl = str(r["Estado_RL"]).strip().replace("'", "")
        acao = r["Tipo"]
        recompensa = recompensa_sinal(acao, r["Status"], limpar_numero(r["Preco_Sinal"]), limpar_numero(r["Preco_Final"]))
        chave_n = f"N_{acao}"

        e = q_table.setdefault(st_rl, {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0})
        n = e.get(chave_n, 0) + 1
        alpha = max(ALPHA_MINIMO, 1.0 / (1 + n))
        e[acao] = e.get(acao, 0.0) + alpha * (recompensa - e.get(acao, 0.0))
        e[chave_n] = n

        ci = f"{r['Item']}|{st_rl}"
        ei = q_table_item.setdefault(ci, {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0})
        ni = ei.get(chave_n, 0) + 1
        alpha_i = max(ALPHA_MINIMO, 1.0 / (1 + ni))
        ei[acao] = ei.get(acao, 0.0) + alpha_i * (recompensa - ei.get(acao, 0.0))
        ei[chave_n] = ni
    return len(df_r)


def calcular_veto_rl(estado_rl, q_table, q_compra_item=None, n_obs_item=0):
    """
    n_obs vem do N_COMPRA da própria Q-table global (antes vinha da contagem em
    Validacao_Sinais, o que dessincroniza quando só a Q-table é migrada).
    """
    dados = q_table.get(estado_rl)
    q_global = dados.get("COMPRA", 0.0) if dados else 0.0
    n_obs = int(dados.get("N_COMPRA", 0)) if dados else 0

    if dados is None and q_compra_item is None:
        return False, 0.0

    if q_compra_item is not None:
        peso_item = min(1.0, n_obs_item / 10.0)
        q_compra = peso_item * q_compra_item + (1 - peso_item) * q_global
    else:
        q_compra = q_global

    if n_obs < 5:
        threshold = -0.5
    elif n_obs < 20:
        threshold = -0.5 + (0.05 / 15) * (n_obs - 5)
    else:
        threshold = -0.45

    bonus = 10.0 if q_compra > 0.7 else (5.0 if q_compra > 0.5 else 0.0)
    return q_compra < threshold, bonus


# ==============================================================================
# RISCO
# ==============================================================================
def calcular_stop_adaptativo(item, preco_atual, roi_pred, df_audit_mem=None,
                             n_sigma=3.0, stop_minimo=-0.08, stop_maximo=-0.35):
    mae_rel = None
    if df_audit_mem is not None and not df_audit_mem.empty:
        audit_item = df_audit_mem[df_audit_mem["Item"] == item].tail(72)
        if not audit_item.empty:
            mae_vals = audit_item["MAE"].apply(limpar_numero)
            mae_vals = mae_vals[mae_vals > 0]
            if not mae_vals.empty and preco_atual > 0:
                mae_rel = mae_vals.mean() / preco_atual
    if mae_rel is not None and mae_rel > 0:
        threshold = max(stop_maximo, min(stop_minimo, -n_sigma * mae_rel))
    else:
        threshold = -0.20
    return roi_pred < threshold, threshold


def calcular_margem_atencao(alvo_in, alvo_out, fracao=0.15, minimo_abs=0.0001, maximo_pct=0.10):
    banda = max(0, alvo_out - alvo_in)
    margem = max(banda * fracao, minimo_abs)
    if alvo_in > 0:
        margem = min(margem, alvo_in * maximo_pct)
    return round(margem, 6)


def preco_congelado(d_item, horas=HORAS_PRECO_CONGELADO):
    """True se o preço não variou nas últimas `horas` (ex.: item de evento sem oferta)."""
    if d_item is None or len(d_item) < horas:
        return False
    p = d_item["Price"].iloc[-horas:]
    return (p.max() - p.min()) <= 1e-9 * max(p.max(), 1e-12)


def detectar_regime(d_item, janela_crash_h=24, limiar_queda=-0.12):
    if d_item is None or len(d_item) < janela_crash_h:
        return False, 0.0
    p_atual = d_item.iloc[-1]["Price"]
    p_24h = d_item.iloc[-janela_crash_h]["Price"]
    if p_24h == 0:
        return False, 0.0
    var = (p_atual - p_24h) / p_24h
    return var < limiar_queda, round(var * 100, 2)


# ==============================================================================
# ESTADO RL
# ==============================================================================
def construir_estado_rl_hibrido(row, origem, custo_medio, alvo_lucro, dias_custodia, roi_pred):
    """Estado discreto de 7 caracteres (idêntico ao da V1)."""
    rsi = row.get("RSI_Micro", 50)
    rsi_disc = 0 if rsi < 35 else (2 if rsi > 65 else 1)
    media, desvio = row.get("Media_Macro", 1), row.get("Desvio_Macro", 0)
    volat_disc = 1 if (desvio / media if media > 0 else 0) > 0.10 else 0
    xgb_disc = 0 if roi_pred < -4 else (2 if roi_pred > 6 else 1)
    estacao_disc = int(row.get("Season_Num", 0))
    origem_disc = int(origem)
    preco = row.get("Price", 0)

    lucro_disc, tempo_disc = 0, 0
    if origem_disc == 2:
        lucro_disc = 0 if preco < custo_medio else (1 if preco < alvo_lucro else 2)
        if dias_custodia is None or dias_custodia <= 3:
            tempo_disc = 1
        elif dias_custodia <= 8:
            tempo_disc = 2
        else:
            tempo_disc = 3
    return f"{rsi_disc}{volat_disc}{xgb_disc}{estacao_disc}{origem_disc}{lucro_disc}{tempo_disc}"


def construir_estado_rl_continuo(row, origem, custo_medio, roi_pred, dias_custodia):
    """Mesmas 7 dimensões, sem discretizar (apenas coleta — ainda não há modelo)."""
    rsi = row.get("RSI_Micro", 50)
    media, desvio = row.get("Media_Macro", 1), row.get("Desvio_Macro", 0)
    prop_volat = desvio / media if media > 0 else 0
    preco = row.get("Price", 0)
    lucro = ((preco - custo_medio) / custo_medio) if (origem == 2 and custo_medio > 0) else 0.0
    dias = dias_custodia if (origem == 2 and dias_custodia is not None) else 0.0
    valores = [rsi, prop_volat, roi_pred, row.get("Season_Num", 0), origem, lucro, dias]
    return ",".join(f"{v:.4f}" for v in valores)


# ==============================================================================
# DECISÃO (pura)
# ==============================================================================
def decidir_item(item, row, roi_pred, saldo_item, compras_log_item, custo_medio, dias_custodia,
                 mult_desvio, veto_manual, q_table, q_table_item,
                 df_audit_mem=None, d_item_hist=None):
    """
    Decisão completa para UM item. Retorna dict (ou None se o item não deve ser
    listado). Não faz I/O — o chamador cuida de dedup, gravação e Telegram.
    `roi_pred` em % (já com ajuste Bayesiano); itens sem previsão usam 0.0.
    """
    media, desvio = row.get("Media_Macro"), row.get("Desvio_Macro")
    if pd.isna(media) or pd.isna(desvio):
        return None
    preco_atual = row["Price"]
    alvo_in = media - mult_desvio * desvio
    if saldo_item <= 0 and preco_atual > alvo_in:
        return None

    sujeira = max(0, saldo_item - compras_log_item)
    prop_suja = (sujeira / saldo_item) if saldo_item > 0 else 0
    origem = 2 if (compras_log_item > 0 and prop_suja <= TOLERANCIA_SUJEIRA) else 1
    if origem == 2 and saldo_item * preco_atual < POSICAO_MINIMA_SFL:
        origem = 1

    alvo_out_est = media + mult_desvio * desvio
    if origem == 2 and custo_medio > 0:
        alvo_lucro = (custo_medio * (1 + LUCRO_REAL_DESEJADO)) / (1 - TAXA_MARKETPLACE)
        alvo_out = max(alvo_lucro, alvo_out_est)
    else:
        alvo_lucro, alvo_out = 0.0, alvo_out_est

    margem = calcular_margem_atencao(alvo_in, alvo_out)
    dias_cust = dias_custodia if origem == 2 else None
    estado_rl = construir_estado_rl_hibrido(row, origem, custo_medio, alvo_lucro, dias_cust, roi_pred)
    estado_cont = construir_estado_rl_continuo(row, origem, custo_medio, roi_pred, dias_cust)

    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in + margem:
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER (Lucro Disponível)" if (origem == 2 and alvo_lucro > alvo_out_est) else "🔴 VENDER"
    elif preco_atual >= alvo_out - margem:
        acao = "🎯 PERTO DA SAÍDA"
    acao_banda = acao

    if origem == 2 and preco_atual < custo_medio:
        deve_stopar, _ = calcular_stop_adaptativo(item, preco_atual, roi_pred / 100, df_audit_mem=df_audit_mem)
        if deve_stopar:
            acao = "⚠️ VENDER (Stop Loss)"
        elif acao == "ESPERAR":
            acao = "ESPERAR (Prejuízo)"
    acao_pos_stop = acao

    if acao == "🟢 COMPRAR":
        retorno_banda = alvo_out * (1 - TAXA_MARKETPLACE) / preco_atual - 1
        if preco_congelado(d_item_hist):
            acao = "🚫 PREÇO CONGELADO (sem mercado)"
        elif retorno_banda < LUCRO_MIN_BANDA:
            acao = "🚫 BANDA < TAXA (compra sem margem)"

    if acao == "🟢 COMPRAR" and veto_manual:
        acao = "🚫 COMPRA VETADA"

    roi_exibido, vetado_rl = roi_pred, False
    if acao == "🟢 COMPRAR":
        q_item = q_table_item.get(f"{item}|{estado_rl}", {})
        vetado_rl, bonus = calcular_veto_rl(
            estado_rl, q_table, q_compra_item=q_item.get("COMPRA"), n_obs_item=q_item.get("N_COMPRA", 0))
        roi_exibido = roi_pred + bonus
        if vetado_rl:
            acao = "🚫 VETO DO CÉREBRO RL (Histórico de Prejuízo)"

    em_crash, var24h = detectar_regime(d_item_hist)
    if em_crash and "COMPRAR" in acao:
        acao = f"⚠️ COMPRAR (QUEDA {var24h:.1f}% em 24h)"

    gera_sinal = (acao in ACOES_SINAL) or (em_crash and "COMPRAR" in acao)
    tipo_sinal = "COMPRA" if "COMPRAR" in acao else "VENDA"
    sufixo = "_STOP" if "Stop Loss" in acao else ""
    if tipo_sinal == "COMPRA":
        ref_fin = alvo_out
    elif sufixo:
        ref_fin = preco_atual
    elif origem == 1:
        ref_fin = media
    else:
        ref_fin = custo_medio

    return {
        "acao": acao, "acao_banda": acao_banda, "acao_pos_stop": acao_pos_stop,
        "origem": origem, "alvo_in": alvo_in, "alvo_out": alvo_out, "alvo_lucro": alvo_lucro,
        "estado_rl": estado_rl, "estado_rl_continuo": estado_cont,
        "roi_exibido": roi_exibido, "veto_manual": bool(veto_manual), "vetado_rl": bool(vetado_rl),
        "em_crash": bool(em_crash), "var24h": var24h, "custo_medio": custo_medio,
        "gera_sinal": bool(gera_sinal), "tipo_sinal": tipo_sinal, "sufixo": sufixo, "ref_fin": ref_fin,
    }


def montar_mapa_ultimo_sinal(df_val):
    """
    {chave: {'preco', 'timestamp'}}; chave = 'Item_COMPRA' | 'Item_VENDA' | 'Item_STOP'.
    Linha corrompida é pulada individualmente. Retorna (mapa, n_ignoradas).
    """
    mapa, ignoradas = {}, 0
    if df_val is None or df_val.empty:
        return mapa, ignoradas
    for _, r in df_val.iterrows():
        try:
            item = str(r["Item"]).strip()
            chave = f"{item}_STOP" if "_STOP" in str(r["Chave_Unica"]) else f"{item}_{str(r['Tipo']).strip().upper()}"
            ts = parse_data_sinal_robusta(r["Data_Sinal"])
            if pd.isnull(ts):
                continue
            preco = limpar_numero(r["Preco_Sinal"])
            if chave not in mapa or ts > mapa[chave]["timestamp"]:
                mapa[chave] = {"preco": preco, "timestamp": ts}
        except Exception:
            ignoradas += 1
    return mapa, ignoradas


def pode_gerar_sinal(ultimo, preco_atual, agora, is_stop, intervalo_min_h=4, variacao_min=0.03):
    """Deduplicação híbrida: 4h mínimo; 4–24h exige variação ≥ 3% (exceto STOP); após 24h sempre."""
    if not ultimo:
        return True
    horas = (agora - ultimo["timestamp"]).total_seconds() / 3600
    if horas >= 24:
        return True
    if horas < intervalo_min_h:
        return False
    variacao = abs(preco_atual - ultimo["preco"]) / ultimo["preco"] if ultimo["preco"] > 0 else 1.0
    if not is_stop and variacao < variacao_min:
        return False
    return True


# ==============================================================================
# INDICADORES (cache incremental) E FEATURES
# ==============================================================================
def _calcular_indicadores_do_zero(d_bruto_item):
    d = d_bruto_item.sort_values("Timestamp").reset_index(drop=True).copy()
    for col in ["Media_Macro", "Desvio_Macro", "RSI_Micro", "Season_Num", "Lag_1", "Lag_2", "Lag_3",
                "Hora_Dia", "Dia_Semana", "Preco_Futuro", "Peso_Temporal"]:
        d[col] = np.nan
    d["is_holdout"] = False

    janela_macro = 30 * REGISTROS_POR_DIA
    janela_micro = 10 * REGISTROS_POR_DIA
    if len(d) < janela_micro:
        return d

    d["Media_Macro"] = d["Price"].rolling(window=janela_macro, min_periods=120).mean()
    d["Desvio_Macro"] = d["Price"].rolling(window=janela_macro, min_periods=120).std()

    delta = d["Price"].diff()
    ganho = delta.clip(lower=0).rolling(window=janela_micro, min_periods=48).mean()
    perda = (-1 * delta.clip(upper=0)).rolling(window=janela_micro, min_periods=48).mean()
    rsi = 100 - (100 / (1 + (ganho / perda.replace(0, np.nan))))
    rsi = rsi.where(~((ganho == 0) & (perda == 0)), 50.0)     # preço constante = neutro (antes virava 100)
    d["RSI_Micro"] = rsi.fillna(100)                           # só ganho (perda = 0) = 100

    d["Season_Num"] = d["Season"].map(MAP_ESTACOES).fillna(0)
    d["Lag_1"], d["Lag_2"], d["Lag_3"] = d["Price"].shift(1), d["Price"].shift(2), d["Price"].shift(3)
    d["Hora_Dia"] = d["Timestamp"].dt.hour
    d["Dia_Semana"] = d["Timestamp"].dt.dayofweek
    return d


def _recalcular_alvos_e_pesos(d):
    """
    Preco_Futuro = preço ~12 dias DEPOIS (por tempo, tolerância 3h), não 288
    linhas à frente: buracos no RecLog ou preços 0 removidos não deslocam o horizonte.
    Sempre recalculado sobre o histórico completo (depende de linhas futuras).
    """
    d = d.sort_values("Timestamp").reset_index(drop=True).copy()
    alvo = pd.DataFrame({"_t": d["Timestamp"] + pd.Timedelta(days=DIAS_SWING)})
    ref = d[["Timestamp", "Price"]].rename(columns={"Timestamp": "_ts", "Price": "_pf"})
    m = pd.merge_asof(alvo, ref, left_on="_t", right_on="_ts", direction="nearest", tolerance=TOLERANCIA_ALVO_FUTURO)
    d["Preco_Futuro"] = m["_pf"].values
    max_date = d["Timestamp"].max()
    d["Peso_Temporal"] = np.exp(-0.01 * (max_date - d["Timestamp"]).dt.days)
    return d


def _atribuir_holdout(d):
    """3 janelas de 72h espalhadas no histórico (precisa do df completo)."""
    d = d.copy()
    d["is_holdout"] = False
    JAN, N_JAN, WARMUP = 72, 3, 120
    zona_fim = len(d) - SHIFT_ALVO
    zona = zona_fim - WARMUP
    col = d.columns.get_loc("is_holdout")
    if zona >= (N_JAN * JAN + 50):
        seg = zona // N_JAN
        for i in range(N_JAN):
            fim = WARMUP + seg * (i + 1)
            ini = max(WARMUP, fim - JAN)
            if fim <= zona_fim:
                d.iloc[ini:fim, col] = True
    else:
        ini = zona_fim - JAN
        if ini > WARMUP:
            d.iloc[ini:zona_fim, col] = True
    return d


def preparar_indicadores_incremental(df_bruto_geral, nome_item, cache):
    """
    Só as linhas novas desde o último ciclo têm Media/Desvio/RSI calculados.
    Preco_Futuro, Peso_Temporal e is_holdout são refeitos sobre o histórico
    completo em todo ciclo (dependem de linhas futuras / do tamanho total).
    cache[item] = {'df': DataFrame processado, 'ultimo_ts': Timestamp}
    """
    d_item = df_bruto_geral[df_bruto_geral["Item"] == nome_item].copy()
    d_item["Timestamp"] = pd.to_datetime(d_item["Timestamp"], dayfirst=True, errors="coerce")
    d_item = d_item.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if d_item.empty:
        return pd.DataFrame()

    ultimo_global = d_item["Timestamp"].max()
    c = cache.get(nome_item)
    janela_micro = 10 * REGISTROS_POR_DIA
    janela_macro = 30 * REGISTROS_POR_DIA

    def finalizar(d, ult):
        d = _atribuir_holdout(_recalcular_alvos_e_pesos(d))
        cache[nome_item] = {"df": d, "ultimo_ts": ult}
        return d

    # Sem cache, ou item ainda "curto" (indicadores NaN): recalcula do zero
    if not c or c.get("df") is None or c["df"].empty or len(c["df"]) < janela_micro:
        return finalizar(_calcular_indicadores_do_zero(d_item), ultimo_global)

    ultimo_cache = c["ultimo_ts"]
    novas = d_item[d_item["Timestamp"] > ultimo_cache]
    if novas.empty:
        return finalizar(c["df"].drop(columns=["Preco_Futuro", "Peso_Temporal"], errors="ignore"), ultimo_cache)

    contexto = d_item[d_item["Timestamp"] <= ultimo_cache].tail(janela_macro)
    bloco = _calcular_indicadores_do_zero(pd.concat([contexto, novas]).reset_index(drop=True))
    novas_proc = bloco[bloco["Timestamp"] > ultimo_cache]
    d_full = pd.concat([c["df"], novas_proc], ignore_index=True)
    return finalizar(d_full, ultimo_global)


def construir_indices_insumos(df_bruto, mapa_receita=MAPA_RECEITA):
    """{tupla_insumos_ordenada: DataFrame[Timestamp, Indice_Insumo]} — 1 índice por combinação única."""
    indices = {}
    for combo in {tuple(sorted(v)) for v in mapa_receita.values()}:
        df_i = df_bruto[df_bruto["Item"].isin(combo)]
        if df_i.empty:
            continue
        pivot = df_i.pivot_table(index="Timestamp", columns="Item", values="Price", aggfunc="last")
        pivot = pivot.reindex(columns=list(combo)).sort_index().ffill()
        indices[combo] = pivot.mean(axis=1, skipna=True).rename("Indice_Insumo").reset_index()
    return indices


def adicionar_features_correlacao(d_item, item, indices_insumos, mapa_receita=MAPA_RECEITA):
    """Indice_Insumo, Margem_Insumo e versões relativas à própria média de 30 dias (Indice_Dist, Margem_Dist)."""
    cols = ["Indice_Insumo", "Margem_Insumo", "Indice_Dist", "Margem_Dist"]
    d = d_item.drop(columns=cols, errors="ignore").copy()
    df_indice = indices_insumos.get(tuple(sorted(mapa_receita[item]))) if item in mapa_receita else None
    if df_indice is None or df_indice.empty:
        for c in cols:
            d[c] = np.nan
        return d

    d = pd.merge_asof(d.sort_values("Timestamp"), df_indice.sort_values("Timestamp"),
                      on="Timestamp", direction="backward", tolerance=pd.Timedelta("2h"))
    d["Margem_Insumo"] = d["Price"] / d["Indice_Insumo"].replace(0, np.nan)
    j = 30 * REGISTROS_POR_DIA
    d["Indice_Dist"] = d["Indice_Insumo"] / d["Indice_Insumo"].rolling(j, min_periods=120).mean() - 1
    d["Margem_Dist"] = d["Margem_Insumo"] / d["Margem_Insumo"].rolling(j, min_periods=120).mean() - 1
    return d


def adicionar_features_relativas(d):
    """Features comparáveis entre itens + alvo Alvo_Ret = log(Preco_Futuro / Price)."""
    d = d.copy()
    p = d["Price"]
    for k in (1, 2, 3):
        d[f"Ret_L{k}"] = d[f"Lag_{k}"] / p - 1
    media = d["Media_Macro"].replace(0, np.nan)
    d["Z_Macro"] = (p - d["Media_Macro"]) / d["Desvio_Macro"].replace(0, np.nan)
    d["Volat_Rel"] = d["Desvio_Macro"] / media
    d["Dist_Media"] = p / media - 1
    d["Alvo_Ret"] = np.log(d["Preco_Futuro"] / p)
    return d


def construir_features_item(df_bruto, item, cache, indices_insumos):
    d = preparar_indicadores_incremental(df_bruto, item, cache)
    if d.empty:
        return d
    return adicionar_features_relativas(adicionar_features_correlacao(d, item, indices_insumos))


# ==============================================================================
# MODELO POOLED (alvo = retorno logarítmico em 12 dias)
# ==============================================================================
def aplicar_categorias(df, categorias):
    df = df.copy()
    df["Item_Cat"] = pd.Categorical(df["Item_Cat"].astype(str), categories=list(categorias))
    return df


def montar_datasets_pooled(cache_indicadores, min_treino=50):
    """Retorna (df_treino, df_teste, categorias). Item_Cat fica como str; use aplicar_categorias()."""
    tr, te = [], []
    obrig = ["Alvo_Ret"] + FEATURES_NUCLEO
    for item, d_it in cache_indicadores.items():
        if d_it is None or d_it.empty or "Alvo_Ret" not in d_it.columns:
            continue
        d = d_it.copy()
        d["Item_Cat"] = item
        d_tr = d[~d["is_holdout"]].dropna(subset=obrig)
        d_te = d[d["is_holdout"]].dropna(subset=obrig)
        if len(d_tr) >= min_treino:
            tr.append(d_tr)
            if not d_te.empty:
                te.append(d_te)
    if not tr:
        return pd.DataFrame(), pd.DataFrame(), []
    df_treino = pd.concat(tr, ignore_index=True)
    df_teste = pd.concat(te, ignore_index=True) if te else pd.DataFrame()
    return df_treino, df_teste, sorted(df_treino["Item_Cat"].unique().tolist())


def treinar_modelo_pooled(df_treino, categorias):
    if df_treino is None or df_treino.empty:
        return None
    d = aplicar_categorias(df_treino, categorias)
    modelo = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, enable_categorical=True, tree_method="hist", verbosity=0)
    modelo.fit(d[COLS_MODELO], d["Alvo_Ret"], sample_weight=d["Peso_Temporal"])
    return modelo


def prever_rois(modelo, categorias, cache_indicadores):
    """{item: ROI previsto em 12 dias, em %} usando a última linha de cada item."""
    rois, cats = {}, list(categorias)
    for item, d_it in cache_indicadores.items():
        if item not in cats or d_it is None or d_it.empty:
            continue
        ultima = d_it.iloc[[-1]]
        if ultima[FEATURES_NUCLEO].isna().any(axis=1).iloc[0]:
            continue
        X = ultima[FEATURES_MODELO].copy()
        X["Item_Cat"] = pd.Categorical([item], categories=cats)
        pred = float(np.clip(modelo.predict(X[COLS_MODELO])[0], -1.0, 1.0))
        rois[item] = (math.exp(pred) - 1) * 100
    return rois


def calcular_audit_linhas(modelo, categorias, df_treino, df_teste, agora_str):
    """Métricas no holdout, em unidades de preço (MAE/RMSE) e direção."""
    if modelo is None or df_teste is None or df_teste.empty:
        return []
    teste = df_teste[df_teste["Item_Cat"].isin(list(categorias))]
    if teste.empty:
        return []
    teste = aplicar_categorias(teste, categorias)
    pred_ret = np.clip(modelo.predict(teste[COLS_MODELO]), -1.0, 1.0)
    teste = teste.assign(Pred_Ret=pred_ret, Pred=teste["Price"].values * np.exp(pred_ret))
    n_tr = df_treino["Item_Cat"].value_counts().to_dict() if df_treino is not None and not df_treino.empty else {}

    linhas = []
    for item, g in teste.groupby("Item_Cat", observed=True):
        if len(g) < 5:
            continue
        erros = g["Pred"].values - g["Preco_Futuro"].values
        mae = float(np.mean(np.abs(erros)))
        rmse = float(np.sqrt(np.mean(erros ** 2)))
        acerto = float(np.mean((g["Preco_Futuro"].values > g["Price"].values) == (g["Pred_Ret"].values > 0)) * 100)
        roi = float(np.mean((np.exp(g["Pred_Ret"].values) - 1) * 100))
        linhas.append([agora_str, str(item), round(mae, 8), round(rmse, 8), round(acerto, 1),
                       int(n_tr.get(str(item), 0)), len(g), round(roi, 2)])
    return linhas


# ==============================================================================
# PORTFÓLIO (informativo)
# ==============================================================================
def calcular_exposicao_por_cluster(inv_dict, precos_atuais, mapa_receita=MAPA_RECEITA):
    grupos = {}
    for item, insumos in mapa_receita.items():
        grupos.setdefault(tuple(sorted(insumos)), []).append(item)
    linhas = []
    for insumos, itens in grupos.items():
        total, detalhe = 0.0, []
        for it in itens:
            valor = inv_dict.get(it, 0.0) * precos_atuais.get(it, 0.0)
            total += valor
            if valor > 0:
                detalhe.append(f"{it}:{round(valor, 2)}")
        linhas.append({"Cluster_Insumos": " + ".join(insumos), "Itens": ", ".join(itens),
                       "Valor_Exposto": round(total, 2), "Detalhe": "; ".join(detalhe)})
    df = pd.DataFrame(linhas)
    return df.sort_values("Valor_Exposto", ascending=False).reset_index(drop=True) if not df.empty else df
