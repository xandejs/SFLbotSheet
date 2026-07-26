"""
bot_v2.py — Arquitetura V2 do bot de trading SFL (FASE SOMBRA)

NÃO decide capital real. Roda em paralelo ao bot_4_24.py (produção), lendo
RecLog/TradeLog/Inventario da planilha original (apenas leitura) e escrevendo
suas próprias abas numa planilha separada (SFL 2025_10_V2). Só passa a gerar
sinais que valem execução real quando o critério de promoção descrito em
ARQUITETURA_V2.md for cumprido.

O QUE MUDA em relação ao bot_4_24.py (V1) — ver ARQUITETURA_V2.md:
  1. Modelo pooled: 1 XGBoost treinado com os 63 itens juntos (Item como
     feature categórica), retreinado 1x/dia, em vez de 63 modelos
     individuais retreinados do zero a cada hora.
  2. Features de correlação cruzada: índice de custo de ração
     (Wheat/Corn/Barley) e margem insumo/saída para itens com receita
     conhecida (MAPA_RECEITA).
  3. Cache incremental: indicadores técnicos (Media_Macro, Desvio_Macro,
     RSI_Micro etc.) persistidos entre execuções via cache do GitHub
     Actions — só as linhas novas desde o último ciclo são recalculadas,
     não o histórico inteiro.
  4. RL contínuo — FASE DE COLETA APENAS: grava o estado contínuo (não
     discretizado) de cada sinal numa coluna extra (Estado_RL_Continuo),
     para no futuro treinar um modelo de regressão sobre ele e comparar
     contra a Q-table tabular. Ainda NÃO decide nada — não há dados
     suficientes ainda pra treinar esse modelo com confiança.
  5. Camada de portfólio: relatório informativo de exposição por cluster de
     itens com insumo compartilhado. Não bloqueia nem reduz sinal.

O QUE NÃO MUDA (herdado do V1 sem alteração de lógica — já testado e fora
do escopo desta proposta): validação de sinais com sustentação de preço e
janela de graça, FIFO híbrido, calibração Bayesiana dos mapas de risco,
Q-learning tabular (global + item-específico), veto RL adaptativo, stop
loss adaptativo, margem de atenção, detector de regime, deduplicação
híbrida, escrita atômica.

⚠️ ESTE É UM PRIMEIRO ESQUELETO — revisar e testar (idealmente contra uma
cópia da planilha, via workflow_dispatch manual) antes de deixar em cron.
Não foi executado num ambiente real ainda.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
import math
import pickle
import requests
from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# --- Isolamento de planilhas (ver ARQUITETURA_V2.md) ---
PLANILHA_ORIGEM = "SFL 2025_10"       # RecLog/TradeLog/Inventario — APENAS LEITURA
PLANILHA_V2     = "SFL 2025_10_V2"    # abas geridas pela V2 — leitura E escrita

REGISTROS_POR_DIA = 24
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.07
DIAS_SWING = 12
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA   # estrutural — idêntico ao V1, não mudar
DIAS_MAX_MONITORAMENTO = 30

TOLERANCIA_SUJEIRA = 0.15
POSICAO_MINIMA_SFL = 5.0

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

HEADERS_AUDIT = [
    "Timestamp", "Item", "MAE", "RMSE",
    "Acerto_Direcao_%", "N_Treino", "N_Teste", "ROI_Previsto_%"
]

# --- NOVO: mapa de receita explícito (item -> insumos), base para as
# features de correlação cruzada. Baseado em conhecimento do jogo, não
# descoberto por correlação estatística cega entre os 63 itens.
# TODO antes de expandir: validar cada entrada com df.corr() com lag sobre
# o RecLog histórico (ver ARQUITETURA_V2.md seção 1) — jogadores podem
# produzir insumo próprio sem passar pelo marketplace, o que enfraquece
# o vínculo de preço observável.
MAPA_RECEITA = {
    # Produtos Animais (Ração/Grãos)
    "Egg":         ["Wheat", "Corn", "Barley"],
    "Feather":     ["Wheat", "Corn", "Barley"],
    "Milk":        ["Wheat", "Corn", "Barley"],
    "Leather":     ["Wheat", "Corn", "Barley"],
    "Wool":        ["Wheat", "Corn", "Barley"],
    "Merino Wool": ["Wheat", "Corn", "Barley"],
    
    # Recursos e Minérios
    "Stone":       ["Wood"],
    "Iron":        ["Wood", "Stone"],
    "Gold":        ["Wood", "Iron"],
    "Crimstone":   ["Wood", "Gold"],
}

# Peso igual para os 3 insumos de ração por ora — ajustar se o teste de
# correlação mostrar que um pesa mais que outro na prática.
PESOS_RACAO = {"Wheat": 1 / 3, "Corn": 1 / 3, "Barley": 1 / 3}

X_COLS_BASE = ['Price', 'Lag_1', 'Lag_2', 'Lag_3', 'RSI_Micro', 'Media_Macro',
               'Desvio_Macro', 'Season_Num', 'Hora_Dia', 'Dia_Semana']
X_COLS_CORRELACAO = ['Indice_Racao', 'Margem_Insumo']
X_COLS_POOLED = X_COLS_BASE + X_COLS_CORRELACAO

# --- NOVO: cache incremental (feature store leve) ---
# Persistido entre execuções via actions/cache (ver bot_v2.yml). Se o
# arquivo não existir ainda (primeira execução, ou cache expirado), começa
# vazio e o comportamento cai para o mesmo recálculo do zero do V1 — sem
# quebrar o ciclo, só mais lento naquela execução específica.
CACHE_DIR = "cache_v2"
CACHE_INDICADORES_PATH = os.path.join(CACHE_DIR, "indicadores_incrementais.pkl")
MODELO_POOLED_PATH = os.path.join(CACHE_DIR, "modelo_pooled.json")
MODELO_POOLED_META_PATH = os.path.join(CACHE_DIR, "modelo_pooled_meta.pkl")

# alpha_rl é a taxa MÁXIMA (estados novos); a taxa real decai com o número
# de visitas — mesmo mecanismo do V1, ver Cerebro_RL abaixo.
ALPHA_MINIMO = 0.05


# ==============================================================================
# FUNÇÕES HERDADAS DO V1 (sem alteração de lógica — ver bot_4_24.py)
# ==============================================================================

def limpar_numero(texto):
    if not texto:
        return 0.0
    s = str(texto).strip()
    if '.' in s and ',' in s:
        s = s.replace('.', '')
    s = s.replace(',', '.')
    if s.count('.') > 1:
        partes = s.split('.')
        s = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return float(s)
    except Exception:
        return 0.0


def obter_agora_brt():
    """Datetime atual em Horário de Brasília (BRT), sem tzinfo (compat pandas)."""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).replace(tzinfo=None)


def enviar_telegram(acao, ativo, preco, linha):
    """
    Alerta via Telegram, com prefixo [V2 SOMBRA] para diferenciar visualmente
    dos alertas reais da produção (V1) — evita confundir o que é sinal de
    verdade com o que é observação da fase sombra.
    """
    token = "8715722660:AAGgJSl2w2RpM0kxZ9Y4MFK4lHpDZaLyQmw"
    chat_id = "873167793"

    preco_formatado = f"{preco:.4f}".replace('.', ',')

    if "COMPRAR" in acao.upper():
        emoji = "🟢"
    elif "STOP" in acao.upper():
        emoji = "⚠️"
    else:
        emoji = "🔴"

    hora_formatada = obter_agora_brt().strftime("%d/%m %H:%M:%S")

    mensagem = (
        f"🧪 [V2 SOMBRA — não executar] 🧪\n"
        f"{emoji} *ALERTA DE OPERAÇÃO (V2)* {emoji}\n\n"
        f"*Ação:* {acao.upper()}\n"
        f"*Ativo:* {ativo}\n"
        f"*Preço:* {preco_formatado}\n"
        f"*Linha:* {linha}\n"
        f"*Hora:* {hora_formatada}"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem para o Telegram: {e}")


def parse_data_sinal_robusta(texto):
    if not texto or str(texto).strip() == "":
        return pd.NaT
    texto = str(texto).strip()
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y"):
        dt = pd.to_datetime(texto, format=fmt, errors='coerce')
        if pd.notnull(dt):
            return dt
    return pd.to_datetime(texto, dayfirst=True, errors='coerce')


def calcular_fifo_hibrido(df_trade_raw, inventario_dict):
    try:
        df_log = df_trade_raw.copy()
        df_log['Qtd'] = df_log['Qtd'].apply(limpar_numero)
        df_log['SFL_Total'] = df_log['SFL'].apply(limpar_numero).abs()
        df_compras = df_log[df_log['Tipo'].str.contains("Compra", case=False)].copy()

        fifo_map = {}
        for item, saldo_real in inventario_dict.items():
            if saldo_real <= 0:
                continue
            compras_item = df_compras[df_compras['Nome do Item'] == item].iloc[::-1]
            qtd_acumulada, custo_total = 0, 0
            for _, row in compras_item.iterrows():
                falta = saldo_real - qtd_acumulada
                if falta <= 0:
                    break
                if row['Qtd'] <= 0:
                    continue
                usar = min(row['Qtd'], falta)
                custo_total += usar * (row['SFL_Total'] / row['Qtd'])
                qtd_acumulada += usar
            fifo_map[item] = custo_total / qtd_acumulada if qtd_acumulada > 0 else 0.0
        return fifo_map
    except Exception as e:
        print(f"Erro FIFO: {e}")
        return {}


def salvar_aba_atomica(sh, titulo, header, linhas, rows_est=None, cols_est=None, formatar_header=True):
    """Escrita atômica: grava numa aba temporária, só substitui a original após confirmar sucesso."""
    titulo_temp = f"{titulo}_tmp_{int(datetime.now().timestamp())}"
    n_linhas = len(linhas) + 1
    n_cols = len(header)
    rows_final = str(max(rows_est or 0, n_linhas + 100))
    cols_final = str(max(cols_est or 0, n_cols))

    ws_temp = None
    try:
        ws_temp = sh.add_worksheet(title=titulo_temp, rows=rows_final, cols=cols_final)
        ws_temp.update(values=[header] + linhas, range_name='A1', value_input_option='USER_ENTERED')
        if formatar_header:
            set_frozen(ws_temp, rows=1)
            ultima_col = chr(ord('A') + min(n_cols, 26) - 1)
            format_cell_range(ws_temp, f'A1:{ultima_col}1', CellFormat(textFormat=TextFormat(bold=True)))
    except Exception as e:
        print(f"❌ Falha ao escrever aba temporária para '{titulo}': {e} — aba original preservada.")
        if ws_temp is not None:
            try:
                sh.del_worksheet(ws_temp)
            except Exception:
                pass
        return False

    try:
        try:
            ws_antiga = sh.worksheet(titulo)
            sh.del_worksheet(ws_antiga)
        except Exception:
            pass
        ws_temp.update_title(titulo)
        return True
    except Exception as e:
        print(f"⚠️ Escrita em '{titulo}' concluída, mas troca de nome falhou: {e}. "
              f"Dados seguros na aba temporária '{titulo_temp}' — verifique manualmente.")
        return False


def calcular_veto_rl(estado_rl, q_table, n_obs_por_estado=None, q_compra_item=None, n_obs_item=0):
    if estado_rl not in q_table:
        q_compra_global = 0.0
    else:
        q_compra_global = q_table[estado_rl].get("COMPRA", 0.0)

    if q_compra_item is not None:
        peso_item = min(1.0, n_obs_item / 10.0)
        q_compra = peso_item * q_compra_item + (1 - peso_item) * q_compra_global
    else:
        q_compra = q_compra_global

    if estado_rl not in q_table and q_compra_item is None:
        return False, 0.0

    n_obs = (n_obs_por_estado or {}).get(estado_rl, 0)

    if n_obs < 5:
        threshold = -0.5
    elif n_obs < 20:
        threshold = -0.5 + (0.05 / 15) * (n_obs - 5)
    else:
        threshold = -0.45

    vetado = q_compra < threshold
    bonus_roi = 0.0
    if q_compra > 0.7:
        bonus_roi = 10.0
    elif q_compra > 0.5:
        bonus_roi = 5.0

    return vetado, bonus_roi


def contar_obs_por_estado(df_val):
    if df_val is None or df_val.empty:
        return {}
    concluidos = df_val[df_val['Status'].str.contains("🏆|❌|💰", na=False)]
    if concluidos.empty:
        return {}
    return (
        concluidos['Estado_RL'].astype(str).str.strip().str.replace("'", "")
        .value_counts().to_dict()
    )


def calcular_taxa_confianca(n_total, n_acertos, soma_recompensas,
                             prior_a=1.0, prior_b=1.0, n_minimo=8, confianca_alvo=0.80):
    if n_total == 0:
        return 0.5, False, 0.0

    a = prior_a + n_acertos
    b = prior_b + (n_total - n_acertos)

    taxa_efetiva = a / (a + b)
    ev_esperado = soma_recompensas / n_total if n_total > 0 else 0.0

    variancia = (a * b) / ((a + b) ** 2 * (a + b + 1))
    ic_inferior = max(0, taxa_efetiva - 1.96 * math.sqrt(variancia))
    confiante = (n_total >= n_minimo) and (ic_inferior > 0.35)

    return taxa_efetiva, confiante, ev_esperado


def calcular_prior_empirico(df_conc, forca_prior=4.0):
    if df_conc is None or df_conc.empty:
        return 1.0, 1.0
    taxa_global = df_conc['Pts'].apply(lambda x: x >= 0.5).mean()
    prior_a = max(forca_prior * taxa_global, 0.5)
    prior_b = max(forca_prior * (1 - taxa_global), 0.5)
    return prior_a, prior_b


def calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto):
    if df_val_mem is None or df_val_mem.empty:
        return

    df_conc = df_val_mem[df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)].copy()
    if df_conc.empty:
        return

    def pontos(st):
        if "🏆" in st:
            return 1.0
        if "💰" in st:
            return 0.5
        return 0.0

    df_conc['Pts'] = df_conc['Status'].apply(pontos)
    prior_a_global, prior_b_global = calcular_prior_empirico(df_conc)

    for item_c, grupo in df_conc.groupby('Item'):
        n = len(grupo)
        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts = grupo['Pts'].sum()

        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_pts, prior_a=prior_a_global, prior_b=prior_b_global
        )

        if not confiante:
            mapa_mult[item_c] = 2.2
            mapa_roi[item_c] = -2.0
            mapa_veto[item_c] = False
        elif ev >= 0.4:
            mapa_mult[item_c] = 2.0
            mapa_roi[item_c] = 0.0
            mapa_veto[item_c] = False
        elif ev >= 0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c] = -3.0
            mapa_veto[item_c] = False
        elif ev >= -0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c] = -5.0
            mapa_veto[item_c] = True
        else:
            mapa_mult[item_c] = 2.8
            mapa_roi[item_c] = -10.0
            mapa_veto[item_c] = True


def calcular_stop_adaptativo(item, preco_atual, roi_pred, df_audit_mem=None,
                              n_sigma=3.0, stop_minimo=-0.08, stop_maximo=-0.35):
    mae_relativo = None

    if df_audit_mem is not None and not df_audit_mem.empty:
        audit_item = df_audit_mem[df_audit_mem['Item'] == item].tail(72)
        if not audit_item.empty:
            mae_vals = audit_item['MAE'].apply(limpar_numero)
            mae_vals = mae_vals[mae_vals > 0]
            if not mae_vals.empty and preco_atual > 0:
                mae_medio = mae_vals.mean()
                mae_relativo = mae_medio / preco_atual

    if mae_relativo is not None and mae_relativo > 0:
        threshold_bruto = -n_sigma * mae_relativo
        threshold = max(stop_maximo, min(stop_minimo, threshold_bruto))
    else:
        threshold = -0.20

    return roi_pred < threshold, threshold


def calcular_margem_atencao(alvo_in, alvo_out, fracao=0.15, minimo_abs=0.0001, maximo_pct=0.10):
    banda = max(0, alvo_out - alvo_in)
    margem = banda * fracao
    margem = max(margem, minimo_abs)
    if alvo_in > 0:
        margem = min(margem, alvo_in * maximo_pct)
    return round(margem, 6)


def detectar_regime(d_item, janela_crash_h=24, limiar_queda=-0.12):
    if d_item is None or len(d_item) < janela_crash_h:
        return False, 0.0

    preco_atual = d_item.iloc[-1]['Price']
    preco_24h = d_item.iloc[-janela_crash_h]['Price']

    if preco_24h == 0:
        return False, 0.0

    variacao_24h = (preco_atual - preco_24h) / preco_24h
    return variacao_24h < limiar_queda, round(variacao_24h * 100, 2)


def calcular_roi_real(tipo, preco_sinal, preco_final):
    if preco_sinal <= 0:
        return 0.0
    if tipo == "COMPRA":
        venda_liquida = preco_final * (1 - TAXA_MARKETPLACE)
        return ((venda_liquida - preco_sinal) / preco_sinal) * 100
    return ((preco_sinal - preco_final) / preco_sinal) * 100


def calcular_alocacao_recursos(df_val_mem, dias_swing=12, min_sinais=3):
    colunas_saida = ["Item", "Taxa_Acerto_%", "EV_Sinal", "ROI_Medio_%",
                      "Dias_Medio", "Retorno_Anualizado_%", "Score", "Peso_Alocacao_%"]

    if df_val_mem is None or df_val_mem.empty:
        return pd.DataFrame(columns=colunas_saida)

    df_conc = df_val_mem[df_val_mem['Status'].astype(str).str.contains("🏆|❌|💰", na=False)].copy()
    if df_conc.empty:
        return pd.DataFrame(columns=colunas_saida)

    def pontos(st):
        if "🏆" in st:
            return 1.0
        if "💰" in st:
            return 0.5
        return 0.0

    df_conc['Pts'] = df_conc['Status'].astype(str).apply(pontos)
    df_conc['Preco_Sinal_N'] = df_conc['Preco_Sinal'].apply(limpar_numero)
    df_conc['Preco_Final_N'] = df_conc['Preco_Final'].apply(limpar_numero)
    df_conc['Dias_Ate_Alvo_N'] = pd.to_numeric(df_conc['Dias_Ate_Alvo'], errors='coerce')

    def roi_realizado(r):
        if r['Preco_Final_N'] > 0:
            return calcular_roi_real(r['Tipo'], r['Preco_Sinal_N'], r['Preco_Final_N'])
        if r['Preco_Sinal_N'] <= 0:
            return 0.0
        ref_n = limpar_numero(r['Referencia_Fin'])
        if r['Tipo'] == "COMPRA":
            venda_liquida = ref_n * (1 - TAXA_MARKETPLACE)
            return ((venda_liquida - r['Preco_Sinal_N']) / r['Preco_Sinal_N']) * 100 * r['Pts']
        return ((ref_n - r['Preco_Sinal_N']) / r['Preco_Sinal_N']) * 100 * r['Pts']

    df_conc['ROI_Realizado'] = df_conc.apply(roi_realizado, axis=1)
    prior_a_global, prior_b_global = calcular_prior_empirico(df_conc)

    linhas = []
    for item_a, grupo in df_conc.groupby('Item'):
        n = len(grupo)
        if n < min_sinais:
            continue

        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts = grupo['Pts'].sum()
        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_pts, prior_a=prior_a_global, prior_b=prior_b_global
        )

        dias_validos = grupo['Dias_Ate_Alvo_N'].dropna()
        dias_medio = dias_validos.mean() if not dias_validos.empty else float(dias_swing)
        dias_medio = max(dias_medio, 1.0)

        roi_medio_pct = grupo['ROI_Realizado'].mean()
        roi_frac = roi_medio_pct / 100

        try:
            retorno_anualizado = ((1 + roi_frac) ** (365 / dias_medio) - 1) * 100
        except (OverflowError, ZeroDivisionError):
            retorno_anualizado = 0.0

        retorno_anualizado = min(retorno_anualizado, 2000.0)
        peso_confianca = taxa_ef if confiante else taxa_ef * 0.5
        score = retorno_anualizado * peso_confianca if ev > 0 else 0.0

        linhas.append({
            "Item": item_a,
            "Taxa_Acerto_%": round(taxa_ef * 100, 1),
            "EV_Sinal": round(ev, 3),
            "ROI_Medio_%": round(roi_medio_pct, 2),
            "Dias_Medio": round(dias_medio, 2),
            "Retorno_Anualizado_%": round(retorno_anualizado, 1),
            "Score": round(score, 2),
        })

    df_score = pd.DataFrame(linhas)
    if df_score.empty:
        return pd.DataFrame(columns=colunas_saida)

    soma_score = df_score[df_score['Score'] > 0]['Score'].sum()
    if soma_score > 0:
        df_score['Peso_Alocacao_%'] = df_score['Score'].apply(
            lambda s: round((s / soma_score) * 100, 2) if s > 0 else 0.0
        )
    else:
        df_score['Peso_Alocacao_%'] = 0.0

    return df_score.sort_values(by='Peso_Alocacao_%', ascending=False).reset_index(drop=True)


def pontuar_status(r):
    st = r['Status']
    if "🏆 ACERTOU" in st:
        return 1.0
    if "💰 LUCRO" in st:
        return 0.5
    return 0.0


def calcular_dias_custodia(item, df_trade):
    """
    Dias desde a compra mais antiga ainda em custódia (FIFO real). Extraído
    como função própria (o V1 calculava isso só dentro de
    construir_estado_rl_hibrido) para ser reaproveitado também pela versão
    contínua do estado (construir_estado_rl_continuo), sem duplicar a busca
    no TradeLog.
    """
    try:
        compras_item = df_trade[(df_trade['Nome do Item'] == item) &
                                 (df_trade['Tipo'].str.contains("Compra", case=False))]
        if compras_item.empty:
            return None
        col_data = 'Timestamp' if 'Timestamp' in df_trade.columns else 'Data'
        data_compra_str = str(compras_item.iloc[0][col_data]).split()[0]
        for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                dt_compra = datetime.strptime(data_compra_str, fmt)
                return (obter_agora_brt() - dt_compra).days
            except Exception:
                continue
        return None
    except Exception:
        return None


def construir_estado_rl_hibrido(row_mercado, item, origem, custo_medio, alvo_lucro, df_trade, roi_pred):
    """Idêntico ao V1 — gera o Estado_RL discreto de 7 caracteres."""
    rsi_val = row_mercado.get('RSI_Micro', 50)
    rsi_disc = 0 if rsi_val < 35 else (2 if rsi_val > 65 else 1)

    media = row_mercado.get('Media_Macro', 1)
    desvio = row_mercado.get('Desvio_Macro', 0)
    proporcao_volat = desvio / media if media > 0 else 0
    volat_disc = 1 if proporcao_volat > 0.10 else 0

    xgb_disc = 0 if roi_pred < -4 else (2 if roi_pred > 6 else 1)
    estacao_disc = int(row_mercado.get('Season_Num', 0))
    origem_disc = int(origem)

    preco_atual = row_mercado.get('Price', 0)
    lucro_disc = 0
    tempo_disc = 0

    if origem_disc == 2:
        if preco_atual < custo_medio:
            lucro_disc = 0
        elif preco_atual < alvo_lucro:
            lucro_disc = 1
        else:
            lucro_disc = 2

        dias_em_custodia = calcular_dias_custodia(item, df_trade)
        if dias_em_custodia is None:
            tempo_disc = 1
        elif dias_em_custodia <= 3:
            tempo_disc = 1
        elif dias_em_custodia <= 8:
            tempo_disc = 2
        else:
            tempo_disc = 3
    else:
        lucro_disc = 0
        tempo_disc = 0

    return f"{rsi_disc}{volat_disc}{xgb_disc}{estacao_disc}{origem_disc}{lucro_disc}{tempo_disc}"


def construir_estado_rl_continuo(row_mercado, origem, custo_medio, roi_pred, dias_custodia):
    """
    NOVO — versão NÃO discretizada das mesmas 7 dimensões do Estado_RL,
    gravada como string separada por vírgula em 'Estado_RL_Continuo'.

    FASE ATUAL: apenas coleta. Ainda não existe um modelo de regressão
    treinado sobre esses valores — não há histórico suficiente ainda para
    treinar com confiança (o V1 nunca gravou o estado em forma contínua).
    Quando Validacao_Sinais (V2) acumular sinais concluídos suficientes com
    esta coluna preenchida, um modelo de aproximação contínua pode ser
    treinado e comparado contra a Q-table tabular em modo log-only, antes de
    cogitar decidir qualquer veto com base nele.
    """
    rsi_val = row_mercado.get('RSI_Micro', 50)
    media = row_mercado.get('Media_Macro', 1)
    desvio = row_mercado.get('Desvio_Macro', 0)
    proporcao_volat = desvio / media if media > 0 else 0
    estacao = row_mercado.get('Season_Num', 0)
    preco_atual = row_mercado.get('Price', 0)
    lucro_proxy = ((preco_atual - custo_medio) / custo_medio) if (origem == 2 and custo_medio > 0) else 0.0
    dias_custodia_v = dias_custodia if (origem == 2 and dias_custodia is not None) else 0.0

    valores = [rsi_val, proporcao_volat, roi_pred, estacao, origem, lucro_proxy, dias_custodia_v]
    return ",".join(f"{v:.4f}" for v in valores)


# ==============================================================================
# NOVO — CACHE INCREMENTAL DE INDICADORES
# ==============================================================================

def carregar_cache_indicadores():
    if os.path.exists(CACHE_INDICADORES_PATH):
        try:
            with open(CACHE_INDICADORES_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Cache de indicadores corrompido/ilegível ({e}) — recomputando do zero neste ciclo.")
    return {}


def salvar_cache_indicadores(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_INDICADORES_PATH, "wb") as f:
        pickle.dump(cache, f)


def _calcular_indicadores_do_zero(d_bruto_item, nome_item):
    """
    Núcleo de cálculo de indicadores — mesma lógica do preparar_indicadores
    do V1, mas sem atribuir is_holdout (feito à parte por _atribuir_holdout,
    sobre o histórico completo já concatenado, para a janela de validação
    cruzada continuar correta mesmo quando só as linhas novas são
    recalculadas aqui).
    """
    d = d_bruto_item.sort_values('Timestamp').reset_index(drop=True).copy()

    for col in ['Media_Macro', 'Desvio_Macro', 'RSI_Micro', 'Season_Num',
                'Lag_1', 'Lag_2', 'Lag_3', 'Hora_Dia', 'Dia_Semana',
                'Preco_Futuro', 'Peso_Temporal', 'Indice_Racao', 'Margem_Insumo']:
        d[col] = np.nan
    d['is_holdout'] = False

    janela_macro_dinamica = 30 * REGISTROS_POR_DIA
    janela_micro_dinamica = 10 * REGISTROS_POR_DIA

    if len(d) < janela_micro_dinamica:
        return d

    d['Media_Macro'] = d['Price'].rolling(window=janela_macro_dinamica, min_periods=120).mean()
    d['Desvio_Macro'] = d['Price'].rolling(window=janela_macro_dinamica, min_periods=120).std()

    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=janela_micro_dinamica, min_periods=48).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=janela_micro_dinamica, min_periods=48).mean()
    rsi_raw = 100 - (100 / (1 + (ganho / perda.replace(0, np.nan))))
    d['RSI_Micro'] = rsi_raw.fillna(100)

    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    d['Lag_1'] = d['Price'].shift(1)
    d['Lag_2'] = d['Price'].shift(2)
    d['Lag_3'] = d['Price'].shift(3)
    d['Hora_Dia'] = d['Timestamp'].dt.hour
    d['Dia_Semana'] = d['Timestamp'].dt.dayofweek
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)

    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))

    return d


def _atribuir_holdout(d):
    """
    Marca is_holdout (3 janelas de 72h) sobre o histórico COMPLETO — precisa
    rodar sobre o df inteiro porque a posição das janelas depende do
    tamanho total do histórico disponível, não só das linhas novas. É uma
    operação barata (atribuição booleana por índice, sem rolling), então
    rodar em todo ciclo não pesa no runtime.
    """
    d = d.copy()
    d['is_holdout'] = False
    HOLDOUT_JANELA = 72
    N_JANELAS_HOLDOUT = 3
    WARMUP_MACRO = 120

    zona_fim = len(d) - SHIFT_ALVO
    zona_disponivel = zona_fim - WARMUP_MACRO

    if zona_disponivel >= (N_JANELAS_HOLDOUT * HOLDOUT_JANELA + 50):
        tamanho_segmento = zona_disponivel // N_JANELAS_HOLDOUT
        for i in range(N_JANELAS_HOLDOUT):
            seg_fim = WARMUP_MACRO + tamanho_segmento * (i + 1)
            seg_ini = max(WARMUP_MACRO, seg_fim - HOLDOUT_JANELA)
            if seg_fim <= zona_fim:
                d.iloc[seg_ini:seg_fim, d.columns.get_loc('is_holdout')] = True
    else:
        holdout_ini = zona_fim - HOLDOUT_JANELA
        if holdout_ini > WARMUP_MACRO:
            d.iloc[holdout_ini:zona_fim, d.columns.get_loc('is_holdout')] = True

    return d


def preparar_indicadores_incremental(df_bruto_geral, nome_item, cache):
    """
    Versão com cache do preparar_indicadores do V1. Estatísticas de janela
    trailing (Media_Macro, Desvio_Macro, RSI_Micro) NÃO mudam de valor para
    linhas antigas quando uma linha nova é adicionada ao final — só as
    linhas novas desde o último ciclo precisam ser calculadas. Isso evita
    reprocessar o histórico inteiro (que só cresce) a cada execução.

    cache[item] = {"df": DataFrame já processado, "ultimo_ts": Timestamp}
    """
    d_bruto_item = df_bruto_geral[df_bruto_geral['Item'] == nome_item].copy()
    d_bruto_item['Timestamp'] = pd.to_datetime(d_bruto_item['Timestamp'], dayfirst=True, errors='coerce')
    d_bruto_item = d_bruto_item.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

    if d_bruto_item.empty:
        return pd.DataFrame()

    ultimo_ts_global = d_bruto_item['Timestamp'].max()
    cache_item = cache.get(nome_item)

    if not cache_item or cache_item.get('df') is None or cache_item['df'].empty:
        d_full = _calcular_indicadores_do_zero(d_bruto_item, nome_item)
        d_full = _atribuir_holdout(d_full)
        cache[nome_item] = {'df': d_full, 'ultimo_ts': ultimo_ts_global}
        return d_full

    ultimo_ts_cache = cache_item['ultimo_ts']
    linhas_novas = d_bruto_item[d_bruto_item['Timestamp'] > ultimo_ts_cache]

    if linhas_novas.empty:
        return cache_item['df']

    janela_macro_dinamica = 30 * REGISTROS_POR_DIA
    contexto = d_bruto_item[d_bruto_item['Timestamp'] <= ultimo_ts_cache].tail(janela_macro_dinamica)
    bloco = pd.concat([contexto, linhas_novas]).reset_index(drop=True)
    bloco_processado = _calcular_indicadores_do_zero(bloco, nome_item)
    novas_processadas = bloco_processado[bloco_processado['Timestamp'] > ultimo_ts_cache]

    d_full = pd.concat([cache_item['df'], novas_processadas], ignore_index=True)
    d_full = _atribuir_holdout(d_full)

    cache[nome_item] = {'df': d_full, 'ultimo_ts': ultimo_ts_global}
    return d_full


# ==============================================================================
# NOVO — FEATURES DE CORRELAÇÃO CRUZADA (índice de ração / margem insumo)
# ==============================================================================

def construir_indice_racao(df_bruto):
    """
    Índice de custo de ração: combinação ponderada de Wheat/Corn/Barley por
    timestamp. Retorna DataFrame [Timestamp, Indice_Racao] pronto para merge
    com o histórico de qualquer item que dependa de ração (MAPA_RECEITA).
    """
    itens_racao = list(PESOS_RACAO.keys())
    df_racao = df_bruto[df_bruto['Item'].isin(itens_racao)].copy()
    if df_racao.empty:
        return pd.DataFrame(columns=['Timestamp', 'Indice_Racao'])

    df_racao['Timestamp'] = pd.to_datetime(df_racao['Timestamp'], dayfirst=True, errors='coerce')
    df_racao = df_racao.dropna(subset=['Timestamp'])

    pivot = df_racao.pivot_table(index='Timestamp', columns='Item', values='Price', aggfunc='last')
    pivot = pivot.reindex(columns=itens_racao)
    pivot = pivot.sort_index().ffill()   # preenche gaps pontuais de coleta

    indice = sum(pivot[item] * peso for item, peso in PESOS_RACAO.items() if item in pivot.columns)
    return indice.rename('Indice_Racao').reset_index()


def adicionar_features_correlacao(d_item, item, df_indice_racao):
    """
    Adiciona Indice_Racao e Margem_Insumo (preço do item / índice de ração)
    ao DataFrame de indicadores do item, se ele depender de ração conforme
    MAPA_RECEITA. NaN para itens fora do mapa — XGBoost trata como valor
    ausente nativamente, sem quebrar o treino/inferência.
    """
    d_item = d_item.copy()

    if item not in MAPA_RECEITA or df_indice_racao.empty:
        d_item['Indice_Racao'] = np.nan
        d_item['Margem_Insumo'] = np.nan
        return d_item

    insumos_item = set(MAPA_RECEITA[item])
    if not insumos_item.issubset(set(PESOS_RACAO.keys())):
        # Receita não baseada em ração (ex.: Gold <- Wood+Stone) — fica fora
        # do índice de ração por ora.
        # TODO: índice de custo separado por cluster de receita quando
        # houver mais itens desse tipo (ex.: índice Wood+Stone para Gold).
        d_item['Indice_Racao'] = np.nan
        d_item['Margem_Insumo'] = np.nan
        return d_item

    # Remove os placeholders NaN (já criados em _calcular_indicadores_do_zero)
    # antes do merge — senão merge_asof colide com a coluna de mesmo nome
    # vinda de df_indice_racao (gera Indice_Racao_x/_y em vez de preencher a
    # coluna esperada, e o KeyError aparece mais adiante quando X_COLS_POOLED
    # tenta acessar 'Indice_Racao').
    d_item = d_item.drop(columns=['Indice_Racao', 'Margem_Insumo'], errors='ignore')

    d_item = pd.merge_asof(
        d_item.sort_values('Timestamp'),
        df_indice_racao.sort_values('Timestamp'),
        on='Timestamp', direction='nearest', tolerance=pd.Timedelta('2h')
    )
    d_item['Margem_Insumo'] = d_item.apply(
        lambda r: (r['Price'] / r['Indice_Racao']) if r.get('Indice_Racao', 0) else np.nan, axis=1
    )
    return d_item


# ==============================================================================
# NOVO — MODELO ÚNICO POOLED (substitui os 63 modelos individuais do V1)
# ==============================================================================

def montar_datasets_pooled(cache_indicadores):
    """
    Monta treino/teste concatenando os DataFrames já processados (via cache)
    de todos os itens, com Item_Cat como feature categórica. Operação
    barata — só concatenação, sem recomputar indicadores — então roda todo
    ciclo, mesmo quando o modelo não é retreinado (mantém as métricas de
    XGB_Audit frescas independente da cadência de retreino).
    """
    frames_treino, frames_teste = [], []

    for item, d_it in cache_indicadores.items():
        if d_it is None or d_it.empty:
            continue
        d_it = d_it.copy()
        d_it['Item_Cat'] = item
        d_treino = d_it[~d_it['is_holdout']].dropna(subset=['Preco_Futuro'] + X_COLS_BASE)
        d_teste = d_it[d_it['is_holdout']].dropna(subset=['Preco_Futuro'] + X_COLS_BASE)
        if len(d_treino) >= 50:
            frames_treino.append(d_treino)
        if not d_teste.empty:
            frames_teste.append(d_teste)

    if not frames_treino:
        return pd.DataFrame(), pd.DataFrame()

    df_treino = pd.concat(frames_treino, ignore_index=True)
    df_teste = pd.concat(frames_teste, ignore_index=True) if frames_teste else pd.DataFrame()

    categorias = sorted(df_treino['Item_Cat'].unique().tolist())
    df_treino['Item_Cat'] = pd.Categorical(df_treino['Item_Cat'], categories=categorias)
    if not df_teste.empty:
        # Item no teste que não estava no treino (raro) vira categoria
        # ausente — XGBoost trata como missing, sem quebrar.
        df_teste['Item_Cat'] = pd.Categorical(df_teste['Item_Cat'], categories=categorias)

    return df_treino, df_teste


def treinar_modelo_pooled(df_treino):
    """
    1 XGBoost para os 63 itens juntos — treina 1x por ciclo de retreino
    (diário), não 63x por hora como no V1. Hiperparâmetros de partida — vale
    revisar depois de medir performance real (mais estimadores/profundidade
    que o individual do V1, já que 1 modelo precisa separar padrões entre
    itens via Item_Cat, não só prever 1 série).
    """
    if df_treino.empty:
        return None

    cols_modelo = X_COLS_POOLED + ['Item_Cat']
    modelo = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        enable_categorical=True,
        verbosity=0
    ).fit(
        df_treino[cols_modelo],
        df_treino['Preco_Futuro'],
        sample_weight=df_treino['Peso_Temporal']
    )
    return modelo


def deve_retreinar_pooled():
    """Retorna True se o modelo pooled deve ser retreinado neste ciclo — 1x/dia."""
    if not os.path.exists(MODELO_POOLED_META_PATH):
        return True
    try:
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)
        ultimo_treino = meta.get("ultimo_treino")
        return ultimo_treino is None or (obter_agora_brt() - ultimo_treino).total_seconds() >= 86400
    except Exception:
        return True


def salvar_modelo_pooled(modelo, categorias):
    os.makedirs(CACHE_DIR, exist_ok=True)
    modelo.save_model(MODELO_POOLED_PATH)
    with open(MODELO_POOLED_META_PATH, "wb") as f:
        pickle.dump({"ultimo_treino": obter_agora_brt(), "categorias_item": list(categorias)}, f)


def carregar_modelo_pooled():
    if not (os.path.exists(MODELO_POOLED_PATH) and os.path.exists(MODELO_POOLED_META_PATH)):
        return None, None
    try:
        modelo = xgb.XGBRegressor(enable_categorical=True)
        modelo.load_model(MODELO_POOLED_PATH)
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)
        return modelo, meta.get("categorias_item")
    except Exception as e:
        print(f"⚠️ Falha ao carregar modelo pooled em cache ({e}) — será retreinado neste ciclo.")
        return None, None


# ==============================================================================
# NOVO — CAMADA DE PORTFÓLIO (informativo, não bloqueia sinal)
# ==============================================================================

def calcular_exposicao_por_cluster(inv_dict, precos_atuais, mapa_receita=MAPA_RECEITA):
    """
    Agrupa itens que compartilham ao menos 1 insumo (MAPA_RECEITA) no mesmo
    cluster de correlação, e soma o valor de posição atual (saldo × preço)
    dentro de cada cluster. Puramente informativo nesta fase — não bloqueia
    nem reduz nenhum sinal de compra ainda (ver ARQUITETURA_V2.md).
    """
    # União por insumo compartilhado: dois itens no mesmo cluster se
    # compartilham qualquer insumo em MAPA_RECEITA.
    grupos = {}
    for item, insumos in mapa_receita.items():
        chave_cluster = tuple(sorted(insumos))
        grupos.setdefault(chave_cluster, []).append(item)

    linhas = []
    for insumos, itens_cluster in grupos.items():
        valor_total = 0.0
        detalhe = []
        for item in itens_cluster:
            saldo = inv_dict.get(item, 0.0)
            preco = precos_atuais.get(item, 0.0)
            valor = saldo * preco
            valor_total += valor
            if valor > 0:
                detalhe.append(f"{item}:{round(valor, 2)}")
        linhas.append({
            "Cluster_Insumos": " + ".join(insumos),
            "Itens": ", ".join(itens_cluster),
            "Valor_Exposto": round(valor_total, 2),
            "Detalhe": "; ".join(detalhe) if detalhe else ""
        })

    df_cluster = pd.DataFrame(linhas)
    if df_cluster.empty:
        return df_cluster
    return df_cluster.sort_values(by='Valor_Exposto', ascending=False).reset_index(drop=True)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🧪 Iniciando Bot V2 (SOMBRA) — não decide capital real...")

escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)

sh_origem = gc.open(PLANILHA_ORIGEM)   # RecLog/TradeLog/Inventario — leitura
sh = gc.open(PLANILHA_V2)              # abas geridas pela V2 — leitura e escrita

# ------------------------------------------------------------------
# 1. Dados de base — lidos da planilha ORIGEM, nunca escritos
# ------------------------------------------------------------------
inv_data = sh_origem.worksheet("Inventario").get_all_values()
inv_dict = {row[0]: limpar_numero(row[1]) for row in inv_data[1:]}

trade_data = sh_origem.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'] \
    .apply(lambda x: sum(map(limpar_numero, x))).to_dict()
mapa_custo_fifo = calcular_fifo_hibrido(df_trade, inv_dict)

rec_data = sh_origem.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

# ==============================================================================
# VALIDAÇÃO DE SINAIS (idêntica ao V1 — ver bot_4_24.py — operando na V2)
# ==============================================================================
print("🔄 Processando Validação de Sinais (V2)...")
headers_val = ["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin",
               "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica",
               "Dias_Ate_Alvo", "Preco_Final", "Estado_RL_Continuo"]

try:
    ws_val = sh.worksheet("Validacao_Sinais")
    val_data = ws_val.get_all_values()
except Exception:
    ws_val = sh.add_worksheet(title="Validacao_Sinais", rows="10000", cols="14")
    val_data = []

if not val_data or len(val_data) == 0:
    ws_val.update(values=[headers_val], range_name='A1', value_input_option='USER_ENTERED')
    val_data = [headers_val]

if len(val_data) > 0 and "Estado_RL_Continuo" not in val_data[0]:
    print("⚠️ Injetando coluna 'Estado_RL_Continuo' (nova na V2) para compatibilidade...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=val_data[0])
    df_val = df_val.reindex(columns=headers_val).fillna("")
else:
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=headers_val)

mapa_mult_desvio = {}
mapa_veto = {}
mapa_ajuste_roi = {}

# ------------------------------------------------------------------
# Cerebro_RL / Cerebro_RL_Item — Q-learning tabular, idêntico ao V1
# ------------------------------------------------------------------
q_table = {}
alpha_rl = 0.2

try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    cerebro_data = ws_cerebro.get_all_values()
    if len(cerebro_data) > 1:
        tem_contadores = len(cerebro_data[0]) >= 5
        for row in cerebro_data[1:]:
            if len(row) >= 3:
                st = str(row[0]).strip().replace("'", "")
                q_table[st] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA": limpar_numero(row[2]),
                    "N_COMPRA": int(limpar_numero(row[3])) if tem_contadores and len(row) > 3 else 0,
                    "N_VENDA": int(limpar_numero(row[4])) if tem_contadores and len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL (V2) carregado: {len(q_table)} estados.")
except Exception:
    try:
        ws_cerebro = sh.add_worksheet(title="Cerebro_RL", rows="5000", cols="5")
        ws_cerebro.update(values=[["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]],
                           range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL (V2) criada.")
    except Exception:
        print("⚠️ Aba Cerebro_RL (V2) já existente, mas vazia ou inacessível.")

q_table_item = {}
try:
    ws_cerebro_item = sh.worksheet("Cerebro_RL_Item")
    cerebro_item_data = ws_cerebro_item.get_all_values()
    if len(cerebro_item_data) > 1:
        for row in cerebro_item_data[1:]:
            if len(row) >= 3:
                chave_ci = str(row[0]).strip().replace("'", "")
                q_table_item[chave_ci] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA": limpar_numero(row[2]),
                    "N_COMPRA": int(limpar_numero(row[3])) if len(row) > 3 else 0,
                    "N_VENDA": int(limpar_numero(row[4])) if len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL item-específico (V2) carregado: {len(q_table_item)} combinações.")
except Exception:
    try:
        ws_cerebro_item = sh.add_worksheet(title="Cerebro_RL_Item", rows="20000", cols="5")
        ws_cerebro_item.update(values=[["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]],
                                range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL_Item (V2) criada.")
    except Exception:
        print("⚠️ Aba Cerebro_RL_Item (V2) já existente, mas vazia ou inacessível.")

indices_concluidos_neste_ciclo = []

if not df_val.empty:
    for idx, row_val in df_val.iterrows():
        if row_val['Status'] == 'Aguardando' or row_val['Status'] == '':
            item_v = row_val['Item']
            tipo_v = row_val['Tipo']
            data_sinal_v = parse_data_sinal_robusta(row_val['Data_Sinal'])
            preco_sinal_v = limpar_numero(row_val['Preco_Sinal'])
            ref_fin_v = limpar_numero(row_val['Referencia_Fin'])

            if pd.isnull(data_sinal_v):
                df_val.at[idx, 'Dias_Passados'] = 0
                df_val.at[idx, 'Status'] = "⚠️ ERRO_DATA (Verificar Manualmente)"
                continue

            dias_p = (obter_agora_brt() - data_sinal_v).days

            d_hist = df_bruto[df_bruto['Item'] == item_v].copy()
            d_hist['Timestamp'] = pd.to_datetime(d_hist['Timestamp'], dayfirst=True, errors='coerce')
            df_pos_sinal = d_hist[d_hist['Timestamp'] >= data_sinal_v]

            is_stop_sinal = "_STOP" in str(row_val['Chave_Unica'])
            if tipo_v == "COMPRA" or is_stop_sinal:
                preco_monitoramento = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            else:
                preco_monitoramento = df_pos_sinal['Price'].min() if not df_pos_sinal.empty else preco_sinal_v

            max_pos = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            horas_no_alvo = int((df_pos_sinal['Price'] >= ref_fin_v).sum()) if not df_pos_sinal.empty else 0
            horas_venda = int((df_pos_sinal['Price'] <= ref_fin_v).sum()) if not df_pos_sinal.empty else 0

            dias_ate_alvo = None
            if not df_pos_sinal.empty and pd.notnull(data_sinal_v):
                if tipo_v == "COMPRA":
                    qualifica = df_pos_sinal['Price'] >= ref_fin_v
                elif tipo_v == "VENDA" and "_STOP" not in str(row_val['Chave_Unica']):
                    qualifica = df_pos_sinal['Price'] <= ref_fin_v
                else:
                    qualifica = None

                if qualifica is not None:
                    cum = qualifica.cumsum()
                    atingiu = cum[cum >= 3]
                    if not atingiu.empty:
                        ts_atingiu = df_pos_sinal.loc[atingiu.index[0], 'Timestamp']
                        dias_ate_alvo = round((ts_atingiu - data_sinal_v).total_seconds() / 86400, 2)

            status_v = "Aguardando"

            if tipo_v == "COMPRA":
                if dias_p >= DIAS_SWING:
                    piso_lucro_real = preco_sinal_v * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
                    if horas_no_alvo >= 3:
                        status_v = "🏆 ACERTOU"
                    elif horas_no_alvo >= 1 or max_pos >= piso_lucro_real:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "💰 LUCRO PARCIAL"
                    else:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "❌ FALHOU"
            elif tipo_v == "VENDA":
                if "_STOP" in str(row_val['Chave_Unica']):
                    if dias_p >= DIAS_SWING:
                        if max_pos <= ref_fin_v:
                            status_v = "🏆 ACERTOU (STOP PROTETIVO)"
                        else:
                            status_v = "❌ FALHOU (STOP PREMATURO)"
                else:
                    if dias_p >= DIAS_SWING:
                        if horas_venda >= 3:
                            status_v = "🏆 ACERTOU (FUNDO)"
                        elif horas_venda >= 1:
                            if dias_p >= DIAS_MAX_MONITORAMENTO:
                                status_v = "💰 LUCRO PARCIAL (SPIKE)"
                        else:
                            if dias_p >= DIAS_MAX_MONITORAMENTO:
                                status_v = "❌ FALHOU"

            df_val.at[idx, 'Dias_Passados'] = int(dias_p)
            df_val.at[idx, 'Monitoramento_12d'] = round(preco_monitoramento, 4)
            df_val.at[idx, 'Status'] = status_v

            if dias_ate_alvo is None and status_v not in ("Aguardando",):
                dias_ate_alvo = float(dias_p)
            df_val.at[idx, 'Dias_Ate_Alvo'] = dias_ate_alvo if dias_ate_alvo is not None else ""

            if status_v not in ("Aguardando",) and not df_pos_sinal.empty:
                df_val.at[idx, 'Preco_Final'] = round(df_pos_sinal['Price'].iloc[-1], 4)
            elif status_v not in ("Aguardando",):
                df_val.at[idx, 'Preco_Final'] = round(preco_sinal_v, 4)

            if "🏆" in status_v or "❌" in status_v or "💰" in status_v:
                indices_concluidos_neste_ciclo.append(idx)

    df_val['ID_Sinal'] = pd.to_numeric(df_val['ID_Sinal'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Preco_Sinal'] = df_val['Preco_Sinal'].apply(limpar_numero)
    df_val['Referencia_Fin'] = df_val['Referencia_Fin'].apply(limpar_numero)
    df_val['Monitoramento_12d'] = df_val['Monitoramento_12d'].apply(limpar_numero)
    df_val['Dias_Passados'] = pd.to_numeric(df_val['Dias_Passados'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Dias_Ate_Alvo'] = df_val['Dias_Ate_Alvo'].apply(
        lambda x: round(limpar_numero(x), 2) if str(x).strip() not in ("", "nan", "None") else ""
    )
    df_val['Preco_Final'] = df_val['Preco_Final'].apply(
        lambda x: round(limpar_numero(x), 4) if str(x).strip() not in ("", "nan", "None") else ""
    )
    df_val['Estado_RL'] = df_val['Estado_RL'].astype(str).apply(lambda x: x.strip().replace("'", ""))
    df_val['Estado_RL'] = df_val['Estado_RL'].apply(lambda x: f"'{x}" if x != "" else "")

    ok_val = salvar_aba_atomica(sh, "Validacao_Sinais", headers_val, df_val.values.tolist(),
                                 rows_est=10000, cols_est=len(headers_val))
    if ok_val:
        ws_val = sh.worksheet("Validacao_Sinais")
    else:
        print("⚠️ Validacao_Sinais (V2) não foi atualizada neste ciclo — dados anteriores preservados.")

    try:
        df_alocacao = calcular_alocacao_recursos(df_val, dias_swing=DIAS_SWING, min_sinais=3)
        if not df_alocacao.empty:
            try:
                ws_aloc = sh.worksheet("Alocacao_Recursos")
            except Exception:
                ws_aloc = sh.add_worksheet(title="Alocacao_Recursos", rows="200", cols="8")
                set_frozen(ws_aloc, rows=1)
                format_cell_range(ws_aloc, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))

            ws_aloc.clear()
            ws_aloc.update(values=[df_alocacao.columns.tolist()] + df_alocacao.values.tolist(),
                            range_name='A1', value_input_option='USER_ENTERED')
            print(f"💰 Alocação de Recursos (V2): {len(df_alocacao)} itens.")
        else:
            print("💰 Alocação de Recursos (V2): aguardando sinais concluídos suficientes.")
    except Exception as e:
        print(f"⚠️ Erro ao calcular Alocação de Recursos (V2): {e}")

    if indices_concluidos_neste_ciclo:
        df_reforco = df_val.loc[indices_concluidos_neste_ciclo].copy()
        df_reforco = df_reforco[df_reforco['Estado_RL'].astype(str).str.strip() != ""].copy()

        for _, row_r in df_reforco.iterrows():
            st_rl = str(row_r['Estado_RL']).strip().replace("'", "")
            acao_rl = row_r['Tipo']
            status_rl = row_r['Status']

            preco_sinal_r = limpar_numero(row_r['Preco_Sinal'])
            preco_final_r = limpar_numero(row_r['Preco_Final'])

            if preco_final_r > 0:
                roi_real_pct = calcular_roi_real(acao_rl, preco_sinal_r, preco_final_r)
                recompensa = max(-2.0, min(2.0, roi_real_pct / (LUCRO_REAL_DESEJADO * 100)))
            elif status_rl in ["🏆 ACERTOU", "🏆 ACERTOU (FUNDO)", "🏆 ACERTOU (STOP PROTETIVO)"]:
                recompensa = 1.0
            elif status_rl in ["💰 LUCRO (SUBIU)", "💰 LUCRO PARCIAL", "💰 LUCRO PARCIAL (SPIKE)"]:
                recompensa = 0.5
            else:
                recompensa = -1.0

            if st_rl not in q_table:
                q_table[st_rl] = {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0}

            chave_n = f"N_{acao_rl}"
            n_visitas = q_table[st_rl].get(chave_n, 0) + 1
            alpha_dinamico = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas))

            q_atual = q_table[st_rl].get(acao_rl, 0.0)
            q_table[st_rl][acao_rl] = q_atual + alpha_dinamico * (recompensa - q_atual)
            q_table[st_rl][chave_n] = n_visitas

            chave_item_estado = f"{row_r['Item']}|{st_rl}"
            if chave_item_estado not in q_table_item:
                q_table_item[chave_item_estado] = {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0}

            n_visitas_item = q_table_item[chave_item_estado].get(chave_n, 0) + 1
            alpha_item = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas_item))
            q_atual_item = q_table_item[chave_item_estado].get(acao_rl, 0.0)
            q_table_item[chave_item_estado][acao_rl] = q_atual_item + alpha_item * (recompensa - q_atual_item)
            q_table_item[chave_item_estado][chave_n] = n_visitas_item

        print(f"🧠 Cérebro RL (V2) atualizado: {len(df_reforco)} novos estados processados.")

# ==============================================================================
# NOVO: Calibração híbrida (idêntica em lógica, lendo XGB_Audit da V2)
# ==============================================================================

def construir_calibracao_hibrida_v2(sh_v2, df_bruto_local, df_val_mem=None, q_table_mem=None):
    mapa_mult = {}
    mapa_veto_local = {}
    mapa_roi = {}
    df_audit_retorno = pd.DataFrame(columns=HEADERS_AUDIT)

    if df_val_mem is not None and not df_val_mem.empty:
        df_concluidos = df_val_mem[df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)].copy()
        if not df_concluidos.empty:
            df_concluidos['Pontos'] = df_concluidos.apply(pontuar_status, axis=1)
            calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto_local)
            print(f"📈 Calibração Bayesiana (V2): {df_concluidos['Item'].nunique()} itens | "
                  f"{len(df_concluidos)} sinais concluídos.")
        else:
            print("ℹ️  Calibração (V2): nenhum sinal concluído ainda — fase bootstrap.")
    else:
        print("ℹ️  Calibração (V2): Validacao_Sinais vazia — fase bootstrap.")

    try:
        ws_audit = sh_v2.worksheet("XGB_Audit")
        audit_data = ws_audit.get_all_values()
        if len(audit_data) > 1:
            df_audit = pd.DataFrame(audit_data[1:], columns=audit_data[0])
            df_audit['MAE'] = df_audit['MAE'].apply(limpar_numero)
            df_audit['Acerto_Direcao_%'] = df_audit['Acerto_Direcao_%'].apply(limpar_numero)
            df_audit_retorno = df_audit.copy()
    except gspread.exceptions.WorksheetNotFound:
        print("ℹ️  XGB_Audit (V2) ainda não existe — será criada neste ciclo.")
    except Exception as e:
        print(f"ℹ️  XGB_Audit (V2) inacessível: {e}")

    n_estados = len(q_table_mem) if q_table_mem is not None else 0
    print(f"{'🌱 BOOTSTRAP' if n_estados < 10 else '🧠 Cerebro_RL operacional'} (V2): {n_estados} estados.")
    print(f"✅ Calibração (V2) final: {len(mapa_mult)} itens com parâmetros ajustados.")
    return mapa_mult, mapa_veto_local, mapa_roi, df_audit_retorno


mapa_mult_desvio, mapa_veto, mapa_ajuste_roi, df_audit_cache = construir_calibracao_hibrida_v2(
    sh, df_bruto, df_val_mem=df_val, q_table_mem=q_table
)
n_obs_rl_map = contar_obs_por_estado(df_val)

# ==============================================================================
# 2. INDICADORES (via cache incremental) + FEATURES DE CORRELAÇÃO
# ==============================================================================
cache_indicadores_persistido = carregar_cache_indicadores()
df_indice_racao = construir_indice_racao(df_bruto)

cache_indicadores = {}
for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores_incremental(df_bruto, item, cache_indicadores_persistido)
        if d_it.empty:
            continue
        d_it = adicionar_features_correlacao(d_it, item, df_indice_racao)
        cache_indicadores[item] = d_it
        # Propaga de volta pro cache persistido (que só guarda as colunas
        # base — correlação é recalculada a cada ciclo por ser barata)
        cache_indicadores_persistido[item] = {
            'df': cache_indicadores_persistido.get(item, {}).get('df', d_it),
            'ultimo_ts': d_it['Timestamp'].max() if not d_it.empty else None
        }
    except Exception as e:
        print(f"⚠️ Indicadores falharam para '{item}': {e}")
        continue

# ==============================================================================
# 3. MODELO POOLED (treino diário, inferência todo ciclo)
# ==============================================================================
df_treino_pooled, df_teste_pooled = montar_datasets_pooled(cache_indicadores)

if deve_retreinar_pooled():
    print("🏋️ Retreinando modelo pooled (V2) — cadência diária...")
    modelo_pooled = treinar_modelo_pooled(df_treino_pooled)
    categorias_item = sorted(df_treino_pooled['Item_Cat'].unique().tolist()) if not df_treino_pooled.empty else None
    if modelo_pooled is not None:
        salvar_modelo_pooled(modelo_pooled, categorias_item)
else:
    modelo_pooled, categorias_item = carregar_modelo_pooled()
    if modelo_pooled is None and not df_treino_pooled.empty:
        # Fallback de segurança — nunca treinado ainda nesta run
        modelo_pooled = treinar_modelo_pooled(df_treino_pooled)
        categorias_item = sorted(df_treino_pooled['Item_Cat'].unique().tolist())
        if modelo_pooled is not None:
            salvar_modelo_pooled(modelo_pooled, categorias_item)

# --- Métricas de auditoria (sempre frescas, mesmo sem retreino) ---
audit_linhas = []
if modelo_pooled is not None and not df_teste_pooled.empty:
    cols_modelo = X_COLS_POOLED + ['Item_Cat']
    pred_teste = modelo_pooled.predict(df_teste_pooled[cols_modelo])
    df_teste_eval = df_teste_pooled.copy()
    df_teste_eval['Pred'] = pred_teste

    for item_aud, grupo in df_teste_eval.groupby('Item_Cat', observed=True):
        if len(grupo) < 5:
            continue
        erros = grupo['Pred'].values - grupo['Preco_Futuro'].values
        mae = float(np.mean(np.abs(erros)))
        rmse = float(np.sqrt(np.mean(erros ** 2)))
        preco_base = grupo['Price'].values
        direcao_real = (grupo['Preco_Futuro'].values > preco_base).astype(int)
        direcao_pred = (grupo['Pred'].values > preco_base).astype(int)
        acerto_direcao = float(np.mean(direcao_real == direcao_pred) * 100)
        roi_medio_previsto = float(np.mean((grupo['Pred'].values - preco_base) / preco_base * 100))
        n_treino_item = int((df_treino_pooled['Item_Cat'] == item_aud).sum()) if not df_treino_pooled.empty else 0

        audit_linhas.append([
            obter_agora_brt().strftime("%d/%m/%Y %H:%M"), str(item_aud),
            round(mae, 4), round(rmse, 4), round(acerto_direcao, 1),
            n_treino_item, len(grupo), round(roi_medio_previsto, 2)
        ])

# --- Ranking / inferência de produção (última linha de cada item) ---
ranking = []
itens_no_modelo = set(categorias_item) if categorias_item else set()

for item in df_bruto['Item'].unique():
    d_it = cache_indicadores.get(item)
    if d_it is None or d_it.empty or modelo_pooled is None or item not in itens_no_modelo:
        continue
    if d_it[X_COLS_BASE].dropna().empty:
        continue

    linha_pred = d_it[X_COLS_POOLED].tail(1).copy()
    linha_pred['Item_Cat'] = pd.Categorical([item], categories=list(categorias_item))
    pred = modelo_pooled.predict(linha_pred[X_COLS_POOLED + ['Item_Cat']])[0]
    preco_atual = d_it.iloc[-1]['Price']
    if preco_atual <= 0:
        continue

    roi = ((pred - preco_atual) / preco_atual) * 100
    roi_ajustado = roi + mapa_ajuste_roi.get(item, 0.0)
    ranking.append({'Item': item, 'ROI': roi, 'ROI_Ajustado': roi_ajustado})

df_ranking = (
    pd.DataFrame(ranking).sort_values(by='ROI_Ajustado', ascending=False)
    if ranking else pd.DataFrame(columns=['Item', 'ROI', 'ROI_Ajustado'])
)

# ==============================================================================
# 4. GERAÇÃO DE SINAIS HIERÁRQUICOS (lógica idêntica ao V1, ver bot_4_24.py)
# ==============================================================================
try:
    max_id_existente = int(pd.to_numeric(df_val['ID_Sinal'], errors='coerce').max()) if not df_val.empty else 0
except Exception:
    max_id_existente = len(df_val)
prox_id_sinal = max_id_existente + 1
novas_linhas_validacao = []
lista_final = []

mapa_ultimo_sinal = {}
usar_dedup_simples = False
if not df_val.empty:
    try:
        for _, row_u in df_val.iterrows():
            chave_u = f"{str(row_u['Item']).strip()}_{str(row_u['Tipo']).strip().upper()}"
            ts_u = parse_data_sinal_robusta(row_u['Data_Sinal'])
            if pd.isnull(ts_u):
                continue
            preco_u = limpar_numero(row_u['Preco_Sinal'])
            if chave_u not in mapa_ultimo_sinal or ts_u > mapa_ultimo_sinal[chave_u]['timestamp']:
                mapa_ultimo_sinal[chave_u] = {'preco': preco_u, 'timestamp': ts_u}
    except Exception as e:
        print(f"⚠️  mapa_ultimo_sinal (V2) falhou: {e} — revertendo para deduplicação por data.")
        mapa_ultimo_sinal = {}
        usar_dedup_simples = True

for item in df_ranking['Item']:
    d_it = cache_indicadores.get(item)
    if d_it is None or d_it.empty:
        continue
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item'] == item]['ROI_Ajustado'].values[0]
    saldo_item = inv_dict.get(item, 0)

    if saldo_item <= 0 and row['Price'] > (row['Media_Macro'] - mapa_mult_desvio.get(item, 2.0) * row['Desvio_Macro']):
        continue
    if pd.isna(row.get('Media_Macro')) or pd.isna(row.get('Desvio_Macro')):
        continue

    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira / saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1

    valor_posicao = saldo_item * row['Price']
    if origem == 2 and valor_posicao < POSICAO_MINIMA_SFL:
        origem = 1

    preco_atual = row['Price']
    m_desvio = mapa_mult_desvio.get(item, 2.0)
    alvo_in = row['Media_Macro'] - (m_desvio * row['Desvio_Macro'])
    alvo_out_estatistico = row['Media_Macro'] + (m_desvio * row['Desvio_Macro'])

    if origem == 2 and custo_medio > 0:
        alvo_lucro = (custo_medio * (1 + LUCRO_REAL_DESEJADO)) / (1 - TAXA_MARKETPLACE)
        alvo_out = max(alvo_lucro, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico
        alvo_lucro = 0.0

    margem_atencao = calcular_margem_atencao(alvo_in, alvo_out)

    estado_rl = construir_estado_rl_hibrido(
        row_mercado=row, item=item, origem=origem, custo_medio=custo_medio,
        alvo_lucro=alvo_lucro, df_trade=df_trade, roi_pred=roi_pred
    )
    # NOVO — coleta de dados para futura V2 do RL (ver docstring da função)
    dias_custodia_atual = calcular_dias_custodia(item, df_trade) if origem == 2 else None
    estado_rl_continuo = construir_estado_rl_continuo(row, origem, custo_medio, roi_pred, dias_custodia_atual)

    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in + margem_atencao:
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER (Lucro Disponível)" if (origem == 2 and alvo_lucro > alvo_out_estatistico) else "🔴 VENDER"
    elif preco_atual >= alvo_out - margem_atencao:
        acao = "🎯 PERTO DA SAÍDA"

    if origem == 2 and preco_atual < custo_medio:
        deve_stopar, _thr_stop = calcular_stop_adaptativo(item, preco_atual, roi_pred / 100, df_audit_mem=df_audit_cache)
        if deve_stopar:
            acao = "⚠️ VENDER (Stop Loss)"
        elif acao == "ESPERAR":
            acao = "ESPERAR (Prejuízo)"

    if acao == "🟢 COMPRAR" and mapa_veto.get(item, False):
        acao = "🚫 COMPRA VETADA"

    roi_pred_exibido = roi_pred
    if acao == "🟢 COMPRAR":
        chave_ci_atual = f"{item}|{estado_rl}"
        q_item_dados = q_table_item.get(chave_ci_atual, {})
        q_compra_item = q_item_dados.get("COMPRA")
        n_obs_item_at = q_item_dados.get("N_COMPRA", 0)

        vetado_rl, bonus_roi_rl = calcular_veto_rl(
            estado_rl, q_table, n_obs_rl_map, q_compra_item=q_compra_item, n_obs_item=n_obs_item_at
        )
        roi_pred_exibido = roi_pred + bonus_roi_rl
        if vetado_rl:
            acao = "🚫 VETO DO CÉREBRO RL (Histórico de Prejuízo)"

    em_crash, var24h = detectar_regime(cache_indicadores.get(item))
    if em_crash and "COMPRAR" in acao:
        acao = f"⚠️ COMPRAR (QUEDA {var24h:.1f}% em 24h)"

    lista_final.append([
        item,
        round(roi_pred_exibido * (1 - TAXA_MARKETPLACE) - TAXA_MARKETPLACE * 100, 1),
        acao, round(alvo_in, 4), round(alvo_out, 4),
        "Revenda" if origem == 2 else "Colheita", row['Season'],
        f"'{estado_rl}", obter_agora_brt().strftime("%d/%m %H:%M")
    ])

    if acao in ["🟢 COMPRAR", "🔴 VENDER", "🔴 VENDER (Lucro Disponível)", "⚠️ VENDER (Stop Loss)"] \
            or (em_crash and "COMPRAR" in acao):
        agora_brt = obter_agora_brt()
        dt_hoje_str = agora_brt.strftime("%d%m%Y")
        dt_hhmm = agora_brt.strftime("%H%M")
        tipo_sinal = "COMPRA" if "COMPRAR" in acao else "VENDA"
        sufixo = "_STOP" if "Stop Loss" in acao else ""

        INTERVALO_MIN_H = 4
        VARIACAO_MIN_PCT = 0.03

        chave_item_tipo = f"{item}_{tipo_sinal.upper()}"
        ultimo = mapa_ultimo_sinal.get(chave_item_tipo)
        pode_gerar = True

        if pode_gerar and ultimo:
            horas_desde = (agora_brt - ultimo['timestamp']).total_seconds() / 3600
            variacao = abs(preco_atual - ultimo['preco']) / ultimo['preco'] if ultimo['preco'] > 0 else 1.0
            if horas_desde >= 24:
                pode_gerar = True
            elif horas_desde < INTERVALO_MIN_H:
                pode_gerar = False
            elif not sufixo and variacao < VARIACAO_MIN_PCT:
                pode_gerar = False

        chave_unica = f"{item}_{tipo_sinal}_{dt_hoje_str}_{dt_hhmm}{sufixo}"

        if pode_gerar:
            if not any(x[10] == chave_unica for x in novas_linhas_validacao):
                if usar_dedup_simples:
                    chave_fallback = f"{item}_{tipo_sinal}_{dt_hoje_str}{sufixo}"
                    if not df_val.empty and df_val['Chave_Unica'].astype(str).str.startswith(chave_fallback).any():
                        continue

                linha_sinais_hoje = len(lista_final)
                enviar_telegram(acao, item, preco_atual, linha_sinais_hoje)

                if tipo_sinal == "COMPRA":
                    ref_financeira = alvo_out
                elif sufixo == "_STOP":
                    ref_financeira = preco_atual
                elif origem == 1:
                    ref_financeira = row['Media_Macro']
                else:
                    ref_financeira = custo_medio

                novas_linhas_validacao.append([
                    int(prox_id_sinal), tipo_sinal, agora_brt.strftime("%d/%m/%Y %H:%M"), item,
                    round(preco_atual, 4), round(ref_financeira, 4), round(preco_atual, 4),
                    "Aguardando", 0, f"'{estado_rl}", chave_unica,
                    "", "", estado_rl_continuo
                ])
                prox_id_sinal += 1
                mapa_ultimo_sinal[chave_item_tipo] = {'preco': preco_atual, 'timestamp': agora_brt}

if novas_linhas_validacao:
    ws_val.append_rows(novas_linhas_validacao, value_input_option='USER_ENTERED')
    print(f"📌 {len(novas_linhas_validacao)} novos sinais (V2) integrados para monitorização.")

# ==============================================================================
# 5. PERSISTÊNCIA — Sinais_Hoje, XGB_Audit, Cerebro_RL, Cerebro_RL_Item
# ==============================================================================
HEADERS_HOJE = ["Item", "ROI_Prev_%", "Ação", "Alvo_Entrada", "Alvo_Saída",
                "Origem", "Estação", "Estado_RL", "Atualizado"]

try:
    try:
        ws_hoje = sh.worksheet("Sinais_Hoje")
    except Exception:
        ws_hoje = sh.add_worksheet(title="Sinais_Hoje", rows="200", cols="9")
        set_frozen(ws_hoje, rows=1)
        format_cell_range(ws_hoje, 'A1:I1', CellFormat(textFormat=TextFormat(bold=True)))

    ws_hoje.clear()
    ws_hoje.update(values=([HEADERS_HOJE] + lista_final), range_name='A1', value_input_option='USER_ENTERED')
    print(f"📋 Sinais_Hoje (V2): {len(lista_final)} itens gravados.")
except Exception as e:
    print(f"❌ Erro ao salvar Sinais_Hoje (V2): {e}")

try:
    try:
        ws_audit = sh.worksheet("XGB_Audit")
    except Exception:
        n_itens_init = df_bruto['Item'].nunique()
        rows_init = n_itens_init * 72 * 2 + 500
        ws_audit = sh.add_worksheet(title="XGB_Audit", rows=str(rows_init), cols="8")
        ws_audit.update(values=[HEADERS_AUDIT], range_name='A1', value_input_option='USER_ENTERED')
        set_frozen(ws_audit, rows=1)
        format_cell_range(ws_audit, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))

    JANELA_AUDIT_H = 72
    n_itens_reais = df_bruto['Item'].nunique()
    RETENCAO_AUDIT = n_itens_reais * JANELA_AUDIT_H * 2

    if audit_linhas:
        dados_atuais = ws_audit.get_all_values()
        header = dados_atuais[0] if dados_atuais else HEADERS_AUDIT
        linhas_existentes = dados_atuais[1:] if len(dados_atuais) > 1 else []
        todas = linhas_existentes + audit_linhas
        if len(todas) > RETENCAO_AUDIT:
            todas = todas[-RETENCAO_AUDIT:]
        ws_audit.clear()
        ws_audit.update(values=([header] + todas), range_name='A1', value_input_option='USER_ENTERED')
        print(f"📊 XGB_Audit (V2): {len(audit_linhas)} novos | {len(todas)} retidos.")
    else:
        print("📊 XGB_Audit (V2): nenhum item com dados suficientes para auditoria.")
except Exception as e:
    print(f"❌ Erro ao salvar XGB_Audit (V2): {e}")

try:
    cerebro_linhas = [
        [f"'{st}", round(acoes.get("COMPRA", 0.0), 4), round(acoes.get("VENDA", 0.0), 4),
         int(acoes.get("N_COMPRA", 0)), int(acoes.get("N_VENDA", 0))]
        for st, acoes in q_table.items()
    ]
    ok_cerebro = salvar_aba_atomica(sh, "Cerebro_RL", ["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
                                     cerebro_linhas, rows_est=5000, cols_est=5)
    print("✅ Cerebro_RL (V2) sincronizado!" if ok_cerebro else "⚠️ Cerebro_RL (V2) não atualizado neste ciclo.")
except Exception as e:
    print(f"❌ Erro ao salvar Cerebro_RL (V2): {e}")

try:
    cerebro_item_linhas = [
        [f"'{chave_ci}", round(acoes.get("COMPRA", 0.0), 4), round(acoes.get("VENDA", 0.0), 4),
         int(acoes.get("N_COMPRA", 0)), int(acoes.get("N_VENDA", 0))]
        for chave_ci, acoes in q_table_item.items()
    ]
    ok_cerebro_item = salvar_aba_atomica(sh, "Cerebro_RL_Item",
                                          ["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
                                          cerebro_item_linhas, rows_est=20000, cols_est=5)
    print("✅ Cerebro_RL_Item (V2) sincronizado!" if ok_cerebro_item else "⚠️ Cerebro_RL_Item (V2) não atualizado.")
except Exception as e:
    print(f"❌ Erro ao salvar Cerebro_RL_Item (V2): {e}")

# ==============================================================================
# 6. NOVO — CAMADA DE PORTFÓLIO (relatório informativo, não bloqueia sinal)
# ==============================================================================
try:
    precos_atuais = {item: cache_indicadores[item].iloc[-1]['Price']
                     for item in cache_indicadores if not cache_indicadores[item].empty}
    df_portfolio = calcular_exposicao_por_cluster(inv_dict, precos_atuais)

    if not df_portfolio.empty:
        try:
            ws_port = sh.worksheet("Portfolio_Correlacao")
        except Exception:
            ws_port = sh.add_worksheet(title="Portfolio_Correlacao", rows="50", cols="4")
            set_frozen(ws_port, rows=1)
            format_cell_range(ws_port, 'A1:D1', CellFormat(textFormat=TextFormat(bold=True)))

        ws_port.clear()
        ws_port.update(values=[df_portfolio.columns.tolist()] + df_portfolio.values.tolist(),
                        range_name='A1', value_input_option='USER_ENTERED')
        print(f"📦 Portfolio_Correlacao (V2): {len(df_portfolio)} clusters — informativo, não bloqueia sinal.")
except Exception as e:
    print(f"⚠️ Erro ao calcular camada de portfólio (V2): {e}")

# ==============================================================================
# 7. SALVAR CACHE DE INDICADORES EM DISCO (para o próximo ciclo, via actions/cache)
# ==============================================================================
try:
    salvar_cache_indicadores(cache_indicadores_persistido)
    print("💾 Cache incremental de indicadores (V2) salvo em disco.")
except Exception as e:
    print(f"⚠️ Falha ao salvar cache de indicadores (V2): {e}")

print("🏁 Ciclo V2 (SOMBRA) finalizado.")
