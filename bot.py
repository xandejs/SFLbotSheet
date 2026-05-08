import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 12 
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.07  # 7% de lucro real alvo
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

# Configurações de Estratégia
MARGEM_ATENCAO = 0.035    # 3.5% para sinalizar proximidade
TRAVA_PREJUIZO = True     
LIMITE_BANCA = 0.25       
TOLERANCIA_SUJEIRA = 0.15 

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# ==============================================================================
# FUNÇÕES DE APOIO E INTELIGÊNCIA
# ==============================================================================

def limpar_numero(texto):
    if not texto: return 0.0
    s = str(texto).strip()
    if '.' in s and ',' in s: s = s.replace('.', '')
    s = s.replace(',', '.')
    if s.count('.') > 1:
        partes = s.split('.')
        s = "".join(partes[:-1]) + "." + partes[-1]
    try: return float(s)
    except: return 0.0

def calcular_fifo_hibrido(sh, inventario_dict):
    try:
        ws = sh.worksheet("TradeLog")
        log_data = ws.get_all_values()
        df_log = pd.DataFrame(log_data[1:], columns=log_data[0])
        df_log['Qtd'] = df_log['Qtd'].apply(limpar_numero)
        df_log['SFL_Total'] = df_log['SFL'].apply(limpar_numero).abs()
        df_compras = df_log[df_log['Tipo'].str.contains("Compra", case=False)].copy()
        
        fifo_map = {}
        for item, saldo_real in inventario_dict.items():
            if saldo_real <= 0: continue
            compras_item = df_compras[df_compras['Nome do Item'] == item].iloc[::-1]
            qtd_acumulada, custo_total = 0, 0
            for _, row in compras_item.iterrows():
                falta = saldo_real - qtd_acumulada
                if falta <= 0: break
                usar = min(row['Qtd'], falta)
                custo_total += (usar * (row['SFL_Total'] / row['Qtd'] if row['Qtd'] > 0 else 0))
                qtd_acumulada += usar
            fifo_map[item] = custo_total / qtd_acumulada if qtd_acumulada > 0 else 0.0
        return fifo_map
    except Exception as e:
        print(f"Erro FIFO: {e}"); return {}

def preparar_indicadores(df_original, nome_item):
    d = df_original[df_original['Item'] == nome_item].copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], dayfirst=True, errors='coerce')
    d = d.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    
    if len(d) < 30: return d

    # --- JANELA MACRO (30 dias / 360 registros) para Alvos ---
    d['Media_Macro'] = d['Price'].rolling(window=360, min_periods=30).mean()
    d['Desvio_Macro'] = d['Price'].rolling(window=360, min_periods=30).std()

    # --- JANELA MICRO (10 dias / 120 registros) para Momentum ---
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=120, min_periods=10).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=120, min_periods=10).mean()
    d['RSI_Micro'] = (100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))).fillna(50)

    # Dados para IA
    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    d['Lag_1'] = d['Price'].shift(1)
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    
    # Peso Temporal (Dados recentes valem mais para o XGBoost)
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))
    
    return d

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🚀 Iniciando Bot com Dual-Window Strategy...")
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)
sh = gc.open(NOME_PLANILHA)

# 1. Dados de Base
inv_data = sh.worksheet("Inventario").get_all_values()
inv_dict = {row[0]: limpar_numero(row[1]) for row in inv_data[1:]}

try:
    sfl_data = sh.worksheet("RegistroBruto").get_all_values()
    df_sfl = pd.DataFrame(sfl_data[1:], columns=sfl_data[0])
    saldo_sfl = limpar_numero(df_sfl[df_sfl['Recurso'] == 'Flower'].iloc[-1]['Quantidade'])
except:
    saldo_sfl = 0

mapa_custo_fifo = calcular_fifo_hibrido(sh, inv_dict)
trade_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()

# 2. IA e Processamento
rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

cache_indicadores = {}
ranking = []

for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it
        d_tr = d_it.dropna(subset=['Preco_Futuro', 'Media_Macro'])
        if len(d_tr) < 30: continue
        
        # Treino rápido com foco na tendência mensal vs momento semanal
        model = xgb.XGBRegressor(n_estimators=40, max_depth=3).fit(
            d_tr[['Price', 'Lag_1', 'RSI_Micro']], d_tr['Preco_Futuro'], 
            sample_weight=d_tr['Peso_Temporal']
        )
        
        pred = model.predict(d_it[['Price', 'Lag_1', 'RSI_Micro']].tail(1))[0]
        roi = ((pred - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        ranking.append({'Item': item, 'ROI': roi})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False) if ranking else pd.DataFrame(columns=['Item', 'ROI'])

# 3. Geração de Sinais
lista_final = []

for item in df_ranking.head(30)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    saldo_item = inv_dict.get(item, 0)
    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    
    # Origem e Preços
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1
    
    preco_atual = row['Price']
    alvo_in = row['Media_Macro'] - (2.0 * row['Desvio_Macro'])
    alvo_out_estatistico = row['Media_Macro'] + (2.0 * row['Desvio_Macro'])
    
    # Alvo de Saída Financeiro para Revenda
    if origem == 2 and custo_medio > 0:
        alvo_lucro = custo_medio * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
        alvo_out = max(alvo_lucro, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico

    # Decisão Hierárquica
    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in * (1 + MARGEM_ATENCAO):
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER"
    elif preco_atual >= alvo_out * (1 - MARGEM_ATENCAO):
        acao = "🎯 PERTO DA SAÍDA"

    # Trava de Segurança
    if ("VENDER" in acao or "SAÍDA" in acao) and origem == 2 and preco_atual < custo_medio:
        acao = "⚠️ VENDER (Stop Loss)" if roi_pred < -20 else "ESPERAR (Prejuízo)"

    lista_final.append([item, round(roi_pred, 1), acao, round(alvo_in, 4), round(alvo_out, 4), "Revenda" if origem==2 else "Colheita", row['Season'], datetime.now().strftime("%d/%m %H:%M")])

# 4. Salvamento
try:
    ws_sinais = sh.worksheet("Sinais_Hoje")
    ws_sinais.clear()
    cabecalho = [["Item", "ROI Prev %", "Decisão", "Entrada Sniper", "Saída Alvo", "Origem", "Estação", "Atualização"]]
    ws_sinais.update(values=(cabecalho + lista_final), range_name='A1')
    set_frozen(ws_sinais, rows=1)
    format_cell_range(ws_sinais, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Planilha Sinais_Hoje atualizada com Estratégia de Janela Dupla!")
except Exception as e:
    print(f"❌ Erro ao salvar: {e}")

print("🏁 Ciclo finalizado!")
