import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
import requests
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 12 
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.08
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

# Configurações de Estratégia
MARGEM_ATENCAO = 0.05    
TRAVA_PREJUIZO = True     
LIMITE_BANCA = 0.25       
TOLERANCIA_SUJEIRA = 0.15 

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==============================================================================
# FUNÇÕES DE APOIO
# ==============================================================================

def enviar_telegram(mensagem):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": mensagem})
    except Exception as e: print(f"Erro Telegram: {e}")

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

# ==============================================================================
# INTELIGÊNCIA E FIFO
# ==============================================================================

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
    if len(d) < 15: return d
    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    d['Lag_1'] = d['Price'].shift(1)
    d['Volatilidade'] = d['Price'].rolling(window=REGISTROS_POR_DIA).std()
    d['Media_120'] = d['Price'].rolling(window=120 * REGISTROS_POR_DIA, min_periods=1).mean()
    d['Desvio_120'] = d['Price'].rolling(window=120 * REGISTROS_POR_DIA, min_periods=1).std()
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=SHIFT_ALVO).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=SHIFT_ALVO).mean()
    d['RSI'] = (100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))).fillna(50)
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    return d

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🚀 Iniciando Bot com FIFO Híbrido e Alertas...")
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)
sh = gc.open(NOME_PLANILHA)

# 1. Dados de Inventário e SFL
inv_data = sh.worksheet("Inventario").get_all_values()
inv_dict = {row[0]: limpar_numero(row[1]) for row in inv_data[1:]}

try:
    sfl_data = sh.worksheet("RegistroBruto").get_all_values()
    df_sfl = pd.DataFrame(sfl_data[1:], columns=sfl_data[0])
    saldo_sfl = limpar_numero(df_sfl[df_sfl['Recurso'] == 'Flower'].iloc[-1]['Quantidade'])
except:
    saldo_sfl = 0

# 2. Cálculos de Base
mapa_custo_fifo = calcular_fifo_hibrido(sh, inv_dict)
trade_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()

# 3. Processamento RecLog e IA
rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

cache_indicadores = {}
ranking = []

for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it
        d_treino = d_it.dropna(subset=['Preco_Futuro', 'RSI'])
        if len(d_treino) < 30: continue
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3).fit(d_treino[['Price', 'Lag_1', 'RSI']], d_treino['Preco_Futuro'], sample_weight=d_treino['Peso_Temporal'])
        pred = model.predict(d_it[['Price', 'Lag_1', 'RSI']].tail(1))[0]
        roi = ((pred - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        ranking.append({'Item': item, 'ROI': roi})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False) if ranking else pd.DataFrame(columns=['Item', 'ROI'])

# ==============================================================================
# 4. SINAIS, ALVOS E TELEGRAM (HIERARQUIA CORRIGIDA)
# ==============================================================================
lista_final, alertas_msg = [], []

for item in df_ranking.head(30)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    saldo_item = inv_dict.get(item, 0)
    custo_medio = mapa_custo_fifo.get(item, row['Media_120'])
    
    # Origem
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1
    
    # Definição dos Alvos
    preco_atual = row['Price']
    alvo_in = row['Media_120'] - (2.0 * row['Desvio_120'])
    alvo_out_estatistico = row['Media_120'] + (2.0 * row['Desvio_120'])
    
    # Ajuste do Alvo de Saída (Garante lucro real se for revenda)
    if origem == 2 and custo_medio > 0:
        alvo_lucro_minimo = custo_medio * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
        alvo_out = max(alvo_lucro_minimo, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico

    # --- Árvore de Decisão ---
    acao = "ESPERAR"
    
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in * (1 + MARGEM_ATENCAO):
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER"
    elif preco_atual >= alvo_out * (1 - MARGEM_ATENCAO):
        acao = "🎯 PERTO DA SAÍDA"

    # Trava de Segurança contra Prejuízo
    if ("VENDER" in acao or "SAÍDA" in acao) and origem == 2 and preco_atual < custo_medio:
        if roi_pred < -20: acao = "⚠️ VENDER (Stop Loss)"
        else: acao = "ESPERAR (Prejuízo)"

    # --- Geração de Alertas Telegram ---
    if acao == "🟢 COMPRAR":
        qtd = (saldo_sfl * LIMITE_BANCA) / preco_atual if preco_atual > 0 else 0
        alertas_msg.append(f"🟢 COMPRA SNIPER: {item}\nPreço: {preco_atual}\nAlvo In: {alvo_in:.4f}\nSugestão: {int(qtd)} un")
    elif acao == "🎯 PERTO DA SAÍDA" and saldo_item > 0:
        distancia = (1 - (preco_atual / alvo_out)) * 100
        alertas_msg.append(f"🎯 PREPARE A VENDA: {item}\nPreço: {preco_atual}\nFaltam subir {distancia:.1f}% para bater o lucro alvo!")
    elif acao == "🔴 VENDER" and saldo_item > 0:
        alertas_msg.append(f"🔴 HORA DE VENDER: {item}\nPreço: {preco_atual}\nCusto Médio: {custo_medio:.4f}")

    lista_final.append([item, round(roi_pred, 1), acao, round(alvo_in, 4), round(alvo_out, 4), "Revenda" if origem==2 else "Colheita", row['Season'], datetime.now().strftime("%d/%m %H:%M")])

# ==============================================================================
# 5. SALVAMENTO NA PLANILHA
# ==============================================================================
print(f"📝 Escrevendo {len(lista_final)} sinais na planilha...")
try:
    try:
        ws_sinais = sh.worksheet("Sinais_Hoje")
    except:
        ws_sinais = sh.add_worksheet(title="Sinais_Hoje", rows="100", cols="10")

    cabecalho = [["Item", "ROI Prev %", "Decisão", "Entrada Sniper", "Saída Alvo", "Origem", "Estação", "Atualização"]]
    ws_sinais.clear()
    ws_sinais.update(values=(cabecalho + lista_final), range_name='A1')
    
    set_frozen(ws_sinais, rows=1)
    format_cell_range(ws_sinais, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Sinais_Hoje atualizada!")
except Exception as e:
    print(f"❌ Erro ao salvar planilha: {e}")

if alertas_msg: 
    print("📡 Enviando alertas...")
    enviar_telegram("\n\n".join(alertas_msg))

print("🏁 Ciclo finalizado com sucesso!")
