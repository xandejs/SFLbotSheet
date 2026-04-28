import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import random
from datetime import datetime
from gspread_formatting import CellFormat, Color, TextFormat, format_cell_ranges
from oauth2client.service_account import ServiceAccountCredentials

# Configurações do Mercado
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 12 
TAXA_MARKETPLACE = 0.1 
DIAS_SWING = 7           
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# ==============================================================================
# FUNÇÕES DE APOIO
# ==============================================================================

def preparar_indicadores(df_original, nome_item):
    d = df_original[df_original['Item'] == nome_item].copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], format='%d/%m/%Y', errors='coerce')
    d = d.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    
    # Técnicos
    d['Lag_1'] = d['Price'].shift(1)
    d['Media_15'] = d['Price'].rolling(window=15 * REGISTROS_POR_DIA).mean()
    d['Volatilidade'] = d['Price'].rolling(window=REGISTROS_POR_DIA).std()
    
    # 🛡️ Visão 120 dias
    JANELA_120 = 120 * REGISTROS_POR_DIA
    d['Media_120'] = d['Price'].rolling(window=JANELA_120, min_periods=1).mean()
    d['Desvio_120'] = d['Price'].rolling(window=JANELA_120, min_periods=1).std()
    
    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=SHIFT_ALVO).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=SHIFT_ALVO).mean()
    d['RSI'] = 100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))
    
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    return d

class SFL_Environment:
    def __init__(self, df_processado):
        self.df = df_processado.dropna(subset=['RSI', 'Volatilidade']).reset_index(drop=True)
        self.reset()
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        # Bins de RSI (0 a 4)
        rsi_val = row['RSI'] if not np.isnan(row['RSI']) else 50
        rsi_bin = int(min(4, max(0, rsi_val // 20)))
        volat_bin = 1 if row['Volatilidade'] > self.df['Volatilidade'].mean() else 0
        ia_sinal = 1 if row['Preco_Futuro'] > row['Price'] * (1 + TAXA_MARKETPLACE) else 0
        estacao = int(row['Season_Num']) 
        return (rsi_bin, volat_bin, ia_sinal, estacao)

    def step(self, action):
        preco_hoje = self.df.iloc[self.current_step]['Price']
        if self.current_step + SHIFT_ALVO >= len(self.df):
            return self._get_state(), 0, True

        self.current_step += 1
        preco_novo = self.df.iloc[self.current_step]['Price']
        
        variacao_pct = ((preco_novo - preco_hoje) / preco_hoje) * 100 if preco_hoje > 0 else 0
            
        if action == 1:   # COMPRAR
            reward = variacao_pct - (TAXA_MARKETPLACE * 100)
        elif action == 2: # VENDER
            reward = -variacao_pct 
        else:             # ESPERAR
            reward = 0
            
        done = (self.current_step >= len(self.df) - SHIFT_ALVO - 1)
        return self._get_state(), reward, done

def gerenciar_memoria(gc, nome_pl, q_tab=None, modo="carregar"):
    sh = gc.open(nome_pl)
    try: ws = sh.worksheet("Cerebro_RL")
    except: ws = sh.add_worksheet(title="Cerebro_RL", rows="1500", cols="7")
    
    if modo == "carregar":
        dados = ws.get_all_values()
        nova_tab = {}
        if len(dados) > 1:
            for l in dados[1:]: 
                if len(l) < 7: continue 
                try:
                    estado = (int(l[0]), int(l[1]), int(l[2]), int(l[3]))
                    pesos = [float(l[4]), float(l[5]), float(l[6])]
                    nova_tab[estado] = pesos
                except: continue
        return nova_tab
    else:
        corpo = [["RSI", "Volat", "XGB", "Estacao", "Q_Esperar", "Q_Comprar", "Q_Vender"]]
        for est, pes in q_tab.items(): 
            corpo.append([est[0], est[1], est[2], est[3], round(pes[0],4), round(pes[1],4), round(pes[2],4)])
        ws.clear()
        ws.update(values=corpo, range_name='A1')

def salvar_sinais_hoje(gc, nome_pl, lista_sinais):
    sh = gc.open(nome_pl)
    try: ws = sh.worksheet("Sinais_Hoje")
    except: ws = sh.add_worksheet(title="Sinais_Hoje", rows="60", cols="5")
    
    cabecalho = [["Item", "ROI Previsto %", "Decisão Final", "Estação", "Atualização"]]
    ws.clear()
    ws.update(values=(cabecalho + lista_sinais), range_name='A1')

    fmt_compra = CellFormat(backgroundColor=Color(0.8, 1.0, 0.8), textFormat=TextFormat(bold=True, foregroundColor=Color(0,0.4,0)))
    fmt_espera = CellFormat(backgroundColor=Color(1.0, 1.0, 0.8))
    fmt_vender = CellFormat(backgroundColor=Color(1.0, 0.8, 0.8), textFormat=TextFormat(bold=True, foregroundColor=Color(0.6,0,0)))
    
    formatos = [('A1:E1', CellFormat(textFormat=TextFormat(bold=True)))]
    for i, linha in enumerate(lista_sinais):
        celula = f"C{i+2}"
        if "COMPRAR" in linha[2]: formatos.append((celula, fmt_compra))
        elif "ESPERAR" in linha[2]: formatos.append((celula, fmt_espera))
        elif "VENDER" in linha[2]: formatos.append((celula, fmt_vender))
    
    format_cell_ranges(ws, formatos)

# ==============================================================================
# EXECUÇÃO PRINCIPAL (Versão GitHub Actions)
# ==============================================================================

print("Iniciando bot trader autônomo...")

# 1. Configura as credenciais da Conta de Serviço
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)

# 2. Faz o login silencioso
gc = gspread.authorize(credenciais)

# 3. Baixa os dados
dados_brutos = gc.open(NOME_PLANILHA).worksheet("RecLog").get_all_values()[1:]

df_bruto = pd.DataFrame([l[:4] for l in dados_brutos], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = pd.to_numeric(df_bruto['Price'].str.replace(',', '.'), errors='coerce')
df_bruto = df_bruto.dropna(subset=['Price'])

print("📡 Analisando Sazonalidade e ROI (XGBoost)...")
ranking = []
for item in df_bruto['Item'].unique():
    d_it = preparar_indicadores(df_bruto, item)
    d_treino = d_it.dropna(subset=['Preco_Futuro', 'RSI', 'Lag_1'])
    if len(d_treino) < 30: continue
    
    X = d_treino[['Price', 'Lag_1', 'RSI', 'Season_Num']]
    y = d_treino['Preco_Futuro']
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1).fit(X, y)
    
    X_hoje = d_it[['Price', 'Lag_1', 'RSI', 'Season_Num']].tail(1)
    preco_previsto = model.predict(X_hoje)[0]
    roi = ((preco_previsto - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
    ranking.append({'Item': item, 'ROI': roi})

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False)
q_table = gerenciar_memoria(gc, NOME_PLANILHA, modo="carregar")

print("🧠 Treinando Cérebro RL (Memória 120d)...")
for item in df_ranking.head(15)['Item']:
    d_it = preparar_indicadores(df_bruto, item)
    if len(d_it) < 60: continue
    env = SFL_Environment(d_it)
    for ep in range(30): 
        state, done = env.reset(), False
        while not done:
            action = random.randint(0, 2) if random.random() < 0.3 else np.argmax(q_table.get(state, [0, 0, 0]))
            n_state, rew, done = env.step(action)
            
            old_q = q_table.get(state, [0, 0, 0])[action]
            q_table.setdefault(state, [0, 0, 0])[action] = old_q + 0.1 * (rew + 0.9 * np.max(q_table.get(n_state, [0, 0, 0])) - old_q)
            state = n_state

gerenciar_memoria(gc, NOME_PLANILHA, q_table, modo="salvar")

# Painel Final
lista_final = []
agora = datetime.now().strftime("%d/%m %H:%M")
for item in df_ranking.head(12)['Item']:
    d_it = preparar_indicadores(df_bruto, item)
    row = d_it.iloc[-1]
    
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    estado = (int(min(4, row['RSI'] // 20)), 1 if row['Volatilidade'] > d_it['Volatilidade'].mean() else 0, 1 if roi_pred > 15 else 0, int(row['Season_Num']))
    
    dec = np.argmax(q_table.get(estado, [0, 0, 0]))
    acao = ["ESPERAR", "COMPRAR", "VENDER"][dec]
    
    # 🛡️ BLOQUEIO ABSOLUTO (CONSERVADOR)
    if acao == "COMPRAR":
        limite_caro = row['Media_120'] + (1.5 * row['Desvio_120'])
        if row['Price'] > limite_caro:
            acao = "ESPERAR (Caro/120d)"

    lista_final.append([item, round(roi_pred, 1), acao, row['Season'], agora])

salvar_sinais_hoje(gc, NOME_PLANILHA, lista_final)
print("✅ Tudo pronto! A planilha Sinais_Hoje foi atualizada.")