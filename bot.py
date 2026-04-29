import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import random
from datetime import datetime
from gspread_formatting import CellFormat, Color, TextFormat, format_cell_ranges
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
# CONFIGURAÇÕES DO MERCADO (ESTRATÉGIA SNIPER + GESTÃO DE ESTOQUE)
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 12 
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.08  # 8% de lucro limpo acima da taxa
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# ==============================================================================
# FUNÇÕES DE APOIO E INTELIGÊNCIA
# ==============================================================================

def limpar_numero(texto):
    """Remove pontos de milhar e converte vírgula decimal para ponto"""
    if not texto: return 0.0
    s = str(texto).strip()
    # Se tiver ponto e vírgula, o ponto é milhar (ex: 1.626,40)
    if '.' in s and ',' in s:
        s = s.replace('.', '')
    # Troca a vírgula decimal por ponto
    s = s.replace(',', '.')
    # Caso ainda tenha ficado com mais de um ponto por erro de entrada
    if s.count('.') > 1:
        partes = s.split('.')
        s = "".join(partes[:-1]) + "." + partes[-1]
    try:
        return float(s)
    except:
        return 0.0

def preparar_indicadores(df_original, nome_item):
    d = df_original[df_original['Item'] == nome_item].copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], format='%d/%m/%Y', errors='coerce')
    d = d.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    
    d['Lag_1'] = d['Price'].shift(1)
    d['Volatilidade'] = d['Price'].rolling(window=REGISTROS_POR_DIA).std()
    
    JANELA_120 = 120 * REGISTROS_POR_DIA
    d['Media_120'] = d['Price'].rolling(window=JANELA_120, min_periods=1).mean()
    d['Desvio_120'] = d['Price'].rolling(window=JANELA_120, min_periods=1).std()
    
    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=SHIFT_ALVO).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=SHIFT_ALVO).mean()
    d['RSI'] = (100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))).fillna(50)
    
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    return d

def analisar_origem_estoque(gc, nome_pl):
    sh = gc.open(nome_pl)
    df_log = pd.DataFrame(sh.worksheet("TradeLog").get_all_values()[1:], 
                          columns=["Tipo", "Item", "Qtd", "SFL", "Parceiro", "Data"])
    df_inv = pd.DataFrame(sh.worksheet("Inventario").get_all_values()[1:], 
                          columns=["Item", "Quantidade"])

    # Limpeza robusta de números
    df_log['Qtd'] = df_log['Qtd'].apply(limpar_numero)
    compras_totais = df_log[df_log['Tipo'].str.contains("Compra", case=False)].groupby('Item')['Qtd'].sum()

    origem_map = {}
    for _, row in df_inv.iterrows():
        item = row['Item']
        saldo_atual = limpar_numero(row['Quantidade'])
        if saldo_atual <= 0: continue
        
        qtd_comprada = compras_totais.get(item, 0)
        origem_map[item] = 2 if qtd_comprada >= saldo_atual else 1
            
    return origem_map

class SFL_Environment:
    def __init__(self, df_processado, origem_item=0):
        self.df = df_processado.dropna(subset=['RSI', 'Volatilidade']).reset_index(drop=True)
        self.origem = origem_item 
        self.reset()
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        row = self.df.iloc[self.current_step]
        rsi_bin = int(min(3, max(0, row['RSI'] // 25)))
        volat_bin = 1 if row['Volatilidade'] > self.df['Volatilidade'].mean() else 0
        limiar_ia = 1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO
        ia_sinal = 1 if row['Preco_Futuro'] > row['Price'] * limiar_ia else 0
        return (rsi_bin, volat_bin, ia_sinal, int(row['Season_Num']), self.origem)

    def step(self, action):
        preco_hoje = self.df.iloc[self.current_step]['Price']
        if self.current_step + SHIFT_ALVO >= len(self.df):
            return self._get_state(), 0, True
        self.current_step += 1
        preco_novo = self.df.iloc[self.current_step]['Price']
        variacao_pct = ((preco_novo - preco_hoje) / preco_hoje) * 100 if preco_hoje > 0 else 0
        if action == 1: reward = variacao_pct - (TAXA_MARKETPLACE * 100)
        elif action == 2: reward = (variacao_pct - (TAXA_MARKETPLACE * 100)) if self.origem == 2 else variacao_pct
        else: reward = -0.05 if self.origem == 2 else 0
        done = (self.current_step >= len(self.df) - SHIFT_ALVO - 1)
        return self._get_state(), reward, done

def gerenciar_memoria(gc, nome_pl, q_tab=None, modo="carregar"):
    sh = gc.open(nome_pl)
    try: ws = sh.worksheet("Cerebro_RL")
    except: ws = sh.add_worksheet(title="Cerebro_RL", rows="2000", cols="8")
    if modo == "carregar":
        dados = ws.get_all_values()
        nova_tab = {}
        if len(dados) > 1:
            for l in dados[1:]: 
                try:
                    estado = (int(l[0]), int(l[1]), int(l[2]), int(l[3]), int(l[4]))
                    pesos = [float(l[5]), float(l[6]), float(l[7])]
                    nova_tab[estado] = pesos
                except: continue
        return nova_tab
    else:
        corpo = [["RSI", "Volat", "XGB", "Estacao", "Origem", "Q_Esperar", "Q_Comprar", "Q_Vender"]]
        for est, pes in q_tab.items(): 
            corpo.append([est[0], est[1], est[2], est[3], est[4], round(pes[0],4), round(pes[1],4), round(pes[2],4)])
        ws.clear()
        ws.update(values=corpo, range_name='A1')

def salvar_sinais_hoje(gc, nome_pl, lista_sinais):
    sh = gc.open(nome_pl)
    try: ws = sh.worksheet("Sinais_Hoje")
    except: ws = sh.add_worksheet(title="Sinais_Hoje", rows="60", cols="6")
    cabecalho = [["Item", "ROI Previsto %", "Decisão Final", "Origem", "Estação", "Atualização"]]
    ws.clear()
    ws.update(values=(cabecalho + lista_sinais), range_name='A1')

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🚀 Iniciando Gestor de Inventário RL...")
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)

mapa_origem = analisar_origem_estoque(gc, NOME_PLANILHA)

dados_brutos = gc.open(NOME_PLANILHA).worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in dados_brutos], columns=['Timestamp', 'Item', 'Price', 'Season'])
# Limpeza de preços também blindada
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)
df_bruto = df_bruto.dropna(subset=['Price'])

cache_indicadores = {}
ranking = []

for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it
        d_treino = d_it.dropna(subset=['Preco_Futuro', 'RSI', 'Lag_1'])
        if len(d_treino) < 30: continue
        X, y = d_treino[['Price', 'Lag_1', 'RSI', 'Season_Num']], d_treino['Preco_Futuro']
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, base_score=0.5).fit(X, y)
        roi = ((model.predict(d_it[['Price', 'Lag_1', 'RSI', 'Season_Num']].tail(1))[0] - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        ranking.append({'Item': item, 'ROI': roi})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False)
q_table = gerenciar_memoria(gc, NOME_PLANILHA, modo="carregar")

print("🤖 Treinando RL...")
for item in df_ranking.head(15)['Item']:
    d_it = cache_indicadores.get(item)
    if d_it is None or len(d_it) < 60: continue
    env = SFL_Environment(d_it, mapa_origem.get(item, 0))
    for ep in range(30):
        state, done = env.reset(), False
        while not done:
            action = random.randint(0, 2) if random.random() < 0.2 else np.argmax(q_table.get(state, [0, 0, 0]))
            n_state, rew, done = env.step(action)
            old_q = q_table.get(state, [0, 0, 0])[action]
            q_table.setdefault(state, [0, 0, 0])[action] = old_q + 0.1 * (rew + 0.9 * np.max(q_table.get(n_state, [0, 0, 0])) - old_q)
            state = n_state

gerenciar_memoria(gc, NOME_PLANILHA, q_table, modo="salvar")

lista_final, agora = [], datetime.now().strftime("%d/%m %H:%M")
for item in df_ranking.head(15)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    origem = mapa_origem.get(item, 0)
    ia_sinal_hoje = 1 if roi_pred > (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO - 1) * 100 else 0
    estado = (int(min(3, row['RSI'] // 25)), 1 if row['Volatilidade'] > d_it['Volatilidade'].mean() else 0, ia_sinal_hoje, int(row['Season_Num']), origem)
    acao = ["ESPERAR", "COMPRAR", "VENDER"][np.argmax(q_table.get(estado, [0, 0, 0]))]
    if acao == "COMPRAR" and row['Price'] > (row['Media_120'] + (2.0 * row['Desvio_120'])): acao = "ESPERAR (Caro/120d)"
    lista_final.append([item, round(roi_pred, 1), acao, "Colheita" if origem == 1 else ("Revenda" if origem == 2 else "Mercado"), row['Season'], agora])

salvar_sinais_hoje(gc, NOME_PLANILHA, lista_final)
print("✅ Gestão finalizada!")
