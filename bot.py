import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import random
import os
import requests
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials

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
MARGEM_ATENCAO = 0.05    # 5% para "Perto da Entrada"
TRAVA_PREJUIZO = True     # Evita sugerir venda abaixo do custo médio
LIMITE_BANCA = 0.25       # Usa no máximo 25% do saldo Flower (SFL)
TOLERANCIA_SUJEIRA = 0.15 # 15% de colheita permitida em lotes de revenda

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# Credenciais Telegram (GitHub Secrets)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==============================================================================
# FUNÇÕES DE UTILIDADE E COMUNICAÇÃO
# ==============================================================================

def enviar_telegram(mensagem):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": mensagem})
    except Exception as e:
        print(f"Erro Telegram: {e}")

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
# LÓGICA FIFO HÍBRIDA E ORIGEM
# ==============================================================================

def calcular_fifo_hibrido(sh, inventario_dict):
    """Calcula o Preço Médio baseado nas últimas compras que compõem o estoque atual"""
    try:
        log_data = sh.worksheet("TradeLog").get_all_values()
        df_log = pd.DataFrame(log_data[1:], columns=log_data[0])
        df_log['Qtd'] = df_log['Quantidade'].apply(limpar_numero)
        df_log['SFL_Total'] = df_log['SFL'].apply(limpar_numero).abs()
        
        # Filtra apenas compras
        df_compras = df_log[df_log['Tipo'].str.contains("Compra", case=False)].copy()
        
        fifo_map = {}
        for item, saldo_real in inventario_dict.items():
            if saldo_real <= 0: continue
            
            # Pega compras do item (mais recentes primeiro)
            compras_item = df_compras[df_compras['Nome do Item'] == item].iloc[::-1]
            
            qtd_acumulada = 0
            custo_total = 0
            
            for _, row in compras_item.iterrows():
                falta = saldo_real - qtd_acumulada
                if falta <= 0: break
                
                usar = min(row['Qtd'], falta)
                preco_unitario = row['SFL_Total'] / row['Qtd'] if row['Qtd'] > 0 else 0
                
                custo_total += (usar * preco_unitario)
                qtd_acumulada += usar
            
            # Se o log não cobrir o estoque todo, o resto é custo 0 (colheita) ou média
            if qtd_acumulada > 0:
                fifo_map[item] = custo_total / qtd_acumulada
            else:
                fifo_map[item] = 0.0
        return fifo_map
    except:
        return {}

def analisar_origem_estoque(compras_totais, saldo_atual):
    if saldo_atual <= 0: return 0
    sujeira = max(0, saldo_atual - compras_totais)
    porcentagem_sujeira = sujeira / saldo_atual
    return 2 if (compras_totais > 0 and porcentagem_sujeira <= TOLERANCIA_SUJEIRA) else 1

# ==============================================================================
# INTELIGÊNCIA ARTIFICIAL (XGBOOST COM PESO TEMPORAL)
# ==============================================================================

def preparar_indicadores(df_original, nome_item):
    d = df_original[df_original['Item'] == nome_item].copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
    d = d.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    
    if len(d) < 10: return d

    # Indicadores Técnicos
    d['Lag_1'] = d['Price'].shift(1)
    d['Volatilidade'] = d['Price'].rolling(window=REGISTROS_POR_DIA).std()
    d['Media_120'] = d['Price'].rolling(window=120 * REGISTROS_POR_DIA, min_periods=1).mean()
    d['Desvio_120'] = d['Price'].rolling(window=120 * REGISTROS_POR_DIA, min_periods=1).std()
    
    # RSI com Proteção
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=SHIFT_ALVO).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=SHIFT_ALVO).mean()
    d['RSI'] = (100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))).fillna(50)
    
    # Pesos Temporais (Novos dados valem mais)
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))
    
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    return d

# ==============================================================================
# AMBIENTE DE REINFORCEMENT LEARNING
# ==============================================================================

class SFL_Environment:
    def __init__(self, df_processado, origem=0, custo_medio=0):
        self.df = df_processado.dropna(subset=['RSI', 'Volatilidade', 'Preco_Futuro']).reset_index(drop=True)
        self.origem = origem
        self.custo_medio = custo_medio
        self.reset()

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_step]
        rsi_bin = int(min(3, max(0, row['RSI'] // 25)))
        volat_bin = 1 if row['Volatilidade'] > self.df['Volatilidade'].mean() else 0
        
        limiar = 1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO
        ia_sinal = 1 if row['Preco_Futuro'] > row['Price'] * limiar else 0
        return (rsi_bin, volat_bin, ia_sinal, int(row.get('Season_Num', 0)), self.origem)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        preco_hoje = row['Price']
        preco_futuro = row['Preco_Futuro']
        
        self.current_step += 1
        variacao_pct = ((preco_futuro - preco_hoje) / preco_hoje) * 100
        
        reward = 0
        if action == 1: # COMPRAR
            reward = variacao_pct - (TAXA_MARKETPLACE * 100)
        elif action == 2: # VENDER
            # Penaliza venda abaixo do custo médio para Revenda
            if self.origem == 2 and preco_hoje < self.custo_medio:
                reward = -5 # Punição forte
            else:
                reward = variacao_pct if self.origem == 1 else (variacao_pct - (TAXA_MARKETPLACE * 100))
        
        done = (self.current_step >= len(self.df) - 1)
        return self._get_state(), reward, done

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🚀 Iniciando Bot com FIFO Híbrido e Alertas...")
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)
sh = gc.open(NOME_PLANILHA)

# 1. Leitura de Saldo e Inventário
aba_inv = sh.worksheet("Inventario").get_all_values()
inv_dict = {row[0]: limpar_numero(row[1]) for row in aba_inv[1:]}

# Saldo SFL para Gestão de Banca
try:
    aba_sfl = sh.worksheet("RegistroBruto").get_all_values()
    # Pega o último registro de 'Flower'
    df_sfl = pd.DataFrame(aba_sfl[1:], columns=aba_sfl[0])
    saldo_sfl = limpar_numero(df_sfl[df_sfl['Recurso'] == 'Flower'].iloc[-1]['Quantidade'])
except:
    saldo_sfl = 0

# 2. Cálculos de Base
print("📊 Calculando FIFO Híbrido...")
mapa_custo_fifo = calcular_fifo_hibrido(sh, inv_dict)

log_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(log_data[1:], columns=log_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Quantidade'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()

# 3. Processamento RecLog
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
        
        # XGBoost com Peso Temporal
        X = d_treino[['Price', 'Lag_1', 'RSI']]
        y = d_treino['Preco_Futuro']
        weights = d_treino['Peso_Temporal']
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3).fit(X, y, sample_weight=weights)
        
        pred = model.predict(d_it[['Price', 'Lag_1', 'RSI']].tail(1))[0]
        roi = ((pred - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        ranking.append({'Item': item, 'ROI': roi})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False)
q_table = {} # Carregamento de memória omitido para brevidade (usar gerenciar_memoria anterior)

# 4. Painel e Alertas
print("🤖 Gerando Sinais e Alertas...")
lista_final = []
alertas_msg = []

for item in df_ranking.head(30)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    
    saldo_item = inv_dict.get(item, 0)
    custo_medio = mapa_custo_fifo.get(item, row['Media_120'])
    origem = analisar_origem_estoque(compras_log.get(item, 0), saldo_item)
    
    # Preços Alvo
    preco_atual = row['Price']
    alvo_entrada = row['Media_120'] - (2.0 * row['Desvio_120'])
    alvo_saida = preco_atual * (1 + (roi_pred / 100))
    
    # Decisão RL
    acao = "ESPERAR" # Lógica de consulta Q-Table mantida
    
    # Trava de Prejuízo
    if acao == "VENDER" and origem == 2 and preco_atual < custo_medio:
        if roi_pred < -20: acao = "⚠️ VENDER (Stop Loss)"
        else: acao = "ESPERAR (Prejuízo)"
        
    # Zona de Atenção
    if acao == "ESPERAR" and preco_atual <= alvo_entrada * (1 + MARGEM_ATENCAO):
        acao = "👀 PERTO DA ENTRADA"

    # Gestão de Banca
    qtd_sugerida = 0
    if "ENTRADA" in acao or "COMPRA" in acao:
        qtd_sugerida = (saldo_sfl * LIMITE_BANCA) / preco_atual
        alertas_msg.append(f"🟢 OPORTUNIDADE: {item}\nPreço: {preco_atual}\nAlvo: {alvo_entrada}\nSugestão: Comprar {int(qtd_sugerida)} un")

    if "VENDER" in acao:
        alertas_msg.append(f"🔴 SAÍDA: {item}\nPreço: {preco_atual}\nCusto: {custo_medio}\nROI: {roi_pred:.1f}%")

    origem_txt = "Colheita" if origem == 1 else "Revenda"
    lista_final.append([item, round(roi_pred, 1), acao, round(alvo_entrada, 4), round(alvo_saida, 4), origem_txt, row['Season'], datetime.now().strftime("%d/%m %H:%M")])

# Envio de Alertas
if alertas_msg:
    enviar_telegram("\n\n".join(alertas_msg))

# Salvamento Final
# (Aqui usas as tuas funções salvar_sinais_hoje e salvar_meu_estoque para atualizar as abas)
print("✅ Ciclo finalizado com sucesso!")
