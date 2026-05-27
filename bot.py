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
    
    # Peso Temporal
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))
    
    return d

def construir_estado_rl_hibrido(row_mercado, item, origem, custo_medio, alvo_lucro, df_trade, roi_pred):
    """
    Gera a string de estado discreta de 7 caracteres para o RL:
    [RSI][Volat][XGB][Estacao][Origem][Lucro][Tempo]
    """
    # 1. RSI Discreto (0=Baixo, 1=Neutro, 2=Alto)
    rsi_val = row_mercado.get('RSI_Micro', 50)
    rsi_disc = 0 if rsi_val < 35 else (2 if rsi_val > 65 else 1)
    
    # 2. Volatilidade Discreta (0=Baixa, 1=Alta)
    media = row_mercado.get('Media_Macro', 1)
    desvio = row_mercado.get('Desvio_Macro', 0)
    proporcao_volat = desvio / media if media > 0 else 0
    volat_disc = 1 if proporcao_volat > 0.10 else 0
    
    # 3. Previsão do XGBoost Discreta (0=Queda, 1=Lateral, 2=Alta)
    xgb_disc = 0 if roi_pred < -4 else (2 if roi_pred > 6 else 1)
    
    # 4. Estação do Jogo (0 a 3)
    estacao_disc = int(row_mercado.get('Season_Num', 0))
    
    # 5. Origem da Custódia (1=Colheita, 2=Revenda)
    origem_disc = int(origem) 

    # 6 e 7. Variáveis Dinâmicas do Portfólio (Lucro e Tempo)
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
            
        try:
            compras_item = df_trade[(df_trade['Nome do Item'] == item) & 
                                    (df_trade['Tipo'].str.contains("Compra", case=False))]
            if not compras_item.empty:
                col_data = 'Timestamp' if 'Timestamp' in df_trade.columns else 'Data'
                data_compra_str = str(compras_item.iloc[-1][col_data]).split()[0]
                
                dt_compra = None
                for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
                    try:
                        dt_compra = datetime.strptime(data_compra_str, fmt)
                        break
                    except: continue
                
                if dt_compra:
                    dias_em_custodia = (datetime.now() - dt_compra).days
                    if dias_em_custodia <= 3:
                        tempo_disc = 1   
                    elif dias_em_custodia <= 8:
                        tempo_disc = 2   
                    else:
                        tempo_disc = 3   
                else: tempo_disc = 1
            else: tempo_disc = 1
        except: tempo_disc = 1
    else:
        lucro_disc = 0
        tempo_disc = 0

    return f"{rsi_disc}{volat_disc}{xgb_disc}{estacao_disc}{origem_disc}{lucro_disc}{tempo_disc}"

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

# Carregamento inicial do histórico de preços brutos
rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

# ==============================================================================
# MOTOR DE VALIDAÇÃO E AUTOAJUSTE HÍBRIDO (CORRIGIDO COM PONTE RL)
# ==============================================================================
print("🔄 Processando Validação de Sinais e Métricas de Rigor...")
# Inclusão da coluna Estado_RL no cabeçalho de validação para permitir o treino offline posterior
headers_val = ["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin", "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica"]

try:
    ws_val = sh.worksheet("Validacao_Sinais")
    val_data = ws_val.get_all_values()
except:
    ws_val = sh.add_worksheet(title="Validacao_Sinais", rows="1000", cols="11")
    val_data = []

if len(val_data) > 0 and not any(str(c).strip() for c in val_data[0]):
    val_data = []

if not val_data or len(val_data) == 0:
    ws_val.update(values=[headers_val], range_name='A1')
    val_data = [headers_val]

if len(val_data) > 1:
    headers_reais = [str(c).strip() for c in val_data[0]]
    
    # Proteção para injetar colunas ausentes caso a tabela física esteja desatualizada
    if 'Status' not in headers_reais: headers_reais.append('Status')
    if 'Estado_RL' not in headers_reais: headers_reais.append('Estado_RL')
        
    df_val = pd.DataFrame(val_data[1:], columns=headers_val if len(headers_reais) != len(headers_val) else headers_reais)
else:
    df_val = pd.DataFrame(columns=headers_val)

mapa_mult_desvio = {}
mapa_veto = {}
mapa_ajuste_roi = {}

if not df_val.empty:
    for idx, row_val in df_val.iterrows():
        if row_val['Status'] == 'Aguardando' or row_val['Status'] == '':
            item_v = row_val['Item']
            tipo_v = row_val['Tipo']
            data_sinal_v = pd.to_datetime(row_val['Data_Sinal'], format="%d/%m/%Y %H:%M", errors='coerce')
            preco_sinal_v = limpar_numero(row_val['Preco_Sinal'])
            ref_fin_v = limpar_numero(row_val['Referencia_Fin'])
            dias_p = int(limpar_numero(row_val['Dias_Passados'])) + 1
            
            d_hist = df_bruto[df_bruto['Item'] == item_v].copy()
            d_hist['Timestamp'] = pd.to_datetime(d_hist['Timestamp'], dayfirst=True, errors='coerce')
            df_pos_sinal = d_hist[d_hist['Timestamp'] >= data_sinal_v]
            
            max_pos = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            status_v = "Aguardando"
            
            if dias_p >= 12:
                if tipo_v == "COMPRA":
                    status_v = "🏆 ACERTOU" if max_pos >= ref_fin_v else "❌ FALHOU"
                elif tipo_v == "VENDA":
                    if "_STOP" in str(row_val['Chave_Unica']):
                        status_v = "❌ FALHOU (STOP LOSS)"
                    else:
                        status_v = "💰 LUCRO (SUBIU)" if max_pos > preco_sinal_v else "🏆 ACERTOU (TOPO)"
            
            df_val.at[idx, 'Dias_Passados'] = str(dias_p)
            df_val.at[idx, 'Monitoramento_12d'] = str(round(max_pos, 4))
            df_val.at[idx, 'Status'] = status_v

    # Alinha estrutura e salva de volta
    df_val = df_val.reindex(columns=headers_val).fillna("0000100")
    ws_val.clear()
    ws_val.update(values=([headers_val] + df_val.values.tolist()), range_name='A1')
    
    df_concluidos = df_val[df_val['Status'].str.contains("🏆|❌|💰", na=False)].copy()
    if not df_concluidos.empty:
        def pontuar_status(r):
            st = r['Status']
            if st in ["🏆 ACERTOU", "🏆 ACERTOU (TOPO)"]: return 1.0
            if st == "💰 LUCRO (SUBIU)": return 0.5
            return 0.0
            
        df_concluidos['Pontos'] = df_concluidos.apply(pontuar_status, axis=1)
        for item_c, group in df_concluidos.groupby('Item'):
            taxa = (group['Pontos'].sum() / len(group)) * 100
            if taxa >= 70:
                mapa_mult_desvio[item_c] = 2.0; mapa_ajuste_roi[item_c] = 0.0; mapa_veto[item_c] = False
            elif taxa >= 45:
                mapa_mult_desvio[item_c] = 2.4; mapa_ajuste_roi[item_c] = -5.0; mapa_veto[item_c] = False
            else:
                mapa_mult_desvio[item_c] = 2.4; mapa_ajuste_roi[item_c] = -5.0; mapa_veto[item_c] = True

prox_id_sinal = len(df_val) + 1
novas_linhas_validacao = []

# ==============================================================================
# 2. IA E TREINAMENTO
# ==============================================================================
cache_indicadores = {}
ranking = []
X_cols = ['Price', 'Lag_1', 'RSI_Micro', 'Media_Macro', 'Desvio_Macro', 'Season_Num']

for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it
        d_tr = d_it.dropna(subset=['Preco_Futuro'] + X_cols)
        if len(d_tr) < 30: continue
        
        model = xgb.XGBRegressor(n_estimators=40, max_depth=3).fit(
            d_tr[X_cols], d_tr['Preco_Futuro'], sample_weight=d_tr['Peso_Temporal']
        )
        
        pred = model.predict(d_it[X_cols].tail(1))[0]
        roi = ((pred - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        roi_ajustado = roi + mapa_ajuste_roi.get(item, 0.0)
        ranking.append({'Item': item, 'ROI': roi, 'ROI_Ajustado': roi_ajustado})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI_Ajustado', ascending=False) if ranking else pd.DataFrame(columns=['Item', 'ROI', 'ROI_Ajustado'])

# ==============================================================================
# 3. GERAÇÃO DE SINAIS HIERÁRQUICOS COM INTEGRALIZAÇÃO DE ESTADO
# ==============================================================================
lista_final = []

for item in df_ranking.head(30)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item'] == item]['ROI'].values[0]
    saldo_item = inv_dict.get(item, 0)
    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1
    
    preco_atual = row['Price']
    m_desvio = mapa_mult_desvio.get(item, 2.0)
    alvo_in = row['Media_Macro'] - (m_desvio * row['Desvio_Macro'])
    alvo_out_estatistico = row['Media_Macro'] + (m_desvio * row['Desvio_Macro'])
    
    alvo_lucro = 0.0
    if origem == 2 and custo_medio > 0:
        alvo_lucro = custo_medio * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
        alvo_out = max(alvo_lucro, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico

    # --- CÁLCULO ATIVO DO ESTADO RL ---
    estado_rl = construir_estado_rl_hibrido(row, item, origem, custo_medio, alvo_lucro, df_trade, roi_pred)

    # Decisão Hierárquica Estrita
    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🟢 COMPRAR"
    elif preco_atual <= alvo_in * (1 + MARGEM_ATENCAO):
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        if origem == 2 and alvo_lucro > alvo_out_estatistico:
            acao = "🔴 VENDER (Lucro Disponível)"
        else:
            acao = "🔴 VENDER"
    elif preco_atual >= alvo_out * (1 - MARGEM_ATENCAO):
        acao = "🎯 PERTO DA SAÍDA"

    if origem == 2 and preco_atual < custo_medio:
        if roi_pred < -20: acao = "⚠️ VENDER (Stop Loss)"
        elif acao == "ESPERAR": acao = "ESPERAR (Prejuízo)"

    if acao == "🟢 COMPRAR" and mapa_veto.get(item, False):
        acao = "🚫 COMPRA VETADA"

    # Salva os 9 parâmetros expandidos na Sinais_Hoje (incluindo o estado_rl)
    lista_final.append([
        item, round(roi_pred, 1), acao, round(alvo_in, 4), round(alvo_out, 4),
        "Revenda" if origem==2 else "Colheita", row['Season'], estado_rl, datetime.now().strftime("%d/%m %H:%M")
    ])

    # --- PONTE DE APRENDIZADO: ASSOCIA ESTADO ATUAL AO SINAL DE VALIDAÇÃO ---
    if acao in ["🟢 COMPRAR", "🔴 VENDER", "🔴 VENDER (Lucro Disponível)", "⚠️ VENDER (Stop Loss)"]:
        dt_hoje_str = datetime.now().strftime("%d%m%Y")
        tipo_sinal = "COMPRA" if "COMPRAR" in acao else "VENDA"
        chave_unica = f"{item}_{tipo_sinal}_{dt_hoje_str}"
        
        if df_val.empty or (chave_unica not in df_val['Chave_Unica'].values):
            if not any(x[10] == chave_unica for x in novas_linhas_validacao):
                ref_financeira = (preco_atual * (1 + roi_pred/100)) if tipo_sinal == "COMPRA" else custo_medio
                if "Stop Loss" in acao: chave_unica += "_STOP"
                    
                novas_linhas_validacao.append([
                    prox_id_sinal, tipo_sinal, datetime.now().strftime("%d/%m/%Y %H:%M"),
                    item, round(preco_atual, 4), round(ref_financeira, 4), round(preco_atual, 4),
                    "Aguardando", 0, estado_rl, chave_unica  # <--- CHAVE SALVA AQUI DENTRO!
                ])
                prox_id_sinal += 1

if novas_linhas_validacao:
    ws_val.append_rows(novas_linhas_validacao)
    print(f"📌 {len(novas_linhas_validacao)} novos sinais acoplados ao Estado RL para monitorização.")

# ==============================================================================
# 4. SALVAMENTO DA PLANILHA DE SELEÇÃO DIÁRIA
# ==============================================================================
try:
    ws_sinais = sh.worksheet("Sinais_Hoje")
    ws_sinais.clear()
    # Cabeçalho atualizado para refletir a nova coluna 'Estado_RL'
    cabecalho = [["Item", "ROI Prev %", "Decisão", "Entrada Sniper", "Saída Alvo", "Origem", "Estação", "Estado_RL", "Atualização"]]
    ws_sinais.update(values=(cabecalho + lista_final), range_name='A1')
    set_frozen(ws_sinais, rows=1)
    format_cell_range(ws_sinais, 'A1:I1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Planilha Sinais_Hoje atualizada com sucesso em 9 Colunas!")
except Exception as e:
    print(f"❌ Erro ao salvar Sinais_Hoje: {e}")

print("🏁 Ciclo finalizado com sucesso!")
