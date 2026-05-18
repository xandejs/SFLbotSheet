import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 12 
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.07  
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

MARGEM_ATENCAO = 0.035    
TRAVA_PREJUIZO = True     
LIMITE_BANCA = 0.25       
TOLERANCIA_SUJEIRA = 0.15 

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# ==============================================================================
# FUNÇÕES DE APOIO E LIMPEZA
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

    # --- JANELA MACRO (30 dias / 360 registros) ---
    d['Media_Macro'] = d['Price'].rolling(window=360, min_periods=30).mean()
    d['Desvio_Macro'] = d['Price'].rolling(window=360, min_periods=30).std()

    # --- JANELA MICRO (10 dias / 120 registros) ---
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=120, min_periods=10).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=120, min_periods=10).mean()
    d['RSI_Micro'] = (100 - (100 / (1 + (ganho / perda.replace(0, 0.0001))))).fillna(50)

    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    d['Lag_1'] = d['Price'].shift(1)
    d['Preco_Futuro'] = d['Price'].shift(-SHIFT_ALVO)
    
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(lambda x: np.exp(-0.01 * (max_date - x).days))
    
    return d

# ==============================================================================
# MOTOR DE VALIDAÇÃO E AUTOAJUSTE HÍBRIDO
# ==============================================================================

def processar_validacao_sinais(sh, df_bruto, novos_sinais_compras):
    try:
        try: ws = sh.worksheet("Validacao_Sinais")
        except: ws = sh.add_worksheet(title="Validacao_Sinais", rows="1000", cols="9")
        
        dados_antigos = ws.get_all_values()
        if not dados_antigos:
            cabecalho = [["ID_Sinal", "Data_Sinal", "Item", "Preco_Entrada", "Alvo_IA", "Maxima_12d", "Status", "Dias_Passados", "Chave_Unica"]]
            ws.append_row(cabecalho[0])
            dados_antigos = cabecalho
            
        df_val = pd.DataFrame(dados_antigos[1:], columns=dados_antigos[0])
        agora = datetime.now()
        
        # 1. Atualizar Sinais Existentes Em Análise
        if not df_val.empty:
            for idx, row_v in df_val.iterrows():
                if row_v['Status'] == '⏳ EM ANÁLISE':
                    dt_sinal = datetime.strptime(row_v['Data_Sinal'], "%d/%m/%Y %H:%M")
                    dias_decorridos = (agora - dt_sinal).days
                    df_val.at[idx, 'Dias_Passados'] = str(dias_decorridos)
                    
                    # Buscar máxima no RecLog desde a data do sinal
                    filtro_rec = df_bruto[(df_bruto['Item'] == row_v['Item']) & (df_bruto['Timestamp'] >= dt_sinal)]
                    if not filtro_rec.empty:
                        max_real = filtro_rec['Price'].max()
                        df_val.at[idx, 'Maxima_12d'] = f"{max_real:.4f}"
                        
                        # Checar se atingiu o alvo previsto
                        if max_real >= float(row_v['Alvo_IA']):
                            df_val.at[idx, 'Status'] = '🏆 ACERTOU'
                        elif dias_decorridos >= DIAS_SWING:
                            df_val.at[idx, 'Status'] = '❌ FALHOU'
        
        # 2. Inserir Novos Sinais Evitando Duplicados Próximos
        novas_linhas = []
        for n in novos_sinais_compras:
            chave = f"{n['Item']}_{datetime.now().strftime('%d%m%Y')}" 
            if df_val.empty or chave not in df_val['Chave_Unica'].values:
                id_sinal = len(df_val) + len(novas_linhas) + 1
                novas_linhas.append([
                    id_sinal, agora.strftime("%d/%m/%Y %H:%M"), n['Item'],
                    f"{n['Preco']:.4f}", f"{n['Alvo']:.4f}", f"{n['Preco']:.4f}",
                    '⏳ EM ANÁLISE', "0", chave
                ])
                
        if novas_linhas:
            df_novos = pd.DataFrame(novas_linhas, columns=dados_antigos[0])
            df_val = pd.concat([df_val, df_novos], ignore_index=True)
            
        # Manter apenas histórico dos últimos 4 meses
        if len(df_val) > 800:
            df_val = df_val.tail(500).reset_index(drop=True)
            
        # Salvar de volta na planilha
        ws.clear()
        ws.update(values=([dados_antigos[0]] + df_val.values.tolist()), range_name='A1')
        set_frozen(ws, rows=1)
        format_cell_range(ws, 'A1:I1', CellFormat(textFormat=TextFormat(bold=True)))
        
        # 3. Calcular Notas de Desempenho para o Autoajuste Híbrido
        mapa_notas = {}
        if not df_val.empty:
            for item in df_val['Item'].unique():
                df_item = df_val[(df_val['Item'] == item) & (df_val['Status'] != '⏳ EM ANÁLISE')]
                if len(df_item) >= 3: 
                    acertos = len(df_item[df_item['Status'] == '🏆 ACERTOU'])
                    taxa = (acertos / len(df_item)) * 100
                    mapa_notes = {"taxa": taxa, "qtd": len(df_item)}
                    mapa_notas[item] = mapa_notes
                    
        return mapa_notas
    except Exception as e:
        print(f"Erro no Motor de Validação: {e}"); return {}

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

print("🚀 Iniciando Bot com Motor Avançado de Autoajuste...")
escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credenciais = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
gc = gspread.authorize(credenciais)
sh = gc.open(NOME_PLANILHA)

# 1. Carregar Dados Cadastrais
inv_data = sh.worksheet("Inventario").get_all_values()
inv_dict = {row[0]: limpar_numero(row[1]) for row in inv_data[1:]}

mapa_custo_fifo = calcular_fifo_hibrido(sh, inv_dict)
trade_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()

rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)
df_bruto['Timestamp'] = pd.to_datetime(df_bruto['Timestamp'], dayfirst=True, errors='coerce')

# Primeiro Passo: Ler validação antiga para obter as regras de autoajuste antes do loop
mapa_notas_ia = processar_validacao_sinais(sh, df_bruto, [])

cache_indicadores = {}
ranking = []

# 2. Processamento de IA (XGBoost) com Janelas Duplas
for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it
        d_tr = d_it.dropna(subset=['Preco_Futuro', 'Media_Macro'])
        if len(d_tr) < 30: continue
        
        model = xgb.XGBRegressor(n_estimators=40, max_depth=3).fit(
            d_tr[['Price', 'Lag_1', 'RSI_Micro']], d_tr['Preco_Futuro'], 
            sample_weight=d_tr['Peso_Temporal']
        )
        
        pred = model.predict(d_it[['Price', 'Lag_1', 'RSI_Micro']].tail(1))[0]
        roi = ((pred - d_it.iloc[-1]['Price']) / d_it.iloc[-1]['Price']) * 100
        ranking.append({'Item': item, 'ROI': roi})
    except: continue

df_ranking = pd.DataFrame(ranking).sort_values(by='ROI', ascending=False) if ranking else pd.DataFrame(columns=['Item', 'ROI'])

# 3. Processamento de Sinais e Aplicação do Mecanismo Híbrido
lista_final = []
novos_sinais_compras = []

for item in df_ranking.head(30)['Item']:
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item']==item]['ROI'].values[0]
    saldo_item = inv_dict.get(item, 0)
    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    
    preco_atual = row['Price']
    
    # --- APLICAÇÃO DO CÉREBRO HÍBRIDO (AUTOAJUSTE E VETO) ---
    nota_item = mapa_notas_ia.get(item, {"taxa": 100.0, "qtd": 0}) 
    taxa_acerto = nota_item["taxa"]
    
    multiplicador_desvio = 2.0
    veto_compra = False
    status_ia = "Confiança Alta"
    
    if nota_item["qtd"] >= 3: 
        if taxa_acerto >= 70:
            multiplicador_desvio = 2.0
            status_ia = "Confiança Alta"
        elif 45 <= taxa_acerto < 70:
            multiplicador_desvio = 2.4 
            status_ia = "Confiança Média"
            if roi_pred < 12.0: roi_pred -= 5.0 
        else:
            veto_compra = True 
            status_ia = "Veto de Segurança"

    # Definição Dinâmica de Alvos
    alvo_in = row['Media_Macro'] - (multiplicador_desvio * row['Desvio_Macro'])
    alvo_out_estatistico = row['Media_Macro'] + (2.0 * row['Desvio_Macro'])
    
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1
    
    if origem == 2 and custo_medio > 0:
        alvo_lucro = custo_medio * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
        alvo_out = max(alvo_lucro, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico

    # Decisão Hierárquica
    acao = "ESPERAR"
    if preco_atual <= alvo_in:
        acao = "🚫 COMPRA VETADA" if veto_compra else "🟢 COMPRAR"
        if acao == "🟢 COMPRAR":
            alvo_saida_ia = preco_atual * (1 + (roi_pred/100))
            novos_sinais_compras.append({"Item": item, "Preco": preco_atual, "Alvo": alvo_saida_ia})
    elif preco_atual <= alvo_in * (1 + MARGEM_ATENCAO):
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        acao = "🔴 VENDER"
    elif preco_atual >= alvo_out * (1 - MARGEM_ATENCAO):
        acao = "🎯 PERTO DA SAÍDA"

    if ("VENDER" in acao or "SAÍDA" in acao) and origem == 2 and preco_atual < custo_medio:
        acao = "⚠️ VENDER (Stop Loss)" if roi_pred < -20 else "ESPERAR (Prejuízo)"

    # Exibição visual da nota com base na amostragem real
    nota_visual = f"{int(taxa_acerto)}% ({status_ia})" if nota_item["qtd"] >= 3 else "100% (Modo Inicial)"

    # Linha 291 higienizada abaixo (fix total do if):
    lista_final.append([
        item, round(roi_pred, 1), acao, round(alvo_in, 4), round(alvo_out, 4), 
        "Revenda" if origem == 2 else "Colheita", nota_visual, datetime.now().strftime("%d/%m %H:%M")
    ])

# 4. Atualizar os Sinais e Processar Gravação Real de Sinais
processar_validacao_sinais(sh, df_bruto, novos_sinais_compras)

try:
    ws_sinais = sh.worksheet("Sinais_Hoje")
    ws_sinais.clear()
    cabecalho = [["Item", "ROI Prev %", "Decisão", "Entrada Sniper", "Saída Alvo", "Origem", "Taxa de Acerto IA %", "Atualização"]]
    ws_sinais.update(values=(cabecalho + lista_final), range_name='A1')
    set_frozen(ws_sinais, rows=1)
    format_cell_range(ws_sinais, 'A1:H1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Planilha Sinais_Hoje atualizada com Sucesso!")
except Exception as e:
    print(f"❌ Erro ao salvar Sinais_Hoje: {e}")

print("🏁 Ciclo finalizado com sucesso!")
