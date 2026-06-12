import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
import requests
from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 24 
TAXA_MARKETPLACE = 0.1
LUCRO_REAL_DESEJADO = 0.07  # 7% de lucro real alvo
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

# Configurações de Estratégia
MARGEM_ATENCAO = 0.035    # 3.5% para sinalizar proximidade
TOLERANCIA_SUJEIRA = 0.15 # até 15% harvest no saldo → ainda conta como Revenda

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# Cabeçalhos da aba de auditoria do XGBoost — definido aqui para estar
# disponível tanto na criação da aba quanto em referências posteriores.
HEADERS_AUDIT = [
    "Timestamp", "Item", "MAE", "RMSE",
    "Acerto_Direcao_%", "N_Treino", "N_Teste", "ROI_Previsto_%"
]

# ==============================================================================
# FUNÇÕES DE APOIO E INTELIGÊNCIA
# ==============================================================================

def obter_agora_brt():
    """Retorna o datetime atual convertido para o Horário de Brasília (BRT),
    limpando o tzinfo para manter total compatibilidade com o Pandas."""
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).replace(tzinfo=None)

def enviar_telegram(acao, ativo, preco, linha):
    """Envia o alerta de operação diretamente para o canal/chat do Telegram."""
    token = "8715722660:AAGgJSl2w2RpM0kxZ9Y4MFK4lHpDZaLyQmw"
    chat_id = "873167793"
    
    # Formata o preço com 5 casas decimais e troca ponto por vírgula
    preco_formatado = f"{preco:.4f}".replace('.', ',')
    
    # Define o emoji principal com base no sinal
    if "COMPRAR" in acao.upper():
        emoji = "🟢"
    elif "STOP" in acao.upper():
        emoji = "⚠️"
    else:
        emoji = "🔴"
        
    hora_formatada = obter_agora_brt().strftime("%d/%m %H:%M:%S")
    
    mensagem = (
        f"{emoji} *ALERTA DE OPERAÇÃO* {emoji}\n\n"
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
                if row['Qtd'] <= 0: continue          # guarda antes de qualquer divisão
                usar = min(row['Qtd'], falta)
                custo_total   += usar * (row['SFL_Total'] / row['Qtd'])
                qtd_acumulada += usar
            fifo_map[item] = custo_total / qtd_acumulada if qtd_acumulada > 0 else 0.0
        return fifo_map
    except Exception as e:
        print(f"Erro FIFO: {e}"); return {}

def preparar_indicadores(df_original, nome_item):
    """
    Prepara os indicadores técnicos e divide os dados em treino/teste
    respeitando a ordem temporal (sem data leakage).

    Retorna o DataFrame completo com as colunas adicionais:
      - is_holdout: True para as últimas SHIFT_ALVO linhas (conjunto de teste)
    """
    d = df_original[df_original['Item'] == nome_item].copy()
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], dayfirst=True, errors='coerce')
    d = d.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)

    # Garante que as colunas do modelo existem mesmo se o histórico for curto
    for col in ['Media_Macro', 'Desvio_Macro', 'RSI_Micro',
                'Season_Num', 'Lag_1', 'Preco_Futuro', 'Peso_Temporal', 'is_holdout']:
        d[col] = np.nan
    d['is_holdout'] = False

    janela_macro_dinamica = 30 * REGISTROS_POR_DIA   # 720 registros
    janela_micro_dinamica = 10 * REGISTROS_POR_DIA   # 240 registros

    if len(d) < janela_micro_dinamica:
        return d

    # --- JANELA MACRO (30 dias) ---
    # min_periods=240 exige 10 dias completos antes de calcular a banda.
    # Antes era 120 (5 dias), o que gerava bandas muito ruidosas no início.
    d['Media_Macro'] = d['Price'].rolling(
        window=janela_macro_dinamica, min_periods=120
    ).mean()
    d['Desvio_Macro'] = d['Price'].rolling(
        window=janela_macro_dinamica, min_periods=120
    ).std()

    # --- JANELA MICRO (10 dias) para RSI ---
    delta = d['Price'].diff()
    ganho = delta.clip(lower=0).rolling(window=janela_micro_dinamica, min_periods=48).mean()
    perda = -1 * delta.clip(upper=0).rolling(window=janela_micro_dinamica, min_periods=48).mean()

    # Quando perda == 0 (mercado só subiu), RSI real é 100, não 50.
    # Marcamos explicitamente em vez de mascarar com fillna(50).
    rsi_raw = 100 - (100 / (1 + (ganho / perda.replace(0, np.nan))))
    d['RSI_Micro'] = rsi_raw.fillna(100)   # sem dados de perda = mercado em alta total = RSI 100

    # --- Features para o XGBoost ---
    d['Season_Num'] = d['Season'].map(MAP_ESTACOES).fillna(0)
    d['Lag_1']       = d['Price'].shift(1)
    d['Preco_Futuro']= d['Price'].shift(-SHIFT_ALVO)

    # Peso temporal exponencial (dados mais recentes pesam mais)
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(
        lambda x: np.exp(-0.01 * (max_date - x).days)
    )

    # --- HOLDOUT TEMPORAL ---
    # Separamos dois conceitos que antes eram confundidos:
    #   SHIFT_ALVO (288) = janela de predição de produção → gera NaN nas últimas 288 linhas
    #   HOLDOUT_AUDIT (72) = janela de medição de erro → colocada ANTES da zona de NaN
    #
    # Layout: [Treino][Holdout=72][NaN=288]
    # Mínimo por item: 120 (warmup) + 50 (treino) + 72 (holdout) + 288 (NaN) = 530 linhas (~22 dias)
    # Itens com menos de 530 linhas: treinados normalmente, MAE fica em branco (aceitável).
    HOLDOUT_AUDIT = 72   # 3 dias — suficiente para MAE/RMSE/Acerto_Direcao confiáveis
    holdout_fim = len(d) - SHIFT_ALVO
    holdout_ini = holdout_fim - HOLDOUT_AUDIT
    if holdout_ini > 0:
        d.iloc[holdout_ini:holdout_fim, d.columns.get_loc('is_holdout')] = True

    return d

def construir_estado_rl_hibrido(row_mercado, item, origem, custo_medio, alvo_lucro, df_trade, roi_pred):
    """
    Gera a string de estado discreta de 7 caracteres para o RL:
    [RSI][Volat][XGB][Estacao][Origem][Lucro][Tempo]
    """
    # 1. RSI Discreto (3 níveis: 0=Baixo, 1=Neutro, 2=Alto)
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
    
    if origem_disc == 2:  # Se for um item mantido para Revenda
        # Cálculo do Lucro Atual Discreto (0=Prejuízo, 1=Lucro Médio, 2=Meta Batida)
        if preco_atual < custo_medio:
            lucro_disc = 0
        elif preco_atual < alvo_lucro:
            lucro_disc = 1
        else:
            lucro_disc = 2
            
        # Cálculo do Tempo de Custódia buscando no DataFrame do TradeLog
        try:
            compras_item = df_trade[(df_trade['Nome do Item'] == item) & 
                                    (df_trade['Tipo'].str.contains("Compra", case=False))]
            if not compras_item.empty:
                col_data = 'Timestamp' if 'Timestamp' in df_trade.columns else 'Data'
                # iloc[0] = compra mais antiga ainda em custódia (FIFO real).
                # iloc[-1] seria a mais recente e subestimaria o tempo de custódia.
                data_compra_str = str(compras_item.iloc[0][col_data]).split()[0]
                
                dt_compra = None
                for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
                    try:
                        dt_compra = datetime.strptime(data_compra_str, fmt)
                        break
                    except: continue
                
                if dt_compra:
                    dias_em_custodia = (obter_agora_brt() - dt_compra).days
                    if dias_em_custodia <= 3:
                        tempo_disc = 1   # Ciclo Inicial
                    elif dias_em_custodia <= 8:
                        tempo_disc = 2   # Ciclo Médio
                    else:
                        tempo_disc = 3   # Alerta de Tempo Limite (>= 9 dias preso)
                else:
                    tempo_disc = 1
            else:
                tempo_disc = 1
        except:
            tempo_disc = 1
    else:
        lucro_disc = 0
        tempo_disc = 0

    return f"{rsi_disc}{volat_disc}{xgb_disc}{estacao_disc}{origem_disc}{lucro_disc}{tempo_disc}"

def pontuar_status(r):
    """Converte o Status de um sinal concluído em pontuação para calibração.
    Separado como função de módulo para evitar redefinição dentro de blocos condicionais."""
    st = r['Status']
    if "🏆 ACERTOU" in st: return 1.0
    if "💰 LUCRO"   in st: return 0.5
    return 0.0

def construir_calibracao_hibrida(sh, df_bruto_local, df_val_mem=None):
    """
    Constrói os três mapas de calibração combinando duas fontes de verdade:

    FONTE PRIMÁRIA — taxa de acerto real (Validacao_Sinais):
        Lê do df_val_mem já carregado em memória (sem chamada extra à API).
        Calibra mult_desvio, veto e ajuste de ROI com base no histórico real.
        Latência de 12 dias — itens novos ficam sem dados desta fonte.

    FONTE SECUNDÁRIA — MAE do XGBoost (XGB_Audit):
        Para itens sem sinais concluídos, usa o erro do modelo como proxy.
        Resposta imediata, mas menos confiável que o resultado de mercado.
        Se XGB_Audit não existir ainda, ignora silenciosamente.

    Retorna: (mapa_mult_desvio, mapa_veto, mapa_ajuste_roi)
    """
    mapa_mult = {}
    mapa_veto = {}
    mapa_roi  = {}
    _confianca = {}   # metadado interno: 0.0 (só proxy) a 1.0 (só histórico real)

    # ------------------------------------------------------------------
    # FONTE 1: Taxa de acerto real — usa df_val já em memória
    # ------------------------------------------------------------------
    if df_val_mem is not None and not df_val_mem.empty:
        df_concluidos = df_val_mem[
            df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)
        ].copy()

        if not df_concluidos.empty:
            df_concluidos['Pontos'] = df_concluidos.apply(pontuar_status, axis=1)

            for item_c, group in df_concluidos.groupby('Item'):
                n    = len(group)
                taxa = (group['Pontos'].sum() / n) * 100
                confianca = min(1.0, n / 10.0)   # plena confiança com 10+ sinais

                if taxa >= 70:
                    mapa_mult[item_c] = 2.0
                    mapa_roi[item_c]  = 0.0
                    mapa_veto[item_c] = False
                elif taxa >= 45:
                    mapa_mult[item_c] = 2.4
                    mapa_roi[item_c]  = -5.0
                    mapa_veto[item_c] = False
                else:
                    mapa_mult[item_c] = 2.4
                    mapa_roi[item_c]  = -5.0
                    mapa_veto[item_c] = True

                _confianca[item_c] = confianca

            n_itens = df_concluidos['Item'].nunique()
            print(f"📊 Calibração por sinais reais: {n_itens} itens | "
                  f"{len(df_concluidos)} sinais concluídos.")
        else:
            print("ℹ️  Calibração: nenhum sinal concluído ainda — fase bootstrap.")
    else:
        print("ℹ️  Calibração: Validacao_Sinais vazia — fase bootstrap.")

    # ------------------------------------------------------------------
    # FONTE 2: MAE do XGBoost — lê XGB_Audit para enriquecer itens sem
    # histórico de sinais concluídos ainda (proxy de curto prazo)
    # ------------------------------------------------------------------
    JANELA_AUDIT = 72   # últimas 72 horas de ciclos

    try:
        ws_audit   = sh.worksheet("XGB_Audit")
        audit_data = ws_audit.get_all_values()

        if len(audit_data) > 1:
            df_audit = pd.DataFrame(audit_data[1:], columns=audit_data[0])
            df_audit['MAE']              = df_audit['MAE'].apply(limpar_numero)
            df_audit['Acerto_Direcao_%'] = df_audit['Acerto_Direcao_%'].apply(limpar_numero)
            df_audit = df_audit[df_audit['MAE'] > 0].copy()

            n_proxy = 0
            for item_a, grupo in df_audit.groupby('Item'):
                ultimos      = grupo.tail(JANELA_AUDIT)
                mae_medio    = ultimos['MAE'].mean()
                acerto_medio = ultimos['Acerto_Direcao_%'].mean()

                preco_medio  = df_bruto_local[df_bruto_local['Item'] == item_a]['Price'].tail(240).mean()
                mae_relativo = (mae_medio / preco_medio) if preco_medio > 0 else 0.15

                if item_a not in mapa_mult:
                    # Sem histórico real → MAE como único sinal de confiança
                    if mae_relativo < 0.05 and acerto_medio >= 60:
                        mapa_mult[item_a] = 2.0
                        mapa_roi[item_a]  = 0.0
                        mapa_veto[item_a] = False
                    elif mae_relativo < 0.12:
                        mapa_mult[item_a] = 2.4
                        mapa_roi[item_a]  = -3.0
                        mapa_veto[item_a] = False
                    else:
                        # Modelo impreciso → bands largas, sem veto (sem evidência real ainda)
                        mapa_mult[item_a] = 2.8
                        mapa_roi[item_a]  = -6.0
                        mapa_veto[item_a] = False
                    _confianca[item_a] = 0.0
                    n_proxy += 1
                else:
                    # Tem histórico real parcial → ajuste leve proporcional à incerteza
                    peso_xgb = 1.0 - _confianca.get(item_a, 1.0)
                    if mae_relativo > 0.12 and peso_xgb > 0:
                        mult_atual = mapa_mult[item_a]
                        mapa_mult[item_a] = round(min(mult_atual + 0.3 * peso_xgb, 3.0), 2)

            print(f"📐 Calibração por XGB_Audit: {n_proxy} itens via proxy MAE.")

    except gspread.exceptions.WorksheetNotFound:
        print("ℹ️  XGB_Audit ainda não existe — será criada neste ciclo.")
    except Exception as e:
        print(f"ℹ️  XGB_Audit inacessível: {e} — calibração por sinais reais apenas.")

    # ------------------------------------------------------------------
    # INDICADOR DE BOOTSTRAP
    # ------------------------------------------------------------------
    n_estados = len(q_table)
    if n_estados < 10:
        print(f"🌱 BOOTSTRAP ATIVO: Cerebro_RL com {n_estados} estados. "
              f"Decisões por estatística pura.")
    else:
        print(f"🧠 Cerebro_RL operacional: {n_estados} estados memorizados.")

    print(f"✅ Calibração final: {len(mapa_mult)} itens com parâmetros ajustados.")
    return mapa_mult, mapa_veto, mapa_roi

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

mapa_custo_fifo = calcular_fifo_hibrido(sh, inv_dict)
trade_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()

# Carregamento inicial do histórico de preços brutos
rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

# ==============================================================================
# MOTOR DE VALIDAÇÃO E AUTOAJUSTE HÍBRIDO + APRENDIZADO POR REFORÇO
# ==============================================================================
print("🔄 Processando Validação de Sinais e Métricas de Rigor...")
headers_val = ["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin", "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica"]

try:
    ws_val = sh.worksheet("Validacao_Sinais")
    val_data = ws_val.get_all_values()
except:
    ws_val = sh.add_worksheet(title="Validacao_Sinais", rows="1000", cols="11")
    val_data = []

if not val_data or len(val_data) == 0:
    ws_val.update(values=[headers_val], range_name='A1', value_input_option='USER_ENTERED')
    val_data = [headers_val]

if len(val_data) > 0 and "Estado_RL" not in val_data[0]:
    print("⚠️ Esquema antigo de 10 colunas detectado. Injetando coluna 'Estado_RL' para compatibilidade...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0])
    df_val.insert(9, "Estado_RL", "") 
    df_val = df_val.reindex(columns=headers_val).fillna("")
else:
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=headers_val)

mapa_mult_desvio = {}   # populados por construir_calibracao_hibrida() após validação
mapa_veto        = {}
mapa_ajuste_roi  = {}

# ======================================================================
# PERSISTÊNCIA DO CÉREBRO RL (CARREGAMENTO DO HISTÓRICO)
# ======================================================================
q_table = {}  
alpha_rl = 0.2  

try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    cerebro_data = ws_cerebro.get_all_values()
    if len(cerebro_data) > 1:
        for row in cerebro_data[1:]:
            if len(row) >= 3:
                st = str(row[0]).strip().replace("'", "")
                q_table[st] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA": limpar_numero(row[2])
                }
        print(f"🧠 Cérebro RL carregado com sucesso: {len(q_table)} estados restaurados da planilha.")
except Exception as e:
    try:
        ws_cerebro = sh.add_worksheet(title="Cerebro_RL", rows="5000", cols="3")
        ws_cerebro.update(values=[["Estado_RL", "Q_COMPRA", "Q_VENDA"]], range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL criada com estrutura de 7 caracteres.")
    except:
        print("⚠️ Aba Cerebro_RL já existente, mas vazia ou inacessível.")

# Lista para capturar APENAS os índices que mudaram de estado NESTE ciclo específico
indices_concluidos_neste_ciclo = []

if not df_val.empty:
    # Atualizar linhas pendentes usando Delta de Tempo Real em BRT
    for idx, row_val in df_val.iterrows():
        if row_val['Status'] == 'Aguardando' or row_val['Status'] == '':
            item_v = row_val['Item']
            tipo_v = row_val['Tipo']
            data_sinal_v = pd.to_datetime(row_val['Data_Sinal'], format="%d/%m/%Y %H:%M", errors='coerce')
            preco_sinal_v = limpar_numero(row_val['Preco_Sinal'])
            ref_fin_v = limpar_numero(row_val['Referencia_Fin'])
            
            if pd.notnull(data_sinal_v):
                dias_p = (obter_agora_brt() - data_sinal_v).days
            else:
                dias_p = 0  
            
            d_hist = df_bruto[df_bruto['Item'] == item_v].copy()
            d_hist['Timestamp'] = pd.to_datetime(d_hist['Timestamp'], dayfirst=True, errors='coerce')
            df_pos_sinal = d_hist[d_hist['Timestamp'] >= data_sinal_v]
            
            # Monitoramento_12d tem semântica diferente por tipo de sinal:
            #   COMPRA      → MAX: confirma se o preço subiu ao alvo de venda
            #   VENDA _STOP → MAX: necessário para validar "preço ficou abaixo do stop"
            #   VENDA comum → MIN: confirma se o preço CAIU após a venda (sell foi sábio)
            is_stop_sinal = "_STOP" in str(row_val['Chave_Unica'])
            if tipo_v == "COMPRA" or is_stop_sinal:
                preco_monitoramento = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v
            else:
                preco_monitoramento = df_pos_sinal['Price'].min() if not df_pos_sinal.empty else preco_sinal_v

            max_pos = df_pos_sinal['Price'].max() if not df_pos_sinal.empty else preco_sinal_v

            # horas_no_alvo  → COMPRA: horas com Price >= ref_fin_v (alvo de compra)
            # horas_venda     → VENDA:  horas com Price >= preco_sinal_v (nível de venda sinalizado)
            # Ambos medem se o mercado sustentou a oportunidade tempo suficiente
            # para execução manual — distinguem spike fugaz de janela real.
            horas_no_alvo = int((df_pos_sinal['Price'] >= ref_fin_v).sum())    if not df_pos_sinal.empty else 0

            # VENDA comum: conta horas em que o preço ficou <= Referencia_Fin (custo médio).
            # Se o preço CAIU abaixo do custo médio após a venda → a venda foi sábia.
            # >= 3h abaixo → 🏆 ACERTOU (TOPO) — queda confirmada, dava pra executar
            # >= 1h abaixo → 💰 LUCRO PARCIAL (SPIKE) — caiu brevemente
            #   0h abaixo  → ❌ FALHOU — preço nunca chegou abaixo da ref, venda foi prematura
            horas_venda = int((df_pos_sinal['Price'] <= ref_fin_v).sum()) if not df_pos_sinal.empty else 0

            status_v = "Aguardando"

            # --- BLOCO DE VALIDAÇÃO COM SUSTENTAÇÃO DE PREÇO ---
            # Regra geral: sinal só conclui após dias_p >= 12.
            # max_pos → registro visual no Monitoramento_12d (painel), não decide status.
            #
            # COMPRA:
            #   horas_no_alvo >= 3 → 🏆 ACERTOU        (sustentou ≥3h no alvo)
            #   horas_no_alvo >= 1 → 💰 LUCRO PARCIAL  (atingiu mas fugaz)
            #   max_pos >= piso    → 💰 LUCRO PARCIAL  (não bateu alvo mas cobriu custo)
            #   caso contrário     → ❌ FALHOU
            #
            # VENDA (não-STOP):
            #   horas_venda >= 3 → 🏆 ACERTOU (TOPO)        (preço caiu ≤ ref_fin por 3h+ → venda confirmada)
            #   horas_venda >= 1 → 💰 LUCRO PARCIAL (SPIKE)  (caiu brevemente → oportunidade existiu)
            #   horas_venda == 0 → ❌ FALHOU                  (nunca caiu abaixo da ref → venda prematura)
            #
            # STOP LOSS:
            #   Mantém lógica própria — avalia se o stop protegeu capital após 12 dias.
            if tipo_v == "COMPRA":
                if dias_p >= 12:
                    piso_lucro_real = preco_sinal_v * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
                    if horas_no_alvo >= 3:
                        status_v = "🏆 ACERTOU"
                    elif horas_no_alvo >= 1:
                        status_v = "💰 LUCRO PARCIAL"
                    elif max_pos >= piso_lucro_real:
                        status_v = "💰 LUCRO PARCIAL"
                    else:
                        status_v = "❌ FALHOU"
            elif tipo_v == "VENDA":
                if "_STOP" in str(row_val['Chave_Unica']):
                    if dias_p >= 12:
                        if max_pos <= preco_sinal_v:
                            status_v = "🏆 ACERTOU (STOP PROTETIVO)"
                        else:
                            status_v = "❌ FALHOU (STOP PREMATURO)"
                else:
                    if dias_p >= 12:
                        if horas_venda >= 3:
                            status_v = "🏆 ACERTOU (TOPO)"
                        elif horas_venda >= 1:
                            status_v = "💰 LUCRO PARCIAL (SPIKE)"
                        else:
                            status_v = "❌ FALHOU"
            
            df_val.at[idx, 'Dias_Passados'] = int(dias_p)
            df_val.at[idx, 'Monitoramento_12d'] = round(preco_monitoramento, 4)
            df_val.at[idx, 'Status'] = status_v

            # Se o status foi finalizado AGORA, guardamos para atualizar o RL sem duplicidade
            if "🏆" in status_v or "❌" in status_v or "💰" in status_v:
                indices_concluidos_neste_ciclo.append(idx)

    # Força todas as colunas a voltarem a ser números nativos do Python antes do update
    df_val['ID_Sinal'] = pd.to_numeric(df_val['ID_Sinal'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Preco_Sinal'] = df_val['Preco_Sinal'].apply(limpar_numero)
    df_val['Referencia_Fin'] = df_val['Referencia_Fin'].apply(limpar_numero)
    df_val['Monitoramento_12d'] = df_val['Monitoramento_12d'].apply(limpar_numero)
    df_val['Dias_Passados'] = pd.to_numeric(df_val['Dias_Passados'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    
    # Isolamento limpo do Estado_RL
    df_val['Estado_RL'] = df_val['Estado_RL'].astype(str).apply(lambda x: x.strip().replace("'", ""))
    df_val['Estado_RL'] = df_val['Estado_RL'].apply(lambda x: f"'{x}" if x != "" else "")

    # Salvar de volta as atualizações estruturadas
    ws_val.clear()
    ws_val.update(values=([headers_val] + df_val.values.tolist()), range_name='A1', value_input_option='USER_ENTERED')

    # ======================================================================
    # MOTOR DE APRENDIZADO POR REFORÇO — apenas sinais novos deste ciclo
    # (calibração dos mapas movida para construir_calibracao_hibrida abaixo)
    # ======================================================================
    if indices_concluidos_neste_ciclo:
        df_reforco = df_val.loc[indices_concluidos_neste_ciclo].copy()
        df_reforco = df_reforco[df_reforco['Estado_RL'].astype(str).str.strip() != ""].copy()

        for _, row_r in df_reforco.iterrows():
            st_rl     = str(row_r['Estado_RL']).strip().replace("'", "")
            acao_rl   = row_r['Tipo']
            status_rl = row_r['Status']

            if status_rl in ["🏆 ACERTOU", "🏆 ACERTOU (TOPO)", "🏆 ACERTOU (STOP PROTETIVO)"]:
                recompensa = 1.0
            elif status_rl in ["💰 LUCRO (SUBIU)", "💰 LUCRO PARCIAL", "💰 LUCRO PARCIAL (SPIKE)"]:
                recompensa = 0.5
            else:
                recompensa = -1.0

            if st_rl not in q_table:
                q_table[st_rl] = {"COMPRA": 0.0, "VENDA": 0.0}

            q_atual = q_table[st_rl].get(acao_rl, 0.0)
            q_table[st_rl][acao_rl] = q_atual + alpha_rl * (recompensa - q_atual)

        print(f"🧠 Cérebro RL Atualizado de forma incremental: {len(df_reforco)} novos estados processados.")

    print(f"🧠 Cérebro RL Atualizado internamente: {len(q_table)} estados de mercado na memória RAM.")

try:
    max_id_existente = int(pd.to_numeric(df_val['ID_Sinal'], errors='coerce').max()) if not df_val.empty else 0
except:
    max_id_existente = len(df_val)
prox_id_sinal = max_id_existente + 1
novas_linhas_validacao = []

# Mapa de último sinal por item+tipo — base para a deduplicação híbrida.
# Lido do df_val atual (já em memória, sem custo extra de API).
# Estrutura: { "Leather_COMPRA": {"preco": 0.1014, "timestamp": datetime} }
mapa_ultimo_sinal = {}
usar_dedup_simples = False   # fallback: se True, reverte para 1 sinal/item/dia
if not df_val.empty:
    try:
        for _, row_u in df_val.iterrows():
            chave_u  = f"{row_u['Item']}_{row_u['Tipo']}"
            ts_u     = pd.to_datetime(row_u['Data_Sinal'], dayfirst=True, errors='coerce')
            preco_u  = limpar_numero(row_u['Preco_Sinal'])
            if chave_u not in mapa_ultimo_sinal or ts_u > mapa_ultimo_sinal[chave_u]['timestamp']:
                mapa_ultimo_sinal[chave_u] = {'preco': preco_u, 'timestamp': ts_u}
    except Exception as e:
        print(f"⚠️  mapa_ultimo_sinal falhou: {e} — revertendo para deduplicação por data.")
        mapa_ultimo_sinal  = {}
        usar_dedup_simples = True

# Calibração híbrida: combina taxa de acerto real (Validacao_Sinais)
# + proxy de MAE do XGBoost (XGB_Audit do ciclo anterior).
# Chamada aqui, após a q_table estar atualizada e df_val pronto em memória.
mapa_mult_desvio, mapa_veto, mapa_ajuste_roi = construir_calibracao_hibrida(sh, df_bruto, df_val_mem=df_val)

# ==============================================================================
# 2. IA E TREINAMENTO
# ==============================================================================
cache_indicadores = {}
ranking = []
audit_linhas = []   # acumula métricas para gravar na aba XGB_Audit ao final
X_cols = ['Price', 'Lag_1', 'RSI_Micro', 'Media_Macro', 'Desvio_Macro', 'Season_Num']

for item in df_bruto['Item'].unique():
    try:
        d_it = preparar_indicadores(df_bruto, item)
        cache_indicadores[item] = d_it

        # Separação temporal: treino = sem holdout, teste = holdout
        d_treino = d_it[~d_it['is_holdout']].dropna(subset=['Preco_Futuro'] + X_cols)
        d_teste  = d_it[ d_it['is_holdout'] ].dropna(subset=['Preco_Futuro'] + X_cols)

        # Mínimo de 50 amostras de treino — exigir 100 bloqueava itens com menos de
        # 26 dias de histórico (100 + 288 NaN + 240 warmup = 628 linhas por item).
        # Com 50 + 288 + 120 = 458 linhas (~19 dias), a maioria dos itens ativos qualifica.
        if len(d_treino) < 50:
            continue

        # --- TREINO ---
        model = xgb.XGBRegressor(
            n_estimators=40,
            max_depth=3,
            verbosity=0           # sem output no console
        ).fit(
            d_treino[X_cols],
            d_treino['Preco_Futuro'],
            sample_weight=d_treino['Peso_Temporal']
        )

        # --- PREDIÇÃO DE PRODUÇÃO (última linha disponível) ---
        ultima_linha = d_it[X_cols].dropna().tail(1)
        if ultima_linha.empty:
            continue

        pred = model.predict(ultima_linha)[0]
        preco_atual = d_it.iloc[-1]['Price']
        roi = ((pred - preco_atual) / preco_atual) * 100
        roi_ajustado = roi + mapa_ajuste_roi.get(item, 0.0)

        # --- MÉTRICAS NO HOLDOUT (aprendizado real) ---
        mae = rmse = acerto_direcao = None
        n_treino = len(d_treino)
        n_teste  = len(d_teste)

        if n_teste >= 5:   # só calcula se houver dados suficientes no holdout
            pred_teste = model.predict(d_teste[X_cols])
            real_teste = d_teste['Preco_Futuro'].values

            erros = pred_teste - real_teste
            mae   = float(np.mean(np.abs(erros)))
            rmse  = float(np.sqrt(np.mean(erros ** 2)))

            # Acerto de direção: o modelo previu subida/queda corretamente?
            preco_base_teste = d_teste['Price'].values
            direcao_real  = (real_teste  > preco_base_teste).astype(int)
            direcao_pred  = (pred_teste  > preco_base_teste).astype(int)
            acerto_direcao = float(np.mean(direcao_real == direcao_pred) * 100)

        ranking.append({
            'Item':          item,
            'ROI':           roi,
            'ROI_Ajustado':  roi_ajustado
        })

        # Acumula linha de auditoria para gravar depois
        audit_linhas.append([
            obter_agora_brt().strftime("%d/%m/%Y %H:%M"),
            item,
            round(mae,   4) if mae   is not None else "",
            round(rmse,  4) if rmse  is not None else "",
            round(acerto_direcao, 1) if acerto_direcao is not None else "",
            n_treino,
            n_teste,
            round(roi, 2)
        ])

    except Exception as e:
        print(f"⚠️ XGBoost falhou para '{item}': {e}")
        continue

df_ranking = (
    pd.DataFrame(ranking).sort_values(by='ROI_Ajustado', ascending=False)
    if ranking else pd.DataFrame(columns=['Item', 'ROI', 'ROI_Ajustado'])
)


# ==============================================================================
# 3. GERAÇÃO DE SINAIS HIERÁRQUICOS
# ==============================================================================
lista_final = []

for item in df_ranking['Item']:  # todos os itens qualificados, sem corte arbitrário
    d_it = cache_indicadores.get(item)
    row = d_it.iloc[-1]
    roi_pred = df_ranking[df_ranking['Item'] == item]['ROI'].values[0]
    saldo_item = inv_dict.get(item, 0)

    # Itens com saldo zero não devem gerar sinais de VENDA nem entrar no FIFO.
    # Permitem sinal de COMPRA se o preço estiver na zona de entrada.
    # Sem esta guarda, o bot geraria VENDA de itens que você não possui.
    if saldo_item <= 0 and row['Price'] > (row['Media_Macro'] - mapa_mult_desvio.get(item, 2.0) * row['Desvio_Macro']):
        continue   # fora da zona de compra e sem estoque — nada a fazer
    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1
    
    preco_atual = row['Price']
    
    m_desvio = mapa_mult_desvio.get(item, 2.0)
    alvo_in = row['Media_Macro'] - (m_desvio * row['Desvio_Macro'])
    alvo_out_estatistico = row['Media_Macro'] + (m_desvio * row['Desvio_Macro'])
    
    if origem == 2 and custo_medio > 0:
        # Correção matemática para embutir a taxa sobre o preço de venda final
        alvo_lucro = (custo_medio * (1 + LUCRO_REAL_DESEJADO)) / (1 - TAXA_MARKETPLACE)
        alvo_out = max(alvo_lucro, alvo_out_estatistico)
    else:
        alvo_out = alvo_out_estatistico
        alvo_lucro = 0.0

    estado_rl = construir_estado_rl_hibrido(
        row_mercado=row, 
        item=item, 
        origem=origem, 
        custo_medio=custo_medio, 
        alvo_lucro=alvo_lucro, 
        df_trade=df_trade, 
        roi_pred=roi_pred
    )

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
        if roi_pred < -20:
            acao = "⚠️ VENDER (Stop Loss)"
        elif acao == "ESPERAR":
            acao = "ESPERAR (Prejuízo)"

    if acao == "🟢 COMPRAR" and mapa_veto.get(item, False):
        acao = "🚫 COMPRA VETADA"

    if acao == "🟢 COMPRAR" and estado_rl in q_table:
        if q_table[estado_rl]["COMPRA"] < -0.3:
            acao = "🚫 VETO DO CÉREBRO RL (Histórico de Prejuízo)"

    lista_final.append([
        item, 
        round(roi_pred, 1), 
        acao, 
        round(alvo_in, 4), 
        round(alvo_out, 4), 
        "Revenda" if origem==2 else "Colheita", 
        row['Season'], 
        f"'{estado_rl}", 
        obter_agora_brt().strftime("%d/%m %H:%M")
    ])

    if acao in ["🟢 COMPRAR", "🔴 VENDER", "🔴 VENDER (Lucro Disponível)", "⚠️ VENDER (Stop Loss)"]:
        agora_brt   = obter_agora_brt()
        dt_hoje_str = agora_brt.strftime("%d%m%Y")
        dt_hhmm     = agora_brt.strftime("%H%M")
        tipo_sinal  = "COMPRA" if "COMPRAR" in acao else "VENDA"
        sufixo      = "_STOP" if "Stop Loss" in acao else ""

        # --- DEDUPLICAÇÃO HÍBRIDA: mínimo 4h + variação ≥ 3% ---
        # Condição 1 — tempo mínimo: evita sinais em sequência durante spike de 1h
        # Condição 2 — variação de preço: só registra quando mercado mudou de verdade
        # STOP sempre passa independente de variação (proteção de capital não espera)
        # Resultado: itens estáveis → poucos sinais; itens voláteis → 2-4/dia
        INTERVALO_MIN_H  = 4      # horas mínimas entre sinais do mesmo item+tipo
        VARIACAO_MIN_PCT = 0.03   # 3% de variação mínima de preço

        chave_item_tipo = f"{item}_{tipo_sinal}"
        ultimo          = mapa_ultimo_sinal.get(chave_item_tipo)
        pode_gerar      = True

        if ultimo and not sufixo:
            horas_desde = (agora_brt - ultimo['timestamp']).total_seconds() / 3600
            variacao    = abs(preco_atual - ultimo['preco']) / ultimo['preco'] if ultimo['preco'] > 0 else 1.0
            if horas_desde < INTERVALO_MIN_H or variacao < VARIACAO_MIN_PCT:
                pode_gerar = False

        # HHmm na chave distingue múltiplos sinais válidos no mesmo dia
        chave_unica = f"{item}_{tipo_sinal}_{dt_hoje_str}_{dt_hhmm}{sufixo}"

        if pode_gerar and chave_unica not in df_val.get('Chave_Unica', pd.Series([], dtype=str)).values:
            if not any(x[10] == chave_unica for x in novas_linhas_validacao):

                linha_sinais_hoje = len(lista_final) + 1
                enviar_telegram(acao, item, preco_atual, linha_sinais_hoje)

                ref_financeira = alvo_out if tipo_sinal == "COMPRA" else custo_medio

                novas_linhas_validacao.append([
                    int(prox_id_sinal),
                    tipo_sinal,
                    agora_brt.strftime("%d/%m/%Y %H:%M"),
                    item,
                    round(preco_atual, 4),
                    round(ref_financeira, 4),
                    round(preco_atual, 4),
                    "Aguardando",
                    0,
                    f"'{estado_rl}",
                    chave_unica
                ])
                prox_id_sinal += 1

                # Atualiza mapa em memória para deduplicar dentro do mesmo ciclo
                mapa_ultimo_sinal[chave_item_tipo] = {
                    'preco':     preco_atual,
                    'timestamp': agora_brt
                }

if novas_linhas_validacao:
    ws_val.append_rows(novas_linhas_validacao, value_input_option='USER_ENTERED')
    print(f"📌 {len(novas_linhas_validacao)} novos sinais foram integrados para monitorização.")

# ==============================================================================
# 4. SALVAMENTO DA PLANILHA DE SELEÇÃO DIÁRIA (Sinais_Hoje)
# ==============================================================================
HEADERS_HOJE = [
    "Item", "ROI_Prev_%", "Ação",
    "Alvo_Entrada", "Alvo_Saída",
    "Origem", "Estação", "Estado_RL", "Atualizado"
]

try:
    try:
        ws_hoje = sh.worksheet("Sinais_Hoje")
    except Exception:
        ws_hoje = sh.add_worksheet(title="Sinais_Hoje", rows="200", cols="9")
        set_frozen(ws_hoje, rows=1)
        format_cell_range(ws_hoje, 'A1:I1', CellFormat(textFormat=TextFormat(bold=True)))
        print("📋 Aba Sinais_Hoje criada com sucesso.")

    ws_hoje.clear()
    ws_hoje.update(
        values=([HEADERS_HOJE] + lista_final),
        range_name='A1',
        value_input_option='USER_ENTERED'
    )
    print(f"📋 Sinais_Hoje: {len(lista_final)} itens gravados.")

except Exception as e:
    print(f"❌ Erro ao salvar Sinais_Hoje: {e}")

# ==============================================================================
# 4b. GRAVAÇÃO DA AUDITORIA DO XGBOOST (XGB_Audit)
# ==============================================================================
try:
    try:
        ws_audit = sh.worksheet("XGB_Audit")
    except Exception:
        n_itens_init = df_bruto['Item'].nunique()
        rows_init    = n_itens_init * 72 * 2 + 500   # teto inicial + folga
        ws_audit = sh.add_worksheet(title="XGB_Audit", rows=str(rows_init), cols="8")
        ws_audit.update(
            values=[HEADERS_AUDIT],
            range_name='A1',
            value_input_option='USER_ENTERED'
        )
        set_frozen(ws_audit, rows=1)
        format_cell_range(
            ws_audit, 'A1:H1',
            CellFormat(textFormat=TextFormat(bold=True))
        )
        print(f"📊 Aba XGB_Audit criada com {rows_init} linhas ({n_itens_init} itens detectados).")

    # Rotação automática com teto dinâmico.
    # Recalcula a cada ciclo com base nos itens únicos reais do RecLog —
    # se novos itens forem adicionados ao jogo, o limite se ajusta sozinho.
    JANELA_AUDIT_H  = 72                              # horas de histórico útil
    n_itens_reais   = df_bruto['Item'].nunique()      # cresce automaticamente
    RETENCAO_AUDIT  = n_itens_reais * JANELA_AUDIT_H * 2  # 2× margem de segurança

    if audit_linhas:
        dados_atuais = ws_audit.get_all_values()
        header = dados_atuais[0] if dados_atuais else HEADERS_AUDIT
        linhas_existentes = dados_atuais[1:] if len(dados_atuais) > 1 else []

        todas = linhas_existentes + audit_linhas
        if len(todas) > RETENCAO_AUDIT:
            todas = todas[-RETENCAO_AUDIT:]

        ws_audit.clear()
        ws_audit.update(
            values=([header] + todas),
            range_name='A1',
            value_input_option='USER_ENTERED'
        )
        print(f"📊 XGB_Audit: {len(audit_linhas)} novos | {len(todas)} retidos | "
              f"teto={RETENCAO_AUDIT} ({n_itens_reais} itens × {JANELA_AUDIT_H*2}h).")
    else:
        print("📊 XGB_Audit: nenhum item com dados suficientes para auditoria.")

except Exception as e:
    print(f"❌ Erro ao salvar XGB_Audit: {e}")

# ==============================================================================
# PERSISTÊNCIA DO CÉREBRO RL (GRAVAÇÃO E SINCRONIZAÇÃO DA Q-TABLE)
# ==============================================================================
try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    ws_cerebro.clear()
    
    cerebro_linhas = [["Estado_RL", "Q_COMPRA", "Q_VENDA"]]
    for st, acoes in q_table.items():
        cerebro_linhas.append([
            f"'{st}", 
            round(acoes.get("COMPRA", 0.0), 4), 
            round(acoes.get("VENDA", 0.0), 4)
        ])
        
    ws_cerebro.update(values=cerebro_linhas, range_name='A1', value_input_option='USER_ENTERED')
    set_frozen(ws_cerebro, rows=1)
    format_cell_range(ws_cerebro, 'A1:C1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Memória de Longo Prazo (Cerebro_RL) sincronizada!")
except Exception as e:
    print(f"❌ Erro ao salvar a Q-Table em Cerebro_RL: {e}")

print("🏁 Ciclo finalizado com sucesso!")
