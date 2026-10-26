import pandas as pd
import numpy as np
import xgboost as xgb
import gspread
import os
import math
import requests
from datetime import datetime, timedelta, timezone
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
REGISTROS_POR_DIA = 24 
TAXA_MARKETPLACE = 0.125
LUCRO_REAL_DESEJADO = 0.07  # 7% de lucro real alvo
DIAS_SWING = 12            
SHIFT_ALVO = DIAS_SWING * REGISTROS_POR_DIA 

# Prazo de desistência da validação — NÃO afeta DIAS_SWING/SHIFT_ALVO (o horizonte
# de previsão do XGBoost continua em 12 dias, intocado). Esta constante só governa
# até quando um sinal em Validacao_Sinais pode continuar "Aguardando" antes de ser
# forçado a concluir como FALHOU/LUCRO PARCIAL. Um item que demora mais para atingir
# o alvo não é necessariamente ruim — só precisa de um Retorno_Anualizado_% maior
# para compensar o tempo de capital parado, e isso já é penalizado naturalmente
# em calcular_alocacao_recursos() via composição (1+roi)^(365/dias_medio).
DIAS_MAX_MONITORAMENTO = 30

# Configurações de Estratégia
TOLERANCIA_SUJEIRA = 0.15 # até 15% harvest no saldo → ainda conta como Revenda

# Piso de posição mínima em SFL para tratar um item como Revenda de fato.
# Sem isso, compras incidentais (ex: Stone comprado para crafting, não para flip)
# disparam Stop Loss e alvo de lucro mesmo com posição irrisória — o histórico
# de compra existe, mas não há intenção real de trade. Ajuste ao tamanho da sua banca.
POSICAO_MINIMA_SFL = 5.0

MAP_ESTACOES = {"PRIMAVERA": 0, "VERÃO": 1, "OUTONO": 2, "INVERNO": 3}

# Cabeçalhos da aba de auditoria do XGBoost — definido aqui para estar
# disponível tanto na criação da aba quanto em referências posteriores.
HEADERS_AUDIT = [
    "Timestamp", "Item", "MAE", "RMSE",
    "Acerto_Direcao_%", "N_Treino", "N_Teste", "ROI_Previsto_%"
]

# ==============================================================================
# PARÂMETROS ADAPTATIVOS — substitui constantes estáticas por funções dinâmicas
# (integrado de parametros_adaptativos.py)
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 1. VETO RL ADAPTATIVO
# Problema original: -0.3 fixo veta após 2 falhas e levanta após 1 acerto.
# Solução: threshold endurece com mais observações (confiança cresce com dados)
#          e estados com Q muito alto recebem bônus no ROI ajustado.
# ──────────────────────────────────────────────────────────────────────────────

def calcular_veto_rl(estado_rl, q_table, n_obs_por_estado=None,
                      q_compra_item=None, n_obs_item=0):
    """
    Retorna (vetado: bool, bonus_roi: float).

    Threshold adaptativo:
      - Com poucos dados (n<5): -0.5  (precisa de evidência forte para vetar)
      - Com muitos dados (n>20): -0.45 (confiante, veta com evidência consolidada)

    Bônus de exploração:
      - Q_COMPRA > +0.5: adiciona +5% ao ROI ajustado (prioriza estado excelente)
      - Q_COMPRA > +0.7: adiciona +10%

    Q-table híbrida (global + específica por item):
      O Q-value do Estado_RL é compartilhado entre TODOS os itens que caem
      nesse mesmo estado discreto (ex: Leather e Kale ambos em "1023100").
      Isso acelera o aprendizado (mais dados por estado), mas mistura itens
      com perfis de risco muito diferentes.

      q_compra_item/n_obs_item (opcionais): quando fornecidos, o Q-value final
      é uma mistura entre o global e o específico do item, com peso crescendo
      conforme o item acumula seu próprio histórico:
          peso_item = min(1.0, n_obs_item / 10.0)
          q_final   = peso_item * q_item + (1 - peso_item) * q_global
      Um item novo (peso≈0) usa quase 100% do conhecimento global — generaliza
      rápido. Um item com 10+ observações próprias (peso=1.0) usa seu histórico
      específico, sem contaminação de outros itens no mesmo estado.
    """
    if estado_rl not in q_table:
        q_compra_global = 0.0
    else:
        q_compra_global = q_table[estado_rl].get("COMPRA", 0.0)

    if q_compra_item is not None:
        peso_item = min(1.0, n_obs_item / 10.0)
        q_compra  = peso_item * q_compra_item + (1 - peso_item) * q_compra_global
    else:
        q_compra = q_compra_global

    if estado_rl not in q_table and q_compra_item is None:
        return False, 0.0

    n_obs = (n_obs_por_estado or {}).get(estado_rl, 0)

    # Threshold endurece logaritmicamente com nº de observações
    if n_obs < 5:
        threshold = -0.5     # poucos dados → exige evidência forte
    elif n_obs < 20:
        # Interpolação linear contínua entre o limiar de n_obs=5 (-0.50, igual
        # ao ramo anterior) e o limiar de n_obs=20 (-0.45, igual ao próximo
        # ramo). Antes havia um salto descontínuo de -0.50 para -0.30 bem em
        # n_obs=5, fazendo o veto ficar subitamente muito mais fácil de
        # disparar assim que a 5ª observação chegava.
        threshold = -0.5 + (0.05 / 15) * (n_obs - 5)
    else:
        threshold = -0.45    # teto: nunca mais rígido que -0.45

    vetado    = q_compra < threshold
    bonus_roi = 0.0
    if q_compra > 0.7:
        bonus_roi = 10.0
    elif q_compra > 0.5:
        bonus_roi = 5.0

    return vetado, bonus_roi


def contar_obs_por_estado(df_val):
    """
    Conta quantas observações concluídas existem por estado RL.
    Usado para calibrar o threshold adaptativo do veto.
    """
    if df_val is None or df_val.empty:
        return {}
    concluidos = df_val[df_val['Status'].str.contains("🏆|❌|💰", na=False)]
    if concluidos.empty:
        return {}
    return (
        concluidos['Estado_RL']
        .astype(str)
        .str.strip()
        .str.replace("'", "")
        .value_counts()
        .to_dict()
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. TAXA DE ACERTO COM CONFIANÇA BAYESIANA
# Problema original: 45% fixo ignora tamanho da amostra e retorno médio.
# Solução: Beta-posterior com prior neutro (1,1); taxa efetiva = E[Beta(a,b)]
#          ponderada pelo valor esperado das recompensas.
# ──────────────────────────────────────────────────────────────────────────────

def calcular_taxa_confianca(n_total, n_acertos, soma_recompensas,
                             prior_a=1.0, prior_b=1.0,
                             n_minimo=8, confianca_alvo=0.80):
    """
    Retorna (taxa_efetiva, confiante, ev_esperado).

    taxa_efetiva: média da distribuição Beta posterior — se amostra pequena,
                  puxa em direção ao prior (50%), evitando vetos prematuros.

    confiante: True se o IC superior da Beta está acima do threshold.
               Com poucos dados, o IC é largo → não veta.

    ev_esperado: valor esperado da recompensa por sinal — substitui taxa bruta
                 como critério de qualidade (captura assimetria de retornos).
    """
    if n_total == 0:
        return 0.5, False, 0.0

    a = prior_a + n_acertos
    b = prior_b + (n_total - n_acertos)

    taxa_efetiva = a / (a + b)          # média da Beta posterior
    ev_esperado  = soma_recompensas / n_total if n_total > 0 else 0.0

    # Variância da Beta: quanto menor a amostra, maior a incerteza
    variancia   = (a * b) / ((a + b)**2 * (a + b + 1))
    ic_inferior = max(0, taxa_efetiva - 1.96 * math.sqrt(variancia))
    confiante   = (n_total >= n_minimo) and (ic_inferior > 0.35)

    return taxa_efetiva, confiante, ev_esperado


def calcular_prior_empirico(df_conc, forca_prior=4.0):
    """
    Retorna (prior_a, prior_b) calibrados pela taxa de acerto GLOBAL do mercado
    (todos os itens combinados), em vez do neutro 50/50 fixo (Beta(1,1)).

    Por que isso importa: um item novo com apenas 2 sinais concluídos herdava
    um prior "chute de moeda" (50%) mesmo que o mercado como um todo tivesse,
    digamos, 65% de acerto médio — subestimando itens novos sistematicamente.
    Com o prior empírico, itens com pouco histórico herdam o comportamento
    médio real do mercado até acumularem dados próprios suficientes.

    forca_prior=4.0: equivale a "4 sinais fantasmas" do prior — fraco o
    bastante para ser dominado rapidamente por dados reais do item (a partir
    de ~8-10 sinais reais, o prior já tem influência mínima).
    """
    if df_conc is None or df_conc.empty:
        return 1.0, 1.0
    taxa_global = df_conc['Pts'].apply(lambda x: x >= 0.5).mean()
    prior_a = max(forca_prior * taxa_global, 0.5)
    prior_b = max(forca_prior * (1 - taxa_global), 0.5)
    return prior_a, prior_b


def calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto):
    """
    Recalibra os mapas usando taxa Bayesiana em vez do threshold fixo de 45%.
    Chamada dentro de construir_calibracao_hibrida, após o loop padrão.
    Modifica os mapas in-place para os itens que têm dados suficientes.

    Correção 4.23.3: removido o parâmetro 'sh' (Google Sheet), que nunca era
    usado dentro da função — ficava só como ruído na assinatura, sem afetar
    comportamento, mas era código morto.
    """
    if df_val_mem is None or df_val_mem.empty:
        return

    df_conc = df_val_mem[df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)].copy()
    if df_conc.empty:
        return

    def pontos(st):
        if "🏆" in st: return 1.0
        if "💰" in st: return 0.5
        return 0.0

    df_conc['Pts'] = df_conc['Status'].apply(pontos)

    # Prior empírico calculado uma vez, a partir da taxa de acerto global —
    # substitui o Beta(1,1) neutro no calcular_taxa_confianca para todos os itens.
    prior_a_global, prior_b_global = calcular_prior_empirico(df_conc)

    for item_c, grupo in df_conc.groupby('Item'):
        n        = len(grupo)
        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts  = grupo['Pts'].sum()

        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_pts,
            prior_a=prior_a_global, prior_b=prior_b_global
        )

        # EV esperado > 0.3 = lucrativo em média; < 0 = capital em risco
        if not confiante:
            # Amostra insuficiente: suaviza sem vetar
            mapa_mult[item_c] = 2.2
            mapa_roi[item_c]  = -2.0
            mapa_veto[item_c] = False
        elif ev >= 0.4:
            mapa_mult[item_c] = 2.0
            mapa_roi[item_c]  = 0.0
            mapa_veto[item_c] = False
        elif ev >= 0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c]  = -3.0
            mapa_veto[item_c] = False
        elif ev >= -0.1:
            mapa_mult[item_c] = 2.4
            mapa_roi[item_c]  = -5.0
            mapa_veto[item_c] = True
        else:
            # EV negativo com confiança: veto mais agressivo
            mapa_mult[item_c] = 2.8
            mapa_roi[item_c]  = -10.0
            mapa_veto[item_c] = True


# ──────────────────────────────────────────────────────────────────────────────
# 3. STOP LOSS NORMALIZADO PELO MAE
# Problema original: -20% fixo é ruído para itens voláteis (Obsidian, Honey).
# Solução: stop dispara quando ROI < -N × MAE_relativo (sinal > ruído).
# ──────────────────────────────────────────────────────────────────────────────

def calcular_stop_adaptativo(item, preco_atual, roi_pred,
                              df_audit_mem=None, n_sigma=3.0,
                              stop_minimo=-0.08, stop_maximo=-0.35):
    """
    Retorna True se o stop loss deve ser ativado.

    Lógica:
      threshold_bruto = -n_sigma × MAE_relativo
      threshold_stop  = clamp(threshold_bruto, entre stop_maximo e stop_minimo)

      O clamp nos dois lados é necessário:
        - min(stop_minimo, ...) evita que itens de baixíssima volatilidade
          fiquem com um stop "sensível demais" (ex.: MAE_relativo minúsculo
          geraria threshold perto de 0%, disparando o stop com qualquer ruído).
        - max(stop_maximo, ...) evita que itens muito voláteis fiquem com um
          stop tão frouxo que na prática nunca dispara.

    n_sigma=3: o ROI previsto precisa ser 3× o erro médio do modelo
               para ser considerado sinal real de queda (não ruído).
    """
    mae_relativo = None

    if df_audit_mem is not None and not df_audit_mem.empty:
        audit_item = df_audit_mem[df_audit_mem['Item'] == item].tail(72)
        if not audit_item.empty:
            mae_vals = audit_item['MAE'].apply(limpar_numero)
            mae_vals = mae_vals[mae_vals > 0]
            if not mae_vals.empty and preco_atual > 0:
                mae_medio    = mae_vals.mean()
                mae_relativo = mae_medio / preco_atual

    if mae_relativo is not None and mae_relativo > 0:
        threshold_bruto = -n_sigma * mae_relativo
        threshold = max(stop_maximo, min(stop_minimo, threshold_bruto))
    else:
        threshold = -0.20   # fallback: -20% original sem dados de MAE

    return roi_pred < threshold, threshold


# ──────────────────────────────────────────────────────────────────────────────
# 4. MARGEM DE ATENÇÃO RELATIVA À BANDA
# Problema original: 3.5% fixo é absurdo para Potato (abaixo do float32)
#                    e insensível para Obsidian (7% da banda).
# Solução: margem = fração fixa da largura da banda de Bollinger.
# ──────────────────────────────────────────────────────────────────────────────

def calcular_margem_atencao(alvo_in, alvo_out, fracao=0.15,
                             minimo_abs=0.0001, maximo_pct=0.10):
    """
    Retorna a margem de atenção como fração da banda.

    fracao=0.15: a zona de atenção é 15% da largura da banda Bollinger.
    Isso garante que independente da faixa de preço do item,
    a zona de atenção seja sempre proporcional à oportunidade real.

    Ex: Leather — banda = 0.038 → margem = 0.0057 (5.7% do preço)
        Potato  — banda = 0.0002 → margem = 0.00003 (ainda > minimo_abs)
        Obsidian — banda = 0.40  → margem = 0.060 (15% da banda, realista)
    """
    banda  = max(0, alvo_out - alvo_in)
    margem = banda * fracao

    # Nunca menor que o mínimo absoluto (precisão de preço)
    margem = max(margem, minimo_abs)

    # Cap: não pode ser mais que maximo_pct do alvo_in
    if alvo_in > 0:
        margem = min(margem, alvo_in * maximo_pct)

    return round(margem, 6)


# ──────────────────────────────────────────────────────────────────────────────
# 5. DETECTOR DE REGIME (crash / rally rápido)
# Problema original: SHIFT_ALVO=288 fixo não detecta crashes intraday.
# Solução: camada de detecção separada que complementa (não substitui) o swing.
# O SHIFT_ALVO continua em 288 — mudar quebraria todo o ecossistema.
# A detecção de regime serve para SUPRIMIR sinais de compra em crash ativo.
# ──────────────────────────────────────────────────────────────────────────────

def detectar_regime(d_item, janela_crash_h=24, limiar_queda=-0.12):
    """
    Retorna (em_crash: bool, variacao_24h: float).

    em_crash: True se o preço caiu mais que limiar_queda nas últimas 24h.
    Quando True, sinais de COMPRA devem ser suprimidos ou alertados.

    Isso não altera o SHIFT_ALVO — só adiciona contexto de regime ao sinal.
    """
    if d_item is None or len(d_item) < janela_crash_h:
        return False, 0.0

    preco_atual  = d_item.iloc[-1]['Price']
    preco_24h    = d_item.iloc[-janela_crash_h]['Price']

    if preco_24h == 0:
        return False, 0.0

    variacao_24h = (preco_atual - preco_24h) / preco_24h

    return variacao_24h < limiar_queda, round(variacao_24h * 100, 2)


def calcular_roi_real(tipo, preco_sinal, preco_final):
    """
    ROI real observado no momento da conclusão de um sinal, usando o preço
    FINAL efetivamente atingido — não o alvo teórico (Referencia_Fin), nem o
    máximo/mínimo histórico (Monitoramento_12d). É o mesmo cálculo usado tanto
    para a recompensa do Cerebro_RL quanto para o ranking de Alocação de
    Recursos, unificando os dois sistemas de aprendizado do bot.

    COMPRA: Preco_Final é o preço de uma venda futura ainda não executada —
            desconta a taxa de marketplace, pois ela ainda vai incidir.
    VENDA/STOP: confirma se a decisão foi validada pela queda de preço após
                a saída — a taxa já foi paga na venda real, não desconta de novo.
                Positivo = preço caiu após a saída (decisão validada).
                Negativo = preço subiu após a saída (saída foi prematura).
    """
    if preco_sinal <= 0:
        return 0.0
    if tipo == "COMPRA":
        venda_liquida = preco_final * (1 - TAXA_MARKETPLACE)
        return ((venda_liquida - preco_sinal) / preco_sinal) * 100
    return ((preco_sinal - preco_final) / preco_sinal) * 100


def calcular_alocacao_recursos(df_val_mem, dias_swing=12, min_sinais=3):
    """
    Calcula pesos de alocacao de capital por item, combinando taxa de acerto
    Bayesiana com retorno anualizado por velocidade real de capital (Dias_Ate_Alvo).

    Por que anualizado e nao ROI/dias simples: 10% em 10 dias bate 20% em 20 dias
    porque o capital gira 2x mais rapido -- isso so aparece corretamente ao compor
    (1+roi)^(365/dias) - 1, nao ao dividir roi/dias (que da a MESMA taxa diaria).

    Fonte: Validacao_Sinais (nao TradeLog) -- cobre os 63 itens sistematicamente,
    sem o vies de selecao de quais itens voce escolheu operar manualmente.

    Retorna DataFrame ordenado por peso decrescente.
    """
    colunas_saida = ["Item", "Taxa_Acerto_%", "EV_Sinal", "ROI_Medio_%",
                      "Dias_Medio", "Retorno_Anualizado_%", "Score", "Peso_Alocacao_%"]

    if df_val_mem is None or df_val_mem.empty:
        return pd.DataFrame(columns=colunas_saida)

    df_conc = df_val_mem[df_val_mem['Status'].astype(str).str.contains("🏆|❌|💰", na=False)].copy()
    if df_conc.empty:
        return pd.DataFrame(columns=colunas_saida)

    def pontos(st):
        if "🏆" in st: return 1.0
        if "💰" in st: return 0.5
        return 0.0

    df_conc['Pts'] = df_conc['Status'].astype(str).apply(pontos)
    df_conc['Preco_Sinal_N']   = df_conc['Preco_Sinal'].apply(limpar_numero)
    df_conc['Preco_Final_N']   = df_conc['Preco_Final'].apply(limpar_numero)
    df_conc['Dias_Ate_Alvo_N'] = pd.to_numeric(df_conc['Dias_Ate_Alvo'], errors='coerce')

    def roi_realizado(r):
        # Fallback para sinais antigos (schema pré-Preco_Final): usa Referencia_Fin
        # ponderado por Pts como antes. Sinais novos usam o preço REAL observado —
        # muito mais fiel, inclusive capturando o tamanho real de uma falha
        # (uma queda de 15% aparece como -15%, não como um "0" categórico).
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

    # Mesmo prior empírico usado em calibrar_por_taxa_confianca — garante que
    # os dois sistemas de calibração (mapas de risco e alocação de capital)
    # tratem itens com pouco histórico de forma consistente entre si.
    prior_a_global, prior_b_global = calcular_prior_empirico(df_conc)

    linhas = []
    for item_a, grupo in df_conc.groupby('Item'):
        n = len(grupo)
        if n < min_sinais:
            continue

        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts  = grupo['Pts'].sum()
        taxa_ef, confiante, ev = calcular_taxa_confianca(
            n, n_acertos, soma_pts,
            prior_a=prior_a_global, prior_b=prior_b_global
        )

        dias_validos = grupo['Dias_Ate_Alvo_N'].dropna()
        dias_medio   = dias_validos.mean() if not dias_validos.empty else float(dias_swing)
        # Piso de 1 dia: abaixo disso, o expoente de anualização (365/dias)
        # explode matematicamente (ex: 0.5 dia → expoente 730). Compor "se eu
        # repetisse isso 700+ vezes por ano" não é uma métrica útil — é ruído
        # de amostra pequena (sustentação em poucas horas) virando um número
        # sem sentido prático, que ainda por cima dominaria 100% do peso de
        # alocação e zeraria todos os outros itens legítimos no ranking.
        dias_medio   = max(dias_medio, 1.0)

        roi_medio_pct = grupo['ROI_Realizado'].mean()
        roi_frac      = roi_medio_pct / 100

        try:
            retorno_anualizado = ((1 + roi_frac) ** (365 / dias_medio) - 1) * 100
        except (OverflowError, ZeroDivisionError):
            retorno_anualizado = 0.0

        # Teto de sanidade: mesmo com o piso de 1 dia, um ROI alto ainda pode
        # gerar um número grande demais para ser útil na tabela. Capado em
        # 2000% (21x ao ano) — já um teto extremamente generoso, que preserva
        # a ordenação relativa sem deixar um único item dominar o ranking.
        retorno_anualizado = min(retorno_anualizado, 2000.0)

        peso_confianca = taxa_ef if confiante else taxa_ef * 0.5
        score = retorno_anualizado * peso_confianca if ev > 0 else 0.0

        linhas.append({
            "Item":                 item_a,
            "Taxa_Acerto_%":        round(taxa_ef * 100, 1),
            "EV_Sinal":             round(ev, 3),
            "ROI_Medio_%":          round(roi_medio_pct, 2),
            "Dias_Medio":           round(dias_medio, 2),
            "Retorno_Anualizado_%": round(retorno_anualizado, 1),
            "Score":                round(score, 2),
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

# ==============================================================================
# FUNÇÕES DE APOIO E INTELIGÊNCIA
# ==============================================================================

def salvar_aba_atomica(sh, titulo, header, linhas, rows_est=None, cols_est=None, formatar_header=True):
    """
    Grava dados numa aba do Google Sheets de forma atômica (parâmetro 6):
    escreve tudo numa aba temporária primeiro, e só substitui a aba original
    se a escrita for bem-sucedida.

    Por que isso importa: o padrão anterior (ws.clear() seguido de ws.update())
    tem uma janela de risco real — se o processo morrer exatamente entre as
    duas chamadas (timeout do GitHub Actions, queda de rede, erro de quota),
    a aba fica VAZIA, perdendo todo o histórico. Para Cerebro_RL isso significa
    apagar meses de aprendizado acumulado; para Validacao_Sinais, anos de
    sinais e seus resultados reais.

    Com a escrita atômica: se a escrita na aba temporária falhar, a aba
    original nunca é tocada. Só depois de confirmar que os dados novos estão
    salvos com sucesso é que a aba antiga é substituída.

    Retorna True se a operação foi bem-sucedida, False caso contrário.
    """
    titulo_temp = f"{titulo}_tmp_{int(datetime.now().timestamp())}"
    n_linhas = len(linhas) + 1
    n_cols   = len(header)
    rows_final = str(max(rows_est or 0, n_linhas + 100))
    cols_final = str(max(cols_est or 0, n_cols))

    ws_temp = None
    try:
        ws_temp = sh.add_worksheet(title=titulo_temp, rows=rows_final, cols=cols_final)
        ws_temp.update(
            values=[header] + linhas,
            range_name='A1',
            value_input_option='USER_ENTERED'
        )
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

    # Escrita temporária confirmada — agora substitui a aba original com segurança
    try:
        try:
            ws_antiga = sh.worksheet(titulo)
            sh.del_worksheet(ws_antiga)
        except Exception:
            pass   # aba original não existia ainda (primeira execução) — normal
        ws_temp.update_title(titulo)
        return True
    except Exception as e:
        print(f"⚠️ Escrita em '{titulo}' concluída, mas troca de nome falhou: {e}. "
              f"Dados estão seguros na aba temporária '{titulo_temp}' — verifique manualmente.")
        return False


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

def parse_data_sinal_robusta(texto):
    """
    Faz o parse de Data_Sinal tentando múltiplos formatos antes de desistir.

    Motivo: o formato gravado pelo próprio bot é sempre "%d/%m/%Y %H:%M", mas
    edição manual da planilha ou um schema antigo podem gravar em outro
    formato. Antes, um parse que falhasse virava NaT e era tratado como
    dias_p=0 — o sinal nunca atingia DIAS_SWING nem DIAS_MAX_MONITORAMENTO e
    ficava preso em 'Aguardando' PARA SEMPRE (silenciosamente, sem contar em
    nenhum lugar). Esta função só retorna pd.NaT se NENHUM formato conhecido
    funcionar, e quem chama trata esse caso como erro explícito, não como
    dias_p=0.
    """
    if not texto or str(texto).strip() == "":
        return pd.NaT
    texto = str(texto).strip()
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y"):
        dt = pd.to_datetime(texto, format=fmt, errors='coerce')
        if pd.notnull(dt):
            return dt
    # Último recurso: inferência automática do pandas (dayfirst=True, já que
    # o formato nativo do bot é dd/mm/aaaa)
    return pd.to_datetime(texto, dayfirst=True, errors='coerce')

def calcular_fifo_hibrido(df_trade_raw, inventario_dict):
    try:
        df_log = df_trade_raw.copy()
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
                'Season_Num', 'Lag_1', 'Lag_2', 'Lag_3', 'Hora_Dia', 'Dia_Semana',
                'Preco_Futuro', 'Peso_Temporal', 'is_holdout']:
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
    # Lag_2/Lag_3: mais contexto temporal além do preço imediatamente anterior —
    # ajuda o modelo a distinguir uma reversão de curto prazo de uma continuação
    # de tendência (Lag_1 sozinho não diferencia isso).
    d['Lag_2']       = d['Price'].shift(2)
    d['Lag_3']       = d['Price'].shift(3)
    # Hora_Dia/Dia_Semana: economias de jogo costumam ter padrão circadiano de
    # atividade de jogadores (mais oferta/demanda em certos horários e dias),
    # o que pode afetar picos de preço de forma sistemática e previsível.
    d['Hora_Dia']    = d['Timestamp'].dt.hour
    d['Dia_Semana']  = d['Timestamp'].dt.dayofweek
    d['Preco_Futuro']= d['Price'].shift(-SHIFT_ALVO)

    # Peso temporal exponencial (dados mais recentes pesam mais)
    max_date = d['Timestamp'].max()
    d['Peso_Temporal'] = d['Timestamp'].apply(
        lambda x: np.exp(-0.01 * (max_date - x).days)
    )

    # --- HOLDOUT TEMPORAL COM VALIDAÇÃO CRUZADA (múltiplas janelas) ---
    # Separamos dois conceitos que antes eram confundidos:
    #   SHIFT_ALVO (288) = janela de predição de produção → gera NaN nas últimas 288 linhas
    #   HOLDOUT_JANELA (72) = tamanho de cada janela de medição de erro
    #
    # Uma única janela de 72h pode ser "sortuda" ou "azarada" — cai bem ou mal
    # dependendo do trecho de mercado que calhou de estar ali. Para reduzir essa
    # variância, usamos 3 janelas de 72h espalhadas em pontos diferentes do
    # histórico (não apenas a mais recente), e o MAE/RMSE/Acerto_Direcao final
    # é calculado sobre a UNIÃO das 3 — uma amostra maior e mais representativa
    # de diferentes regimes de mercado, não só o que aconteceu nos últimos 3 dias.
    #
    # Layout: [Treino][Janela1=72][...][Janela2=72][...][Janela3=72][NaN=288]
    # Mínimo para 3 janelas: 120 (warmup) + 3×72 (janelas) + 50 (treino) + 288 (NaN)
    #                       ≈ 674 linhas (~28 dias). Abaixo disso, cai para uma
    # única janela (como antes, ~22 dias mínimo). Abaixo disso ainda, sem MAE.
    HOLDOUT_JANELA = 72
    N_JANELAS_HOLDOUT = 3
    WARMUP_MACRO = 120   # mesmo mínimo usado no rolling da Media_Macro

    zona_fim = len(d) - SHIFT_ALVO          # início da zona de NaN (Preco_Futuro)
    zona_disponivel = zona_fim - WARMUP_MACRO

    if zona_disponivel >= (N_JANELAS_HOLDOUT * HOLDOUT_JANELA + 50):
        tamanho_segmento = zona_disponivel // N_JANELAS_HOLDOUT
        for i in range(N_JANELAS_HOLDOUT):
            seg_fim = WARMUP_MACRO + tamanho_segmento * (i + 1)
            seg_ini = max(WARMUP_MACRO, seg_fim - HOLDOUT_JANELA)
            if seg_fim <= zona_fim:
                d.iloc[seg_ini:seg_fim, d.columns.get_loc('is_holdout')] = True
    else:
        # Fallback: dados insuficientes para 3 janelas — usa 1 única janela,
        # mantendo o comportamento original para itens com histórico mais curto.
        holdout_ini = zona_fim - HOLDOUT_JANELA
        if holdout_ini > WARMUP_MACRO:
            d.iloc[holdout_ini:zona_fim, d.columns.get_loc('is_holdout')] = True

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

def construir_calibracao_hibrida(sh, df_bruto_local, df_val_mem=None, q_table_mem=None):
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
    # df_audit_retorno: cache de XGB_Audit devolvido ao chamador para reaproveitar
    # no stop loss adaptativo mais adiante — evita uma 2ª leitura da mesma aba
    # no mesmo ciclo (economiza 1 chamada de API por execução).
    df_audit_retorno = pd.DataFrame(columns=HEADERS_AUDIT)

    # ------------------------------------------------------------------
    # FONTE 1: Taxa de acerto real — usa df_val já em memória
    # ------------------------------------------------------------------
    if df_val_mem is not None and not df_val_mem.empty:
        df_concluidos = df_val_mem[
            df_val_mem['Status'].str.contains("🏆|❌|💰", na=False)
        ].copy()

        if not df_concluidos.empty:
            df_concluidos['Pontos'] = df_concluidos.apply(pontuar_status, axis=1)

            # Calibração Bayesiana — substitui o threshold fixo 45%/70%.
            # Usa valor esperado (EV) por sinal em vez de taxa bruta,
            # e exige amostra mínima antes de vetar (evita falso negativo com poucos dados).
            calibrar_por_taxa_confianca(df_val_mem, mapa_mult, mapa_roi, mapa_veto)
            print(f"📈 Calibração Bayesiana: {df_concluidos['Item'].nunique()} itens | "
                  f"{len(df_concluidos)} sinais concluídos processados.")

            # Popula _confianca para itens da FONTE 1 — necessário para que
            # FONTE 2 (XGB_Audit proxy) calcule peso_xgb corretamente.
            # Sem isso, peso_xgb = 1 - _confianca.get(item, 1.0) = 0 para
            # todos os itens com histórico real, anulando o ajuste de MAE.
            for item_c, group in df_concluidos.groupby('Item'):
                _confianca[item_c] = min(1.0, len(group) / 10.0)
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

            # Versão completa (sem filtro de MAE>0) devolvida ao chamador —
            # é a mesma estrutura que calcular_stop_adaptativo espera, que já
            # filtra MAE>0 internamente por conta própria.
            df_audit_retorno = df_audit.copy()

            df_audit_proxy = df_audit[df_audit['MAE'] > 0].copy()

            n_proxy = 0
            for item_a, grupo in df_audit_proxy.groupby('Item'):
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
    n_estados = len(q_table_mem) if q_table_mem is not None else 0
    if n_estados < 10:
        print(f"🌱 BOOTSTRAP ATIVO: Cerebro_RL com {n_estados} estados. "
              f"Decisões por estatística pura.")
    else:
        print(f"🧠 Cerebro_RL operacional: {n_estados} estados memorizados.")

    print(f"✅ Calibração final: {len(mapa_mult)} itens com parâmetros ajustados.")
    return mapa_mult, mapa_veto, mapa_roi, df_audit_retorno

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

# TradeLog lido UMA vez só — antes era buscado 2x (aqui e dentro de calcular_fifo_hibrido),
# consumindo uma chamada de API extra por ciclo sem necessidade.
trade_data = sh.worksheet("TradeLog").get_all_values()
df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
compras_log = df_trade[df_trade['Tipo'].str.contains("Compra", case=False)].groupby('Nome do Item')['Qtd'].apply(lambda x: sum(map(limpar_numero, x))).to_dict()
mapa_custo_fifo = calcular_fifo_hibrido(df_trade, inv_dict)

# Carregamento inicial do histórico de preços brutos
rec_data = sh.worksheet("RecLog").get_all_values()[1:]
df_bruto = pd.DataFrame([l[:4] for l in rec_data], columns=['Timestamp', 'Item', 'Price', 'Season'])
df_bruto['Price'] = df_bruto['Price'].apply(limpar_numero)

# ==============================================================================
# MOTOR DE VALIDAÇÃO E AUTOAJUSTE HÍBRIDO + APRENDIZADO POR REFORÇO
# ==============================================================================
print("🔄 Processando Validação de Sinais e Métricas de Rigor...")
headers_val = ["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin", "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica", "Dias_Ate_Alvo", "Preco_Final"]

try:
    ws_val = sh.worksheet("Validacao_Sinais")
    val_data = ws_val.get_all_values()
except:
    ws_val = sh.add_worksheet(title="Validacao_Sinais", rows="10000", cols="13")
    val_data = []

if not val_data or len(val_data) == 0:
    ws_val.update(values=[headers_val], range_name='A1', value_input_option='USER_ENTERED')
    val_data = [headers_val]

if len(val_data) > 0 and "Estado_RL" not in val_data[0]:
    print("⚠️ Esquema antigo de 10 colunas detectado. Injetando coluna 'Estado_RL' para compatibilidade...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0])
    df_val.insert(9, "Estado_RL", "") 
    df_val = df_val.reindex(columns=headers_val).fillna("")
elif len(val_data) > 0 and "Dias_Ate_Alvo" not in val_data[0]:
    print("⚠️ Esquema de 11 colunas detectado. Injetando coluna 'Dias_Ate_Alvo' para compatibilidade...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0])
    df_val = df_val.reindex(columns=headers_val).fillna("")
elif len(val_data) > 0 and "Preco_Final" not in val_data[0]:
    print("⚠️ Esquema de 12 colunas detectado. Injetando coluna 'Preco_Final' para compatibilidade...")
    df_val = pd.DataFrame(val_data[1:], columns=val_data[0])
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
# alpha_rl agora é a taxa MÁXIMA (usada para estados novos). A taxa real usada
# em cada atualização decai com o número de visitas: alpha = 1/(1+n_visitas),
# com piso em 0.05 para não travar completamente e ainda reagir a mudanças
# genuínas de regime. Alpha fixo em 0.2 nunca convergia de fato — cada novo
# sinal seguia empurrando o Q-value na mesma proporção, mesmo após 50 observações.
alpha_rl = 0.2
ALPHA_MINIMO = 0.05

try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    cerebro_data = ws_cerebro.get_all_values()
    if len(cerebro_data) > 1:
        tem_contadores = len(cerebro_data[0]) >= 5   # esquema novo com N_COMPRA/N_VENDA
        for row in cerebro_data[1:]:
            if len(row) >= 3:
                st = str(row[0]).strip().replace("'", "")
                q_table[st] = {
                    "COMPRA": limpar_numero(row[1]),
                    "VENDA":  limpar_numero(row[2]),
                    # Migração segura: planilhas antigas (3 colunas) começam com
                    # contador 0 — o alpha decai a partir daqui em diante, sem
                    # perder os Q-values já aprendidos.
                    "N_COMPRA": int(limpar_numero(row[3])) if tem_contadores and len(row) > 3 else 0,
                    "N_VENDA":  int(limpar_numero(row[4])) if tem_contadores and len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL carregado com sucesso: {len(q_table)} estados restaurados da planilha.")
except Exception as e:
    try:
        ws_cerebro = sh.add_worksheet(title="Cerebro_RL", rows="5000", cols="5")
        ws_cerebro.update(values=[["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]], range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL criada com estrutura de 7 caracteres.")
    except:
        print("⚠️ Aba Cerebro_RL já existente, mas vazia ou inacessível.")

# ======================================================================
# Q-TABLE ESPECÍFICA POR ITEM (Cerebro_RL_Item) — parâmetro adaptativo 3
# Complementa (não substitui) a q_table global. Chave = "Item|Estado_RL".
# Usada em calcular_veto_rl() para misturar o conhecimento global (rápido
# de generalizar) com o específico do item (mais preciso, mas exige mais
# dados próprios). Ver docstring de calcular_veto_rl para a fórmula de blend.
# ======================================================================
q_table_item = {}

try:
    ws_cerebro_item = sh.worksheet("Cerebro_RL_Item")
    cerebro_item_data = ws_cerebro_item.get_all_values()
    if len(cerebro_item_data) > 1:
        for row in cerebro_item_data[1:]:
            if len(row) >= 3:
                chave_ci = str(row[0]).strip().replace("'", "")
                q_table_item[chave_ci] = {
                    "COMPRA":   limpar_numero(row[1]),
                    "VENDA":    limpar_numero(row[2]),
                    "N_COMPRA": int(limpar_numero(row[3])) if len(row) > 3 else 0,
                    "N_VENDA":  int(limpar_numero(row[4])) if len(row) > 4 else 0,
                }
        print(f"🧠 Cérebro RL (item-específico) carregado: {len(q_table_item)} combinações item|estado.")
except Exception:
    try:
        ws_cerebro_item = sh.add_worksheet(title="Cerebro_RL_Item", rows="20000", cols="5")
        ws_cerebro_item.update(values=[["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]], range_name='A1', value_input_option='USER_ENTERED')
        print("🧠 Nova aba Cerebro_RL_Item criada.")
    except:
        print("⚠️ Aba Cerebro_RL_Item já existente, mas vazia ou inacessível.")

# Lista para capturar APENAS os índices que mudaram de estado NESTE ciclo específico
indices_concluidos_neste_ciclo = []

if not df_val.empty:
    # Atualizar linhas pendentes usando Delta de Tempo Real em BRT
    for idx, row_val in df_val.iterrows():
        if row_val['Status'] == 'Aguardando' or row_val['Status'] == '':
            item_v = row_val['Item']
            tipo_v = row_val['Tipo']
            data_sinal_v = parse_data_sinal_robusta(row_val['Data_Sinal'])
            preco_sinal_v = limpar_numero(row_val['Preco_Sinal'])
            ref_fin_v = limpar_numero(row_val['Referencia_Fin'])

            if pd.isnull(data_sinal_v):
                # Data ilegível mesmo após tentar formatos alternativos. Antes
                # isso virava dias_p=0 e o sinal ficava em 'Aguardando' para
                # sempre (nunca atinge DIAS_SWING). Agora força um status de
                # erro explícito — sai do loop de Aguardando imediatamente,
                # sem entrar em Cerebro_RL nem em Alocação de Recursos (não
                # contém 🏆/❌/💰, então os filtros que procuram por essas
                # marcações ignoram esta linha), e fica visível na planilha
                # para verificação manual.
                df_val.at[idx, 'Dias_Passados'] = 0
                df_val.at[idx, 'Status'] = "⚠️ ERRO_DATA (Verificar Manualmente)"
                continue

            dias_p = (obter_agora_brt() - data_sinal_v).days
            
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
            # >= 3h abaixo → 🏆 ACERTOU (FUNDO) — queda confirmada, dava pra executar
            # >= 1h abaixo → 💰 LUCRO PARCIAL (SPIKE) — caiu brevemente
            #   0h abaixo  → ❌ FALHOU — preço nunca chegou abaixo da ref, venda foi prematura
            horas_venda = int((df_pos_sinal['Price'] <= ref_fin_v).sum()) if not df_pos_sinal.empty else 0

            # Dias_Ate_Alvo: dia real (dentro da janela de 12) em que a sustentação
            # de 3h foi atingida pela primeira vez. Não muda QUANDO o sinal conclui
            # (isso continua esperando 12 dias, por segurança) — só registra QUANDO
            # a oportunidade real apareceu, para permitir comparação de velocidade
            # de retorno entre itens (10% em 3 dias vs 10% em 11 dias são diferentes).
            # Para 🏆/💰: dia da 1ª sustentação. Para ❌/STOP: fica em branco (não aplicável).
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

            # --- BLOCO DE VALIDAÇÃO COM SUSTENTAÇÃO DE PREÇO E JANELA DE GRAÇA ---
            # A partir de dias_p >= DIAS_SWING (12), o sinal já PODE concluir se
            # atingiu sustentação plena (🏆 ACERTOU) — quanto mais cedo, melhor,
            # não faz sentido esperar até 30 dias para confirmar algo que já aconteceu.
            # Mas se ainda não atingiu, o sinal ganha até DIAS_MAX_MONITORAMENTO (30)
            # dias de prazo antes de ser forçado a um veredito final (LUCRO PARCIAL
            # ou FALHOU). Isso reflete que "não bateu o alvo em 12 dias" não é fracasso
            # — é só capital ainda parado, e o ranking de alocação já penaliza isso
            # via retorno anualizado sem precisar de um corte rígido aqui.
            #
            # COMPRA:
            #   horas_no_alvo >= 3 → 🏆 ACERTOU (a qualquer momento entre 12-30 dias)
            #   horas_no_alvo >= 1 ou max_pos >= piso, e dias_p >= 30 → 💰 LUCRO PARCIAL
            #   nada disso, e dias_p >= 30                            → ❌ FALHOU
            #   caso contrário (ainda dentro do prazo)                → segue Aguardando
            #
            # VENDA (não-STOP): mesma lógica espelhada com horas_venda.
            # STOP LOSS: sem capital parado após o stop — mantém veredito em 12 dias.
            if tipo_v == "COMPRA":
                if dias_p >= DIAS_SWING:
                    piso_lucro_real = preco_sinal_v * (1 + TAXA_MARKETPLACE + LUCRO_REAL_DESEJADO)
                    if horas_no_alvo >= 3:
                        status_v = "🏆 ACERTOU"
                    elif horas_no_alvo >= 1 or max_pos >= piso_lucro_real:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "💰 LUCRO PARCIAL"
                        # else: segue "Aguardando" — ainda dentro do prazo de graça
                    else:
                        if dias_p >= DIAS_MAX_MONITORAMENTO:
                            status_v = "❌ FALHOU"
                        # else: segue "Aguardando"
            elif tipo_v == "VENDA":
                if "_STOP" in str(row_val['Chave_Unica']):
                    if dias_p >= DIAS_SWING:
                        # ref_fin_v == preco_sinal_v para STOP (gravados iguais).
                        # max_pos <= ref_fin_v: preço nunca voltou acima do stop → protetivo.
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
                            # else: segue "Aguardando"
                        else:
                            if dias_p >= DIAS_MAX_MONITORAMENTO:
                                status_v = "❌ FALHOU"
                            # else: segue "Aguardando"
            
            df_val.at[idx, 'Dias_Passados'] = int(dias_p)
            df_val.at[idx, 'Monitoramento_12d'] = round(preco_monitoramento, 4)
            df_val.at[idx, 'Status'] = status_v

            # Fallback de Dias_Ate_Alvo quando o sinal concluiu sem nunca ter
            # atingido sustentação plena (❌ FALHOU, ou 💰 LUCRO PARCIAL decidido
            # apenas por ter batido o prazo de graça). Usa dias_p real — o tempo
            # verdadeiro que o capital ficou parado — em vez de uma constante fixa,
            # já que agora a conclusão pode ocorrer em qualquer dia entre 12 e 30.
            if dias_ate_alvo is None and status_v not in ("Aguardando",):
                dias_ate_alvo = float(dias_p)
            df_val.at[idx, 'Dias_Ate_Alvo'] = dias_ate_alvo if dias_ate_alvo is not None else ""

            # Preco_Final: preço REAL observado no momento da conclusão (última
            # linha do histórico filtrado), não o alvo teórico nem o máximo/mínimo.
            # Base para o ROI unificado usado tanto no Cerebro_RL quanto na
            # Alocação de Recursos — reflete o que aconteceu de verdade, inclusive
            # o tamanho real de uma falha (não apenas "0" categórico).
            if status_v not in ("Aguardando",) and not df_pos_sinal.empty:
                df_val.at[idx, 'Preco_Final'] = round(df_pos_sinal['Price'].iloc[-1], 4)
            elif status_v not in ("Aguardando",):
                df_val.at[idx, 'Preco_Final'] = round(preco_sinal_v, 4)

            # Se o status foi finalizado AGORA, guardamos para atualizar o RL sem duplicidade
            if "🏆" in status_v or "❌" in status_v or "💰" in status_v:
                indices_concluidos_neste_ciclo.append(idx)

    # Força todas as colunas a voltarem a ser números nativos do Python antes do update
    df_val['ID_Sinal'] = pd.to_numeric(df_val['ID_Sinal'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    df_val['Preco_Sinal'] = df_val['Preco_Sinal'].apply(limpar_numero)
    df_val['Referencia_Fin'] = df_val['Referencia_Fin'].apply(limpar_numero)
    df_val['Monitoramento_12d'] = df_val['Monitoramento_12d'].apply(limpar_numero)
    df_val['Dias_Passados'] = pd.to_numeric(df_val['Dias_Passados'].apply(limpar_numero), errors='coerce').fillna(0).astype(int)
    # Dias_Ate_Alvo fica vazio ("") quando não aplicável (sinal ainda Aguardando,
    # ou STOP, que não é um alvo de lucro) — não força para 0, pois 0 teria
    # significado próprio (sustentação no mesmo ciclo do sinal).
    df_val['Dias_Ate_Alvo'] = df_val['Dias_Ate_Alvo'].apply(
        lambda x: round(limpar_numero(x), 2) if str(x).strip() not in ("", "nan", "None") else ""
    )
    # Preco_Final: mesmo padrão — vazio quando ainda Aguardando, número quando concluído.
    df_val['Preco_Final'] = df_val['Preco_Final'].apply(
        lambda x: round(limpar_numero(x), 4) if str(x).strip() not in ("", "nan", "None") else ""
    )
    
    # Isolamento limpo do Estado_RL
    df_val['Estado_RL'] = df_val['Estado_RL'].astype(str).apply(lambda x: x.strip().replace("'", ""))
    df_val['Estado_RL'] = df_val['Estado_RL'].apply(lambda x: f"'{x}" if x != "" else "")

    # Salvar de volta as atualizações estruturadas — escrita atômica (parâmetro 6):
    # aba temporária primeiro, só substitui a original se a escrita for confirmada.
    ok_val = salvar_aba_atomica(
        sh, "Validacao_Sinais", headers_val, df_val.values.tolist(),
        rows_est=10000, cols_est=len(headers_val)
    )
    if ok_val:
        ws_val = sh.worksheet("Validacao_Sinais")   # referência atualizada após o rename
    else:
        print("⚠️ Validacao_Sinais não foi atualizada neste ciclo — dados anteriores preservados.")

    # ======================================================================
    # ALOCAÇÃO DE RECURSOS — pesos por item usando taxa Bayesiana + retorno
    # anualizado (velocidade real de capital via Dias_Ate_Alvo). Fonte:
    # Validacao_Sinais (cobertura sistemática dos 63 itens, sem viés de
    # seleção manual que o TradeLog teria).
    # ======================================================================
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
            ws_aloc.update(
                values=[df_alocacao.columns.tolist()] + df_alocacao.values.tolist(),
                range_name='A1',
                value_input_option='USER_ENTERED'
            )
            top3 = ", ".join(df_alocacao.head(3)['Item'].tolist())
            print(f"💰 Alocação de Recursos: {len(df_alocacao)} itens com dados suficientes "
                  f"(min. 3 sinais concluídos) | Top 3: {top3}")
        else:
            print("💰 Alocação de Recursos: aguardando sinais concluídos suficientes (mín. 3 por item).")
    except Exception as e:
        print(f"⚠️ Erro ao calcular Alocação de Recursos: {e}")

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

            # Recompensa contínua via ROI real (mesma calcular_roi_real usada na
            # Alocação de Recursos) — substitui a categórica 1.0/0.5/-1.0.
            # Um acerto de 25% de lucro agora vale mais que um de 7%; uma falha
            # que só perdeu 2% pesa menos que uma que perdeu 20% — antes as duas
            # eram idênticas (-1.0). Normalizado pelo LUCRO_REAL_DESEJADO (7%)
            # para manter a escala comparável ao sistema categórico anterior
            # (bater exatamente a meta de 7% ainda dá recompensa ~1.0).
            preco_sinal_r = limpar_numero(row_r['Preco_Sinal'])
            preco_final_r = limpar_numero(row_r['Preco_Final'])

            if preco_final_r > 0:
                roi_real_pct = calcular_roi_real(acao_rl, preco_sinal_r, preco_final_r)
                recompensa = max(-2.0, min(2.0, roi_real_pct / (LUCRO_REAL_DESEJADO * 100)))
            # Fallback categórico para sinais antigos sem Preco_Final (schema pré-migração)
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

            # Alpha decai com o número de visitas — early sinais movem o Q-value
            # com força total, mas após dezenas de observações a taxa converge
            # para o piso (0.05), estabilizando o valor em vez de oscilar
            # indefinidamente com o ruído da última amostra.
            alpha_dinamico = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas))

            q_atual = q_table[st_rl].get(acao_rl, 0.0)
            q_table[st_rl][acao_rl] = q_atual + alpha_dinamico * (recompensa - q_atual)
            q_table[st_rl][chave_n] = n_visitas

            # Atualização espelhada na Q-table item-específica (parâmetro 3).
            # Mesma recompensa, mesmo mecanismo de alpha decrescente, mas com
            # contador de visitas PRÓPRIO do item — assim itens raros continuam
            # de fato "sem histórico próprio" (peso_item baixo em calcular_veto_rl)
            # mesmo que o estado global já tenha muitas observações de outros itens.
            chave_item_estado = f"{row_r['Item']}|{st_rl}"
            if chave_item_estado not in q_table_item:
                q_table_item[chave_item_estado] = {"COMPRA": 0.0, "VENDA": 0.0, "N_COMPRA": 0, "N_VENDA": 0}

            n_visitas_item = q_table_item[chave_item_estado].get(chave_n, 0) + 1
            alpha_item     = max(ALPHA_MINIMO, 1.0 / (1 + n_visitas_item))
            q_atual_item   = q_table_item[chave_item_estado].get(acao_rl, 0.0)
            q_table_item[chave_item_estado][acao_rl] = q_atual_item + alpha_item * (recompensa - q_atual_item)
            q_table_item[chave_item_estado][chave_n] = n_visitas_item

        print(f"🧠 Cérebro RL Atualizado de forma incremental: {len(df_reforco)} novos estados processados.")

    print(f"🧠 Cérebro RL Atualizado internamente: {len(q_table)} estados globais | "
          f"{len(q_table_item)} combinações item|estado na memória RAM.")

try:
    max_id_existente = int(pd.to_numeric(df_val['ID_Sinal'], errors='coerce').max()) if not df_val.empty else 0
except:
    max_id_existente = len(df_val)
prox_id_sinal = max_id_existente + 1
novas_linhas_validacao = []

# Mapa de último sinal por item+tipo — base para a deduplicação híbrida.
# Lido do df_val atual (já em memória, sem custo extra de API).
# Estrutura: { "Leather_COMPRA": {"preco": 0.1014, "timestamp": datetime} }
mapa_ultimo_sinal  = {}
usar_dedup_simples = False
if not df_val.empty:
    try:
        for _, row_u in df_val.iterrows():
            chave_u = f"{str(row_u['Item']).strip()}_{str(row_u['Tipo']).strip().upper()}"
            # Correção 4.23.3: usa a mesma função de parsing robusto já aplicada
            # na validação de sinais (parse_data_sinal_robusta), em vez de um
            # pd.to_datetime cru que não tentava formatos alternativos — as duas
            # fontes de verdade sobre Data_Sinal agora concordam entre si.
            # Se mesmo assim vier NaT (data ilegível), pula a linha em vez de
            # deixar um timestamp inválido entrar no mapa: uma comparação
            # "NaT > timestamp_valido" é sempre False, então guardar NaT aqui
            # tanto pode mascarar silenciosamente uma entrada mais recente
            # válida quanto, se for a única linha para a chave, propagar NaT
            # para o cálculo de horas_desde mais adiante.
            ts_u = parse_data_sinal_robusta(row_u['Data_Sinal'])
            if pd.isnull(ts_u):
                continue
            preco_u = limpar_numero(row_u['Preco_Sinal'])
            if chave_u not in mapa_ultimo_sinal or ts_u > mapa_ultimo_sinal[chave_u]['timestamp']:
                mapa_ultimo_sinal[chave_u] = {'preco': preco_u, 'timestamp': ts_u}
    except Exception as e:
        print(f"⚠️  mapa_ultimo_sinal falhou: {e} — revertendo para deduplicação por data.")
        mapa_ultimo_sinal  = {}
        usar_dedup_simples = True

# Calibração híbrida: combina taxa de acerto real (Validacao_Sinais)
# + proxy de MAE do XGBoost (XGB_Audit do ciclo anterior).
# Chamada aqui, após a q_table estar atualizada e df_val pronto em memória.
# df_audit_cache vem de dentro da própria função — antes a aba XGB_Audit era
# lida 2x por ciclo (aqui e dentro de construir_calibracao_hibrida); agora é
# lida 1x só e reaproveitada, economizando 1 chamada de API por execução.
mapa_mult_desvio, mapa_veto, mapa_ajuste_roi, df_audit_cache = construir_calibracao_hibrida(sh, df_bruto, df_val_mem=df_val, q_table_mem=q_table)

# Nº de observações concluídas por estado RL — usado pelo veto adaptativo (item 1)
# para endurecer/afrouxar o threshold conforme a quantidade de evidência disponível.
n_obs_rl_map = contar_obs_por_estado(df_val)

# ==============================================================================
# 2. IA E TREINAMENTO
# ==============================================================================
cache_indicadores = {}
ranking = []
audit_linhas = []   # acumula métricas para gravar na aba XGB_Audit ao final
X_cols = ['Price', 'Lag_1', 'Lag_2', 'Lag_3', 'RSI_Micro', 'Media_Macro',
          'Desvio_Macro', 'Season_Num', 'Hora_Dia', 'Dia_Semana']

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
        # ROI_Prev_% é a previsão de movimento de preço do XGBoost — sinal direcional.
        # A taxa de marketplace (10%) já está embutida no Alvo_Saída via:
        #   alvo_lucro = (custo_medio × (1 + LUCRO_REAL_DESEJADO)) / (1 - TAXA_MARKETPLACE)
        # Aplicar a taxa aqui seria contá-la duas vezes. O ROI exibido é bruto.
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
    # ROI_Ajustado (não ROI cru): incorpora a penalidade/bônus da calibração
    # Bayesiana (mapa_ajuste_roi) na decisão de fato — antes essa penalidade
    # só reordenava df_ranking (via sort_values acima) e nunca chegava ao
    # Estado_RL, ao stop loss adaptativo, ao veto RL ou ao ROI exibido, que
    # liam a coluna 'ROI' crua. Ver ARQUITETURA.md para o histórico da mudança.
    roi_pred = df_ranking[df_ranking['Item'] == item]['ROI_Ajustado'].values[0]
    saldo_item = inv_dict.get(item, 0)

    # Itens com saldo zero não devem gerar sinais de VENDA nem entrar no FIFO.
    if saldo_item <= 0 and row['Price'] > (row['Media_Macro'] - mapa_mult_desvio.get(item, 2.0) * row['Desvio_Macro']):
        continue

    # Guarda NaN: itens recém-adicionados ao jogo (<120h de histórico) têm
    # Media_Macro=NaN. Sem esta verificação, alvo_in/alvo_out ficam NaN e
    # "nan" é gravado no Sinais_Hoje, corrompendo a planilha visualmente.
    if pd.isna(row.get('Media_Macro')) or pd.isna(row.get('Desvio_Macro')):
        continue
    custo_medio = mapa_custo_fifo.get(item, row['Media_Macro'])
    
    sujeira = max(0, saldo_item - compras_log.get(item, 0))
    origem = 2 if (compras_log.get(item, 0) > 0 and (sujeira/saldo_item if saldo_item > 0 else 0) <= TOLERANCIA_SUJEIRA) else 1

    # Piso de posição mínima: se o valor da posição (saldo × preço atual) não
    # atinge POSICAO_MINIMA_SFL, trata como Colheita mesmo com histórico de compra.
    # Evita Stop Loss e alvo de lucro em itens comprados só para uso (crafting).
    valor_posicao = saldo_item * row['Price']
    if origem == 2 and valor_posicao < POSICAO_MINIMA_SFL:
        origem = 1
    
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

    # Margem de atenção adaptativa (item 4): fração da largura da banda de
    # Bollinger, em vez do MARGEM_ATENCAO fixo de 3,5% sobre o preço.
    margem_atencao = calcular_margem_atencao(alvo_in, alvo_out)

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
    elif preco_atual <= alvo_in + margem_atencao:
        acao = "👀 PERTO DA ENTRADA"
    elif preco_atual >= alvo_out:
        if origem == 2 and alvo_lucro > alvo_out_estatistico:
            acao = "🔴 VENDER (Lucro Disponível)"
        else:
            acao = "🔴 VENDER"
    elif preco_atual >= alvo_out - margem_atencao:
        acao = "🎯 PERTO DA SAÍDA"

    if origem == 2 and preco_atual < custo_medio:
        # Stop loss adaptativo (item 3): dispara quando o ROI previsto é
        # N×MAE_relativo do item, em vez do -20% fixo (ruído para itens voláteis).
        deve_stopar, _thr_stop = calcular_stop_adaptativo(
            item, preco_atual, roi_pred / 100,
            df_audit_mem=df_audit_cache
        )
        if deve_stopar:
            acao = "⚠️ VENDER (Stop Loss)"
        elif acao == "ESPERAR":
            acao = "ESPERAR (Prejuízo)"

    if acao == "🟢 COMPRAR" and mapa_veto.get(item, False):
        acao = "🚫 COMPRA VETADA"

    # Veto RL adaptativo (item 1) + Q-table híbrida (item 3): threshold endurece
    # com nº de observações do estado, e o Q-value usado é uma mistura entre o
    # global (rápido de generalizar) e o específico do item (mais preciso).
    roi_pred_exibido = roi_pred
    if acao == "🟢 COMPRAR":
        chave_ci_atual = f"{item}|{estado_rl}"
        q_item_dados   = q_table_item.get(chave_ci_atual, {})
        q_compra_item  = q_item_dados.get("COMPRA")
        n_obs_item_at  = q_item_dados.get("N_COMPRA", 0)

        vetado_rl, bonus_roi_rl = calcular_veto_rl(
            estado_rl, q_table, n_obs_rl_map,
            q_compra_item=q_compra_item, n_obs_item=n_obs_item_at
        )
        roi_pred_exibido = roi_pred + bonus_roi_rl
        if vetado_rl:
            acao = "🚫 VETO DO CÉREBRO RL (Histórico de Prejuízo)"

    # Detector de regime (item 5): não altera o SHIFT_ALVO, só sinaliza quando
    # uma COMPRA está sendo gerada em meio a uma queda rápida (crash 24h).
    em_crash, var24h = detectar_regime(cache_indicadores.get(item))
    if em_crash and "COMPRAR" in acao:
        acao = f"⚠️ COMPRAR (QUEDA {var24h:.1f}% em 24h)"

    lista_final.append([
        item,
        # ROI visual: desconta a taxa de 10% do marketplace para mostrar o retorno real.
        # Fórmula: roi_bruto × 0.9 − 10  (ex: 8.4% → −2.4% | −25.8% → −33.2%)
        # Não altera ranking, stop loss, Validacao_Sinais nem nenhum outro cálculo.
        round(roi_pred_exibido * (1 - TAXA_MARKETPLACE) - TAXA_MARKETPLACE * 100, 1),
        acao, 
        round(alvo_in, 4), 
        round(alvo_out, 4), 
        "Revenda" if origem==2 else "Colheita", 
        row['Season'], 
        f"'{estado_rl}", 
        obter_agora_brt().strftime("%d/%m %H:%M")
    ])

    if acao in ["🟢 COMPRAR", "🔴 VENDER", "🔴 VENDER (Lucro Disponível)", "⚠️ VENDER (Stop Loss)"] \
            or (em_crash and "COMPRAR" in acao):
        agora_brt   = obter_agora_brt()
        dt_hoje_str = agora_brt.strftime("%d%m%Y")
        dt_hhmm     = agora_brt.strftime("%H%M")
        # Correção 4.23.3: antes, "or em_crash" sozinho podia forçar tipo_sinal="COMPRA"
        # mesmo quando a ação real era uma VENDA/STOP LOSS que coincidisse com um item
        # também em crash de 24h (cenário nada raro: stop loss e crash tendem a
        # acontecer juntos). Isso contaminava Referencia_Fin, Chave_Unica, a validação
        # do sinal e o aprendizado do Cerebro_RL para esse sinal. Agora tipo_sinal
        # depende só do conteúdo real de "acao" — em_crash já foi corretamente
        # aplicado antes (linha do alerta "⚠️ COMPRAR (QUEDA...)") e continua refletido
        # em "acao" quando de fato for uma compra.
        tipo_sinal  = "COMPRA" if "COMPRAR" in acao else "VENDA"
        sufixo      = "_STOP" if "Stop Loss" in acao else ""

        # --- DEDUPLICAÇÃO HÍBRIDA: mínimo 4h + variação ≥ 3% ---
        # Condição 1 — tempo mínimo: evita sinais em sequência durante spike de 1h
        # Condição 2 — variação de preço: só registra quando mercado mudou de verdade
        # STOP sempre passa independente de variação (proteção de capital não espera)
        # Resultado: itens estáveis → poucos sinais; itens voláteis → 2-4/dia
        INTERVALO_MIN_H  = 4      # horas mínimas entre sinais do mesmo item+tipo
        VARIACAO_MIN_PCT = 0.03   # 3% de variação mínima de preço

        chave_item_tipo = f"{item}_{tipo_sinal.upper()}"
        ultimo          = mapa_ultimo_sinal.get(chave_item_tipo)
        pode_gerar      = True

        if pode_gerar and ultimo:
            horas_desde = (agora_brt - ultimo['timestamp']).total_seconds() / 3600
            variacao    = abs(preco_atual - ultimo['preco']) / ultimo['preco'] if ultimo['preco'] > 0 else 1.0

            if horas_desde >= 24:
                pode_gerar = True
            elif horas_desde < INTERVALO_MIN_H:
                pode_gerar = False
            elif not sufixo and variacao < VARIACAO_MIN_PCT:
                pode_gerar = False

        # HHmm na chave distingue múltiplos sinais válidos no mesmo dia
        chave_unica = f"{item}_{tipo_sinal}_{dt_hoje_str}_{dt_hhmm}{sufixo}"

        # A deduplicação real é feita por 'pode_gerar' (mapa_ultimo_sinal, acima):
        # tempo mínimo + variação de preço. Uma checagem "chave_unica not in
        # df_val['Chave_Unica']" existia aqui antes, mas como a chave inclui
        # HH:mm, ela só colidiria se o bot rodasse duas vezes no mesmo minuto —
        # praticamente nunca acontece e não somava proteção real. Removida.
        if pode_gerar:
            if not any(x[10] == chave_unica for x in novas_linhas_validacao):
                # Fallback de segurança: se mapa_ultimo_sinal falhou, usa deduplicação
                # por data simples (1 sinal/item/dia) para evitar spam por ciclo.
                if usar_dedup_simples:
                    chave_fallback = f"{item}_{tipo_sinal}_{dt_hoje_str}{sufixo}"
                    if not df_val.empty and df_val['Chave_Unica'].astype(str).str.startswith(chave_fallback).any():
                        continue

                linha_sinais_hoje = len(lista_final) + 1
                enviar_telegram(acao, item, preco_atual, linha_sinais_hoje)

                # Referencia_Fin por tipo de sinal:
                # COMPRA     → alvo_out (preço de saída desejado, acima do sinal)
                # VENDA STOP → preco_atual (o próprio nível do stop — validação checa se
                #              o preço ficou abaixo dele após a saída)
                # VENDA comum, origem=Revenda  → custo_medio (confirma se caiu abaixo do
                #              que você realmente pagou — referência de custo é significativa)
                # VENDA comum, origem=Colheita → Media_Macro (não existe custo real a
                #              proteger num item majoritariamente colhido/grátis; o FIFO
                #              calcula a média só da fração comprada, que pode estar
                #              desconectada do saldo total. Media_Macro serve de
                #              referência neutra: "caiu abaixo do preço normal recente?")
                if tipo_sinal == "COMPRA":
                    ref_financeira = alvo_out
                elif sufixo == "_STOP":
                    ref_financeira = preco_atual   # referência = nível do stop
                elif origem == 1:
                    ref_financeira = row['Media_Macro']
                else:
                    ref_financeira = custo_medio

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
                    chave_unica,
                    "",  # Dias_Ate_Alvo — preenchido na conclusão do sinal
                    ""   # Preco_Final — preenchido na conclusão do sinal
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
    cerebro_linhas = []
    for st, acoes in q_table.items():
        cerebro_linhas.append([
            f"'{st}", 
            round(acoes.get("COMPRA", 0.0), 4), 
            round(acoes.get("VENDA", 0.0), 4),
            int(acoes.get("N_COMPRA", 0)),
            int(acoes.get("N_VENDA", 0))
        ])

    # Escrita atômica (parâmetro 6): este é o dado mais crítico de todo o
    # sistema — meses de aprendizado acumulado. Uma escrita comum (clear+update)
    # que morre no meio zera a memória inteira. Aqui, só substitui a aba
    # original depois de confirmar que os novos dados foram gravados com êxito.
    ok_cerebro = salvar_aba_atomica(
        sh, "Cerebro_RL",
        ["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
        cerebro_linhas, rows_est=5000, cols_est=5
    )
    if ok_cerebro:
        print("✅ Memória de Longo Prazo (Cerebro_RL) sincronizada com segurança!")
    else:
        print("⚠️ Cerebro_RL não foi atualizado neste ciclo — estado anterior preservado.")
except Exception as e:
    print(f"❌ Erro ao salvar a Q-Table em Cerebro_RL: {e}")

# ==============================================================================
# PERSISTÊNCIA DA Q-TABLE ITEM-ESPECÍFICA (Cerebro_RL_Item)
# ==============================================================================
try:
    cerebro_item_linhas = []
    for chave_ci, acoes in q_table_item.items():
        cerebro_item_linhas.append([
            f"'{chave_ci}",
            round(acoes.get("COMPRA", 0.0), 4),
            round(acoes.get("VENDA", 0.0), 4),
            int(acoes.get("N_COMPRA", 0)),
            int(acoes.get("N_VENDA", 0))
        ])

    ok_cerebro_item = salvar_aba_atomica(
        sh, "Cerebro_RL_Item",
        ["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"],
        cerebro_item_linhas, rows_est=20000, cols_est=5
    )
    if ok_cerebro_item:
        print("✅ Cerebro_RL_Item sincronizado com segurança!")
    else:
        print("⚠️ Cerebro_RL_Item não foi atualizado neste ciclo — estado anterior preservado.")
except Exception as e:
    print(f"❌ Erro ao salvar Cerebro_RL_Item: {e}")

print("🏁 Ciclo finalizado com sucesso!")
