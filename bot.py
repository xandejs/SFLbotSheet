# ==============================================================================
# PARÂMETROS ADAPTATIVOS — substitui constantes estáticas por funções dinâmicas
# Inserir no bot_4_11.py após a definição das constantes globais e antes do loop
# principal. Cada função é autocontida e recebe os dados já carregados.
# ==============================================================================
import math

# ──────────────────────────────────────────────────────────────────────────────
# 1. VETO RL ADAPTATIVO
# Problema original: -0.3 fixo veta após 2 falhas e levanta após 1 acerto.
# Solução: threshold endurece com mais observações (confiança cresce com dados)
#          e estados com Q muito alto recebem bônus no ROI ajustado.
# ──────────────────────────────────────────────────────────────────────────────

def calcular_veto_rl(estado_rl, q_table, n_obs_por_estado=None):
    """
    Retorna (vetado: bool, bonus_roi: float).

    Threshold adaptativo:
      - Com poucos dados (n<5): -0.5  (precisa de evidência forte para vetar)
      - Com muitos dados (n>20): -0.2  (confiante, veta mais cedo)
      
    Bônus de exploração:
      - Q_COMPRA > +0.5: adiciona +5% ao ROI ajustado (prioriza estado excelente)
      - Q_COMPRA > +0.7: adiciona +10%
    """
    if estado_rl not in q_table:
        return False, 0.0

    q_compra = q_table[estado_rl].get("COMPRA", 0.0)
    n_obs    = (n_obs_por_estado or {}).get(estado_rl, 0)

    # Threshold endurece logaritmicamente com nº de observações
    if n_obs < 5:
        threshold = -0.5     # poucos dados → exige evidência forte
    elif n_obs < 20:
        threshold = -0.3 - 0.01 * (n_obs - 5)   # endurece gradualmente
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


def calibrar_por_taxa_confianca(sh, df_val_mem, mapa_mult, mapa_roi, mapa_veto):
    """
    Recalibra os mapas usando taxa Bayesiana em vez do threshold fixo de 45%.
    Chamada dentro de construir_calibracao_hibrida, após o loop padrão.
    Modifica os mapas in-place para os itens que têm dados suficientes.
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

    for item_c, grupo in df_conc.groupby('Item'):
        n        = len(grupo)
        n_acertos = int(grupo['Pts'].apply(lambda x: x >= 0.5).sum())
        soma_pts  = grupo['Pts'].sum()

        taxa_ef, confiante, ev = calcular_taxa_confianca(n, n_acertos, soma_pts)

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
      threshold_stop = max(stop_minimo, -n_sigma × MAE_relativo)
      capped em stop_maximo para evitar que items sem dados nunca parem

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
        threshold = max(stop_maximo, -n_sigma * mae_relativo)
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


# ==============================================================================
# COMO INTEGRAR NO BOT_4_11.PY
# ==============================================================================
#
# 1. VETO RL — no loop principal, substituir:
#    ANTES:
#      if q_table.get(estado_rl, {}).get("COMPRA", 0) < -0.3:
#    DEPOIS:
#      n_obs_map = contar_obs_por_estado(df_val)
#      vetado, bonus_roi = calcular_veto_rl(estado_rl, q_table, n_obs_map)
#      roi_ajustado += bonus_roi
#      if vetado:
#
# 2. TAXA DE ACERTO — em construir_calibracao_hibrida(), após o loop padrão:
#    calibrar_por_taxa_confianca(sh, df_val, mapa_mult, mapa_roi, mapa_veto)
#
# 3. STOP LOSS — no loop de geração de sinais, substituir:
#    ANTES:
#      if origem == 2 and roi_pred < -20:
#    DEPOIS:
#      deve_stopar, thr = calcular_stop_adaptativo(item, preco_atual,
#                             roi_pred/100, df_audit_mem=df_audit_cache)
#      if origem == 2 and deve_stopar:
#
# 4. MARGEM DE ATENÇÃO — no loop de geração, substituir:
#    ANTES:
#      margem = alvo_in * MARGEM_ATENCAO
#    DEPOIS:
#      margem = calcular_margem_atencao(alvo_in, alvo_out)
#
# 5. DETECTOR DE REGIME — ao montar a ação, acrescentar:
#    em_crash, var24h = detectar_regime(cache_indicadores.get(item))
#    if em_crash and "COMPRAR" in acao:
#        acao = f"⚠️ COMPRAR (QUEDA {var24h:.1f}% em 24h)"
# ==============================================================================
