"""
bot_v2.py — Bot de trading SFL V2 (produção; substitui o bot_4_24.py).

Este arquivo cuida SÓ de I/O: planilhas (gspread), trava do RecLog, cache em
disco, Telegram e a ordem de persistência. Toda a lógica está em sfl_core.py.

Variáveis de ambiente (secrets do GitHub):
  TRAVA_URL, TRAVA_TOKEN            — trava de leitura do RecLog (Apps Script)
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID  — alertas (antes o token estava hardcoded)
  DEBUG_DECISOES=1                  — (opcional) grava a aba Debug_Decisao

Ordem de persistência (importante): Validacao_Sinais -> Cerebro_RL ->
Cerebro_RL_Item logo após o reforço, ANTES de indicadores/modelo. Assim um
timeout nas etapas pesadas não perde o aprendizado dos sinais concluídos.
"""

import os
import re
import sys
import time
import pickle
import traceback
from datetime import datetime

import requests
import gspread
import pandas as pd
import xgboost as xgb
from oauth2client.service_account import ServiceAccountCredentials
from gspread_formatting import set_frozen, format_cell_range, CellFormat, TextFormat

from sfl_core import *   # noqa: F401,F403  (funções puras e constantes)

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
PLANILHA_ORIGEM = "SFL 2025_10"      # RecLog / TradeLog / Inventario — APENAS LEITURA
PLANILHA_V2 = "SFL 2025_10_V2"       # abas geridas pela V2

CACHE_DIR = "cache_v2"
CACHE_VERSION = 2                    # subir quando mudar o cálculo dos indicadores (invalida o cache)
CACHE_INDICADORES_PATH = os.path.join(CACHE_DIR, "indicadores_incrementais.pkl")
MODELO_POOLED_PATH = os.path.join(CACHE_DIR, "modelo_pooled.json")
MODELO_POOLED_META_PATH = os.path.join(CACHE_DIR, "modelo_pooled_meta.pkl")

MAX_IDADE_RECLOG_H = 3.0             # RecLog mais velho que isto = coleta parada; não gera sinais
DEBUG_DECISOES = os.environ.get("DEBUG_DECISOES", "0") == "1"

HEADERS_VAL = ["ID_Sinal", "Tipo", "Data_Sinal", "Item", "Preco_Sinal", "Referencia_Fin",
               "Monitoramento_12d", "Status", "Dias_Passados", "Estado_RL", "Chave_Unica",
               "Dias_Ate_Alvo", "Preco_Final", "Estado_RL_Continuo"]
HEADERS_Q = ["Estado_RL", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]
HEADERS_Q_ITEM = ["Item_Estado", "Q_COMPRA", "Q_VENDA", "N_COMPRA", "N_VENDA"]
HEADERS_HOJE = ["Item", "ROI_Prev_%", "Ação", "Alvo_Entrada", "Alvo_Saída",
                "Origem", "Estação", "Estado_RL", "Atualizado"]
HEADERS_DEBUG = ["Timestamp", "Item", "Preco_Atual", "Media_Macro", "Desvio_Macro", "Alvo_In", "Alvo_Out",
                 "Saldo_Item", "Origem", "Acao_Banda", "Acao_Pos_Stop", "Veto_Manual", "Vetado_RL",
                 "Em_Crash", "Acao_Final", "Horas_Desde_Ultimo_Sinal_Mesmo_Tipo", "Variacao_Pct_Vs_Ultimo"]


# ==============================================================================
# TELEGRAM / TRAVA
# ==============================================================================
def _telegram_post(texto, parse_mode=None):
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ TELEGRAM_TOKEN/TELEGRAM_CHAT_ID não configurados — alerta não enviado.")
        return False
    payload = {"chat_id": chat_id, "text": texto}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=15)
        if not r.ok:
            print(f"❌ Telegram HTTP {r.status_code}: {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem para o Telegram: {e}")
        return False


def enviar_telegram(acao, ativo, preco, linha):
    emoji = "🟢" if "COMPRAR" in acao.upper() else ("⚠️" if "STOP" in acao.upper() else "🔴")
    preco_txt = f"{preco:.4f}".replace(".", ",")
    msg = (f"{emoji} *ALERTA DE OPERAÇÃO* {emoji}\n\n"
           f"*Ação:* {acao.upper()}\n*Ativo:* {ativo}\n"
           f"*Preço:* {preco_txt}\n"
           f"*Linha:* {linha}\n*Hora:* {obter_agora_brt().strftime('%d/%m %H:%M:%S')}")
    if not _telegram_post(msg, "Markdown"):
        _telegram_post(msg.replace("*", ""))   # fallback sem Markdown


def trava(acao):
    """Handshake com o Apps Script: 'adquirir' antes de ler o RecLog, 'liberar' logo depois. Falha = segue sem trava."""
    url, token = os.environ.get("TRAVA_URL"), os.environ.get("TRAVA_TOKEN")
    if not url or not token:
        print(f"⚠️ TRAVA_URL/TRAVA_TOKEN ausentes — leitura do RecLog SEM trava ({acao}).")
        return False
    try:
        r = requests.post(url, json={"token": token, "acao": acao}, timeout=90)
        ok = bool(r.ok and r.json().get("ok"))
        print(f"{'🔒' if acao == 'adquirir' else '🔓'} trava '{acao}': {'ok' if ok else 'falhou — ' + r.text[:150]}")
        return ok
    except Exception as e:
        print(f"⚠️ trava '{acao}' falhou ({e}) — seguindo.")
        return False


# ==============================================================================
# PLANILHAS
# ==============================================================================
def salvar_aba_atomica(sh, titulo, header, linhas, rows_est=None, cols_est=None, formatar_header=True):
    """Grava numa aba temporária; só substitui a original depois de confirmar sucesso."""
    titulo_temp = f"{titulo}_tmp_{int(datetime.now().timestamp())}"
    n_linhas, n_cols = len(linhas) + 1, len(header)
    ws_temp = None
    try:
        ws_temp = sh.add_worksheet(title=titulo_temp, rows=str(max(rows_est or 0, n_linhas + 100)),
                                   cols=str(max(cols_est or 0, n_cols)))
        ws_temp.update(values=[header] + linhas, range_name="A1", value_input_option="USER_ENTERED")
        if formatar_header:
            set_frozen(ws_temp, rows=1)
            ultima_col = chr(ord("A") + min(n_cols, 26) - 1)
            format_cell_range(ws_temp, f"A1:{ultima_col}1", CellFormat(textFormat=TextFormat(bold=True)))
    except Exception as e:
        print(f"❌ Falha ao escrever aba temporária de '{titulo}': {e} — original preservada.")
        if ws_temp is not None:
            try:
                sh.del_worksheet(ws_temp)
            except Exception:
                pass
        return False
    try:
        try:
            sh.del_worksheet(sh.worksheet(titulo))
        except gspread.exceptions.WorksheetNotFound:
            pass
        ws_temp.update_title(titulo)
        return True
    except Exception as e:
        print(f"⚠️ '{titulo}' escrita, mas a troca de nome falhou: {e}. Dados seguros em '{titulo_temp}' — verificar.")
        return False


def limpar_abas_orfas(sh):
    """Apaga *_tmp_<ts> com mais de 1h SOMENTE se a aba original existe (senão o tmp é a única cópia)."""
    try:
        wss = sh.worksheets()
        titulos = {w.title for w in wss}
        for w in wss:
            m = re.match(r"^(.*)_tmp_(\d+)$", w.title)
            if m and m.group(1) in titulos and time.time() - int(m.group(2)) > 3600:
                sh.del_worksheet(w)
                print(f"🧹 Aba órfã removida: {w.title}")
    except Exception as e:
        print(f"⚠️ Limpeza de abas órfãs falhou: {e}")


def carregar_q(sh, nome, header, linhas_init):
    """
    Lê uma Q-table. Só cria a aba se ela NÃO EXISTE. Qualquer outro erro aborta
    o ciclo: antes, uma falha transitória de API virava Q-table vazia, que era
    gravada por cima da original no fim do ciclo.
    """
    q = {}
    try:
        ws = sh.worksheet(nome)
    except gspread.exceptions.WorksheetNotFound:
        sh.add_worksheet(title=nome, rows=str(linhas_init), cols="5")
        sh.worksheet(nome).update(values=[header], range_name="A1", value_input_option="USER_ENTERED")
        print(f"🧠 Nova aba {nome} criada.")
        return q
    dados = ws.get_all_values()
    for row in dados[1:]:
        if len(row) >= 3:
            q[str(row[0]).strip().replace("'", "")] = {
                "COMPRA": limpar_numero(row[1]), "VENDA": limpar_numero(row[2]),
                "N_COMPRA": int(limpar_numero(row[3])) if len(row) > 3 else 0,
                "N_VENDA": int(limpar_numero(row[4])) if len(row) > 4 else 0,
            }
    print(f"🧠 {nome}: {len(q)} entradas carregadas.")
    return q


def _linhas_q(q):
    return [[f"'{k}", round(a.get("COMPRA", 0.0), 4), round(a.get("VENDA", 0.0), 4),
             int(a.get("N_COMPRA", 0)), int(a.get("N_VENDA", 0))] for k, a in q.items()]


def salvar_q_tables(sh, q_table, q_table_item):
    ok1 = salvar_aba_atomica(sh, "Cerebro_RL", HEADERS_Q, _linhas_q(q_table), rows_est=5000, cols_est=5)
    ok2 = salvar_aba_atomica(sh, "Cerebro_RL_Item", HEADERS_Q_ITEM, _linhas_q(q_table_item), rows_est=20000, cols_est=5)
    print(f"{'✅' if ok1 else '⚠️'} Cerebro_RL | {'✅' if ok2 else '⚠️'} Cerebro_RL_Item")
    return ok1 and ok2


def _obter_ou_criar(sh, nome, rows, cols, header=None, bold_ate=None):
    try:
        return sh.worksheet(nome)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=nome, rows=str(rows), cols=str(cols))
        if header:
            ws.update(values=[header], range_name="A1", value_input_option="USER_ENTERED")
        set_frozen(ws, rows=1)
        if bold_ate:
            format_cell_range(ws, f"A1:{bold_ate}1", CellFormat(textFormat=TextFormat(bold=True)))
        return ws


# ==============================================================================
# CACHE EM DISCO
# ==============================================================================
def carregar_cache_indicadores():
    """Retorna (cache, descartado_por_versao)."""
    if os.path.exists(CACHE_INDICADORES_PATH):
        try:
            with open(CACHE_INDICADORES_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and obj.get("versao") == CACHE_VERSION:
                return obj["itens"], False
            print(f"⚠️ Cache de indicadores com versão diferente (esperado {CACHE_VERSION}) — recalculando do zero.")
            return {}, True
        except Exception as e:
            print(f"⚠️ Cache de indicadores ilegível ({e}) — recalculando do zero.")
    return {}, False


def salvar_cache_indicadores(itens):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_INDICADORES_PATH, "wb") as f:
        pickle.dump({"versao": CACHE_VERSION, "itens": itens}, f)


def deve_retreinar_pooled(agora):
    """1x por DIA DE CALENDÁRIO (BRT) — antes eram 24h corridas e o modelo podia retreinar toda hora."""
    if not (os.path.exists(MODELO_POOLED_META_PATH) and os.path.exists(MODELO_POOLED_PATH)):
        return True
    try:
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)
        ult = meta.get("ultimo_treino")
        return ult is None or ult.date() != agora.date()
    except Exception:
        return True


def salvar_modelo_pooled(modelo, categorias, agora):
    os.makedirs(CACHE_DIR, exist_ok=True)
    modelo.save_model(MODELO_POOLED_PATH)
    with open(MODELO_POOLED_META_PATH, "wb") as f:
        pickle.dump({"ultimo_treino": agora, "categorias_item": list(categorias),
                     "features_treino": list(COLS_MODELO)}, f)


def carregar_modelo_pooled():
    try:
        with open(MODELO_POOLED_META_PATH, "rb") as f:
            meta = pickle.load(f)
        if meta.get("features_treino") != COLS_MODELO:
            print("⚠️ Schema de features do modelo em cache difere do código — retreinando.")
            return None, None
        m = xgb.XGBRegressor(enable_categorical=True, tree_method="hist")
        m.load_model(MODELO_POOLED_PATH)
        return m, meta.get("categorias_item")
    except Exception as e:
        print(f"⚠️ Falha ao carregar modelo em cache ({e}) — será retreinado.")
        return None, None


# ==============================================================================
# CICLO PRINCIPAL
# ==============================================================================
def main():
    t0 = time.time()
    agora = obter_agora_brt()
    print(f"🚀 Bot V2 (produção) — {agora.strftime('%d/%m/%Y %H:%M')} BRT")

    escopo = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    cred = ServiceAccountCredentials.from_json_keyfile_name("credenciais_robo.json", escopo)
    gc = gspread.authorize(cred)
    sh_origem, sh = gc.open(PLANILHA_ORIGEM), gc.open(PLANILHA_V2)
    limpar_abas_orfas(sh)

    # ------------------------------------------------------------------
    # 1. Leitura da origem (com trava do RecLog; somente leitura)
    # ------------------------------------------------------------------
    trava("adquirir")
    try:
        inv_data = sh_origem.worksheet("Inventario").get_all_values()
        trade_data = sh_origem.worksheet("TradeLog").get_all_values()
        rec_data = sh_origem.worksheet("RecLog").get_all_values()
    finally:
        trava("liberar")

    inv_dict = {r[0]: limpar_numero(r[1]) for r in inv_data[1:] if len(r) > 1 and r[0]}
    df_trade = pd.DataFrame(trade_data[1:], columns=trade_data[0])
    compras_log = (df_trade[df_trade["Tipo"].astype(str).str.contains("Compra", case=False, na=False)]
                   .groupby("Nome do Item")["Qtd"].apply(lambda x: sum(map(limpar_numero, x))).to_dict())
    mapa_custo_fifo, mapa_data_lote = calcular_fifo_hibrido(df_trade, inv_dict)

    df_bruto, st = preparar_df_bruto(rec_data)
    print(f"📥 RecLog: {st['total']} linhas | {st['uteis']} úteis | {st['preco_zero']} com preço 0 descartadas | "
          f"{st['duplicadas']} duplicadas | {st['data_invalida']} com data inválida")
    if df_bruto.empty:
        raise RuntimeError("RecLog sem dados úteis.")

    ts_max = df_bruto["Timestamp"].max()
    idade_h = (agora - ts_max).total_seconds() / 3600
    if idade_h > MAX_IDADE_RECLOG_H:
        msg = f"RecLog parado: último registro há {idade_h:.1f}h ({ts_max}). Sem sinais neste ciclo."
        print("🛑 " + msg)
        if idade_h < MAX_IDADE_RECLOG_H + 1:   # com ciclo horário, avisa uma única vez por incidente
            _telegram_post("⚠️ Bot V2: " + msg)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Validação de sinais + reforço do RL
    # ------------------------------------------------------------------
    print("🔄 Validação de sinais...")
    try:
        ws_val = sh.worksheet("Validacao_Sinais")
        val_data = ws_val.get_all_values()
    except gspread.exceptions.WorksheetNotFound:
        ws_val = sh.add_worksheet(title="Validacao_Sinais", rows="10000", cols=str(len(HEADERS_VAL)))
        val_data = []
    if not val_data:
        ws_val.update(values=[HEADERS_VAL], range_name="A1", value_input_option="USER_ENTERED")
        val_data = [HEADERS_VAL]

    if "Estado_RL_Continuo" not in val_data[0]:
        print("⚠️ Injetando coluna 'Estado_RL_Continuo' para compatibilidade...")
        df_val = (pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1
                  else pd.DataFrame(columns=val_data[0])).reindex(columns=HEADERS_VAL).fillna("")
    else:
        df_val = pd.DataFrame(val_data[1:], columns=val_data[0]) if len(val_data) > 1 else pd.DataFrame(columns=HEADERS_VAL)

    q_table = carregar_q(sh, "Cerebro_RL", HEADERS_Q, 5000)
    q_table_item = carregar_q(sh, "Cerebro_RL_Item", HEADERS_Q_ITEM, 20000)

    if not df_val.empty:
        concluidos = atualizar_sinais_abertos(df_val, df_bruto, agora)
        df_val = normalizar_df_val(df_val)
        n_reforco = aplicar_reforco(df_val, concluidos, q_table, q_table_item)

        # Persistência IMEDIATA (antes das etapas pesadas): sinais -> Q-tables
        if salvar_aba_atomica(sh, "Validacao_Sinais", HEADERS_VAL, df_val.values.tolist(),
                              rows_est=10000, cols_est=len(HEADERS_VAL)):
            if n_reforco:
                salvar_q_tables(sh, q_table, q_table_item)
                print(f"🧠 RL atualizado e salvo: {n_reforco} sinais concluídos processados.")
        else:
            print("⚠️ Validacao_Sinais não atualizada — Q-tables NÃO foram salvas neste ciclo (evita reforço duplicado).")
            # desfaz o reforço em memória: reprocessa no próximo ciclo
            q_table = carregar_q(sh, "Cerebro_RL", HEADERS_Q, 5000)
            q_table_item = carregar_q(sh, "Cerebro_RL_Item", HEADERS_Q_ITEM, 20000)

        try:
            df_aloc = calcular_alocacao_recursos(preparar_concluidos(df_val), dias_swing=DIAS_SWING)
            if not df_aloc.empty:
                ws_aloc = _obter_ou_criar(sh, "Alocacao_Recursos", 200, 8, bold_ate="H")
                ws_aloc.clear()
                ws_aloc.update(values=[df_aloc.columns.tolist()] + df_aloc.values.tolist(),
                               range_name="A1", value_input_option="USER_ENTERED")
                print(f"💰 Alocação de Recursos: {len(df_aloc)} itens.")
        except Exception as e:
            print(f"⚠️ Alocação de Recursos falhou: {e}")
    elif not q_table:
        print("🌱 Validacao_Sinais vazia e Q-table vazia — bootstrap.")

    # ------------------------------------------------------------------
    # 3. Calibração
    # ------------------------------------------------------------------
    try:
        audit_data = sh.worksheet("XGB_Audit").get_all_values()
        df_audit = (pd.DataFrame(audit_data[1:], columns=audit_data[0]) if len(audit_data) > 1
                    else pd.DataFrame(columns=HEADERS_AUDIT))
    except gspread.exceptions.WorksheetNotFound:
        df_audit = pd.DataFrame(columns=HEADERS_AUDIT)
    df_audit = normalizar_audit(df_audit)
    mapa_mult, mapa_veto, mapa_ajuste_roi, _ = calibrar_hibrido(df_val, df_audit, df_bruto)

    # ------------------------------------------------------------------
    # 4. Indicadores (cache incremental) + features
    # ------------------------------------------------------------------
    cache_persist, cache_descartado = carregar_cache_indicadores()
    indices_insumos = construir_indices_insumos(df_bruto)
    cache_ind = {}
    for item in df_bruto["Item"].unique():
        try:
            d_it = construir_features_item(df_bruto, item, cache_persist, indices_insumos)
            if d_it is not None and not d_it.empty:
                cache_ind[item] = d_it
        except Exception as e:
            print(f"⚠️ Indicadores falharam para '{item}': {e}")
    print(f"📊 Indicadores: {len(cache_ind)} itens.")

    # ------------------------------------------------------------------
    # 5. Modelo pooled (retorno em 12 dias)
    # ------------------------------------------------------------------
    df_treino, df_teste, categorias = montar_datasets_pooled(cache_ind)
    modelo = None
    if not (cache_descartado or deve_retreinar_pooled(agora)):
        modelo, cats_salvas = carregar_modelo_pooled()
        if modelo is not None:
            categorias = cats_salvas or categorias
    if modelo is None and not df_treino.empty:
        print("🏋️ Treinando modelo pooled (retorno 12d)...")
        modelo = treinar_modelo_pooled(df_treino, categorias)
        if modelo is not None:
            salvar_modelo_pooled(modelo, categorias, agora)

    audit_linhas = calcular_audit_linhas(modelo, categorias, df_treino, df_teste, agora.strftime("%d/%m/%Y %H:%M"))
    rois = prever_rois(modelo, categorias, cache_ind) if modelo is not None else {}
    print(f"🔮 Previsões: {len(rois)} itens.")

    # ------------------------------------------------------------------
    # 6. Sinais. Itens em carteira SEMPRE são avaliados (sem previsão: roi = 0)
    # ------------------------------------------------------------------
    roi_ajust = {it: r + mapa_ajuste_roi.get(it, 0.0) for it, r in rois.items()}
    fila = sorted(roi_ajust, key=roi_ajust.get, reverse=True)
    em_carteira = [it for it, s in inv_dict.items() if s > 0 and it in cache_ind and it not in roi_ajust]
    if em_carteira:
        print(f"📦 {len(em_carteira)} item(ns) em carteira sem previsão do modelo — avaliados por banda/stop: {em_carteira}")
    fila += em_carteira

    max_id = int(pd.to_numeric(df_val["ID_Sinal"], errors="coerce").max()) if not df_val.empty else 0
    prox_id = (max_id if max_id == max_id else 0) + 1     # NaN-safe
    mapa_ultimo, ignoradas = montar_mapa_ultimo_sinal(df_val)
    if ignoradas:
        print(f"⚠️ {ignoradas} linha(s) corrompida(s) de Validacao_Sinais ignoradas no dedup.")

    novas_validacao, alertas, lista_final, debug_linhas = [], [], [], []
    ts_global = df_bruto["Timestamp"].max()

    for item in fila:
        d_it = cache_ind[item]
        row = d_it.iloc[-1]
        if (ts_global - row["Timestamp"]).total_seconds() / 3600 > MAX_IDADE_PRECO_ITEM_H:
            continue   # sem preço recente (ex.: sem oferta) — não decide com dado velho

        saldo = inv_dict.get(item, 0)
        custo = mapa_custo_fifo.get(item, row["Media_Macro"])
        dl = mapa_data_lote.get(item)
        dias_cust = (agora - dl).days if dl is not None else None

        r = decidir_item(
            item, row, roi_ajust.get(item, 0.0), saldo, compras_log.get(item, 0), custo, dias_cust,
            mapa_mult.get(item, 2.0), bool(mapa_veto.get(item, False)), q_table, q_table_item,
            df_audit_mem=df_audit, d_item_hist=d_it)
        if r is None:
            continue

        preco = row["Price"]
        chave_tipo = f"{item}_STOP" if r["sufixo"] else f"{item}_{r['tipo_sinal']}"
        ultimo = mapa_ultimo.get(chave_tipo)

        if DEBUG_DECISOES:
            ch_dbg = f"{item}_COMPRA" if "COMPRAR" in r["acao_banda"] else (f"{item}_VENDA" if "VENDER" in r["acao_banda"] else None)
            u = mapa_ultimo.get(ch_dbg) if ch_dbg else None
            h_dbg = round((agora - u["timestamp"]).total_seconds() / 3600, 2) if u else None
            v_dbg = round(abs(preco - u["preco"]) / u["preco"] * 100, 2) if (u and u["preco"] > 0) else None
            debug_linhas.append([agora.strftime("%d/%m/%Y %H:%M"), item, round(preco, 4), round(row["Media_Macro"], 4),
                                 round(row["Desvio_Macro"], 4), round(r["alvo_in"], 4), round(r["alvo_out"], 4), saldo,
                                 "Revenda" if r["origem"] == 2 else "Colheita", r["acao_banda"], r["acao_pos_stop"],
                                 r["veto_manual"], r["vetado_rl"], r["em_crash"], r["acao"], h_dbg, v_dbg])

        lista_final.append([
            item, round(r["roi_exibido"] * (1 - TAXA_MARKETPLACE) - TAXA_MARKETPLACE * 100, 1), r["acao"],
            round(r["alvo_in"], 4), round(r["alvo_out"], 4), "Revenda" if r["origem"] == 2 else "Colheita",
            row["Season"], f"'{r['estado_rl']}", agora.strftime("%d/%m %H:%M")])

        if r["gera_sinal"] and pode_gerar_sinal(ultimo, preco, agora, bool(r["sufixo"])):
            chave_unica = f"{item}_{r['tipo_sinal']}_{agora.strftime('%d%m%Y')}_{agora.strftime('%H%M')}{r['sufixo']}"
            if not any(x[10] == chave_unica for x in novas_validacao):
                novas_validacao.append([
                    int(prox_id), r["tipo_sinal"], agora.strftime("%d/%m/%Y %H:%M"), item,
                    round(preco, 4), round(r["ref_fin"], 4), round(preco, 4), "Aguardando", 0,
                    f"'{r['estado_rl']}", chave_unica, "", "", r["estado_rl_continuo"]])
                alertas.append((r["acao"], item, preco, len(lista_final) + 1))
                prox_id += 1
                mapa_ultimo[chave_tipo] = {"preco": preco, "timestamp": agora}

    # Telegram SÓ depois de gravar (antes: alerta saía mesmo se a gravação falhasse)
    if novas_validacao:
        try:
            _obter_ou_criar(sh, "Validacao_Sinais", 10000, len(HEADERS_VAL)).append_rows(
                novas_validacao, value_input_option="USER_ENTERED")
            print(f"📌 {len(novas_validacao)} novo(s) sinal(is) registrados.")
            for a in alertas:
                enviar_telegram(*a)
        except Exception as e:
            print(f"❌ Falha ao gravar novos sinais — alertas NÃO enviados: {e}")
            _telegram_post(f"⚠️ Bot V2: {len(novas_validacao)} sinal(is) gerado(s) mas a gravação falhou ({str(e)[:120]}).")

    # ------------------------------------------------------------------
    # 7. Persistência restante
    # ------------------------------------------------------------------
    if lista_final:
        try:
            ws_hoje = _obter_ou_criar(sh, "Sinais_Hoje", 200, 9, bold_ate="I")
            ws_hoje.clear()
            ws_hoje.update(values=[HEADERS_HOJE] + lista_final, range_name="A1", value_input_option="USER_ENTERED")
            print(f"📋 Sinais_Hoje: {len(lista_final)} itens.")
        except Exception as e:
            print(f"❌ Erro ao salvar Sinais_Hoje: {e}")
    else:
        print("⚠️ Nenhum item para Sinais_Hoje — painel anterior mantido (não sobrescrito com vazio).")

    try:
        n_itens = df_bruto["Item"].nunique()
        ws_audit = _obter_ou_criar(sh, "XGB_Audit", n_itens * 72 * 2 + 500, 8, header=HEADERS_AUDIT, bold_ate="H")
        if audit_linhas:
            atuais = ws_audit.get_all_values()
            header = atuais[0] if atuais else HEADERS_AUDIT
            todas = (atuais[1:] if len(atuais) > 1 else []) + audit_linhas
            todas = todas[-(n_itens * 72 * 2):]
            ws_audit.clear()
            ws_audit.update(values=[header] + todas, range_name="A1", value_input_option="USER_ENTERED")
            print(f"📊 XGB_Audit: {len(audit_linhas)} novos | {len(todas)} retidos.")
    except Exception as e:
        print(f"❌ Erro ao salvar XGB_Audit: {e}")

    if DEBUG_DECISOES and debug_linhas:
        try:
            n_itens = df_bruto["Item"].nunique()
            ws_dbg = _obter_ou_criar(sh, "Debug_Decisao", n_itens * 72 + 500, len(HEADERS_DEBUG),
                                     header=HEADERS_DEBUG, bold_ate=chr(64 + len(HEADERS_DEBUG)))
            atuais = ws_dbg.get_all_values()
            header = atuais[0] if atuais else HEADERS_DEBUG
            todas = ((atuais[1:] if len(atuais) > 1 else []) + debug_linhas)[-(n_itens * 72):]
            ws_dbg.clear()
            ws_dbg.update(values=[header] + todas, range_name="A1", value_input_option="USER_ENTERED")
            print(f"🔍 Debug_Decisao: {len(debug_linhas)} novas | {len(todas)} retidas.")
        except Exception as e:
            print(f"❌ Erro ao salvar Debug_Decisao: {e}")

    try:
        precos = {it: d.iloc[-1]["Price"] for it, d in cache_ind.items() if not d.empty}
        df_port = calcular_exposicao_por_cluster(inv_dict, precos)
        if not df_port.empty:
            ws_port = _obter_ou_criar(sh, "Portfolio_Correlacao", 50, 4, bold_ate="D")
            ws_port.clear()
            ws_port.update(values=[df_port.columns.tolist()] + df_port.values.tolist(),
                           range_name="A1", value_input_option="USER_ENTERED")
            print(f"📦 Portfolio_Correlacao: {len(df_port)} clusters (informativo).")
    except Exception as e:
        print(f"⚠️ Portfólio falhou: {e}")

    try:
        salvar_cache_indicadores(cache_persist)
        print("💾 Cache incremental salvo.")
    except Exception as e:
        print(f"⚠️ Falha ao salvar cache: {e}")

    print(f"🏁 Ciclo V2 concluído em {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        _telegram_post(f"❌ Bot V2 falhou: {type(exc).__name__}: {str(exc)[:300]}")
        sys.exit(1)
