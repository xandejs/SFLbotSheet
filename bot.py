# ==============================================================================
# 5. SALVAMENTO E SINCRONIZAÇÃO DA Q-TABLE (CÉREBRO RL)
# ==============================================================================
try:
    ws_cerebro = sh.worksheet("Cerebro_RL")
    ws_cerebro.clear()
    
    # Prepara o cabeçalho e as linhas usando a aspa protetiva para manter texto puro no Sheets
    cerebro_linhas = [["Estado_RL", "Q_COMPRA", "Q_VENDA"]]
    for st, acoes in q_table.items():
        cerebro_linhas.append([
            f"'{st}", 
            round(acoes.get("COMPRA", 0.0), 4), 
            round(acoes.get("VENDA", 0.0), 4)
        ])
        
    ws_cerebro.update(values=cerebro_linhas, range_name='A1')
    set_frozen(ws_cerebro, rows=1)
    format_cell_range(ws_cerebro, 'A1:C1', CellFormat(textFormat=TextFormat(bold=True)))
    print("✅ Memória de Longo Prazo (Cerebro_RL) sincronizada e salva com sucesso!")
except Exception as e:
    print(f"❌ Erro catastrófico ao salvar Cerebro_RL: {e}")

print("🏁 Ciclo finalizado com sucesso!")
