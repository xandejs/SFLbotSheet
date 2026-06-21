import os
import gspread
import pandas as pd
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10"
ABA_CEREBRO = "Cerebro_RL"
PASTA_BACKUP = "backups_cerebro"

# Caminho para o seu arquivo de credenciais JSON (ajuste se necessário)
# Se o arquivo estiver na mesma pasta, basta o nome dele: 'credenciais.json'
CAMINHO_CREDENCIAIS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credenciais.json")

def executar_backup():
    print(f"🔄 Iniciando backup da aba '{ABA_CEREBRO}'... [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    
    # 1. Garante que a pasta de destino exista localmente
    if not os.path.exists(PASTA_BACKUP):
        os.makedirs(PASTA_BACKUP)
        print(f"📁 Pasta '{PASTA_BACKUP}' criada com sucesso.")

    try:
        # 2. Autenticação na API do Google
        escopos = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credenciais = ServiceAccountCredentials.from_json_keyfile_name(CAMINHO_CREDENCIAIS, escopos)
        gc = gspread.authorize(credenciais)
        
        # 3. Abertura da planilha e coleta de dados
        sh = gc.open(NOME_PLANILHA)
        ws_cerebro = sh.worksheet(ABA_CEREBRO)
        dados_brutos = ws_cerebro.get_all_values()
        
        if len(dados_brutos) <= 1:
            print("⚠️ A aba está vazia ou contém apenas o cabeçalho. Backup cancelado para evitar sobrescrever com dados vazios.")
            return

        # 4. Conversão para DataFrame para salvar de forma limpa
        df = pd.DataFrame(dados_brutos[1:], columns=dados_brutos[0])
        
        # 5. Geração do nome do arquivo com Timestamp (Ano-Mês-Dia_Hora-Minuto)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"backup_{ABA_CEREBRO}_{timestamp}.csv"
        caminho_final = os.path.join(PASTA_BACKUP, nome_arquivo)
        
        # 6. Salva localmente em formato CSV
        df.to_csv(caminho_final, index=False, encoding="utf-8")
        
        print(f"✅ Backup concluído com sucesso!")
        print(f"💾 Arquivo salvo em: {caminho_final}")
        print(f"📊 Total de estados salvos: {len(df)}")

    except Exception as e:
        print(f"❌ Erro crítico ao executar o backup: {e}")

if __name__ == "__main__":
    executar_backup()