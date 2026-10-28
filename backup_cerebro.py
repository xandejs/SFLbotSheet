import os
import re
import gspread
import pandas as pd
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
NOME_PLANILHA = "SFL 2025_10_V2"

# Defina aqui todas as abas que você deseja fazer backup
ABAS_PARA_BACKUP = [
    "Cerebro_RL",
    "Validacao_Sinais",
    "Cerebro_RL_Item",
    "Alocacao_Recursos"  # Adicione quantas abas precisar aqui
]

PASTA_BACKUP = "backups_cerebro"
BACKUPS_A_MANTER = 3  # quantos backups mais recentes manter POR ABA
CAMINHO_CREDENCIAIS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credenciais.json")


def rotacionar_backups(nome_aba, manter=BACKUPS_A_MANTER):
    """
    Apaga os backups mais antigos de UMA aba, mantendo só os `manter` mais recentes.

    O match é feito por regex exata (backup_<aba>_AAAAMMDD_HHMMSS.csv), então
    'Cerebro_RL' NÃO pega por engano os arquivos de 'Cerebro_RL_Item'.
    O timestamp no nome ordena corretamente como texto (mais novo = maior).
    """
    padrao = re.compile(rf"^backup_{re.escape(nome_aba)}_(\d{{8}}_\d{{6}})\.csv$")
    arquivos = sorted(
        (f for f in os.listdir(PASTA_BACKUP) if padrao.match(f)),
        reverse=True,  # mais recentes primeiro
    )

    for antigo in arquivos[manter:]:
        try:
            os.remove(os.path.join(PASTA_BACKUP, antigo))
            print(f"🗑️ Backup antigo removido: {antigo}")
        except Exception as e:
            print(f"⚠️ Não foi possível remover '{antigo}': {e}")


def executar_backup():
    print(f"🔄 Iniciando processo de backup múltiplo... [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

    # 1. Garante que a pasta de destino exista localmente
    if not os.path.exists(PASTA_BACKUP):
        os.makedirs(PASTA_BACKUP)
        print(f"📁 Pasta '{PASTA_BACKUP}' criada com sucesso.")

    try:
        # 2. Autenticação na API do Google (feita apenas UMA vez para todas as abas)
        escopos = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credenciais = ServiceAccountCredentials.from_json_keyfile_name(CAMINHO_CREDENCIAIS, escopos)
        gc = gspread.authorize(credenciais)
        sh = gc.open(NOME_PLANILHA)

        # Geramos um único timestamp para sincronizar todos os arquivos do mesmo lote
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    except Exception as e:
        print(f"❌ Erro crítico na autenticação ou abertura da planilha: {e}")
        return

    # 3. Loop para processar cada aba individualmente
    for nome_aba in ABAS_PARA_BACKUP:
        print(f"\n--- Processando aba: {nome_aba} ---")
        try:
            ws = sh.worksheet(nome_aba)
            dados_brutos = ws.get_all_values()

            if len(dados_brutos) <= 1:
                print(f"⚠️ Aba '{nome_aba}' está vazia ou contém apenas cabeçalho. Pulando...")
                continue

            # 4. Conversão para DataFrame
            df = pd.DataFrame(dados_brutos[1:], columns=dados_brutos[0])

            # 5. Geração do nome dinâmico com o nome da aba correspondente
            nome_arquivo = f"backup_{nome_aba}_{timestamp}.csv"
            caminho_final = os.path.join(PASTA_BACKUP, nome_arquivo)

            # 6. Salva localmente em formato CSV
            df.to_csv(caminho_final, index=False, encoding="utf-8")

            print(f"✅ Backup da aba '{nome_aba}' concluído com sucesso!")
            print(f"💾 Salvo em: {caminho_final}")
            print(f"📊 Total de registros: {len(df)}")

            # 7. Só rotaciona DEPOIS de o novo backup estar salvo: se algo falhar
            #    acima, os backups antigos continuam intactos.
            rotacionar_backups(nome_aba)

        except gspread.exceptions.WorksheetNotFound:
            print(f"❌ Erro: A aba '{nome_aba}' não foi encontrada na planilha '{NOME_PLANILHA}'.")
        except Exception as e:
            print(f"❌ Erro ao processar a aba '{nome_aba}': {e}")

    print("\n🏁 Processo de backup finalizado.")


if __name__ == "__main__":
    executar_backup()
