#ifndef SFL_SHEETS_H
#define SFL_SHEETS_H

#include <stddef.h>

/* Tabela de valores em formato "ragged" (linhas de tamanhos diferentes),
 * igual ao que ws.get_all_values() devolve em Python (lista de listas). */
typedef struct {
    char ***rows;      /* rows[i][j] = célula (string), sempre em formato de exibição */
    int *n_cols;        /* n_cols[i] = nº de colunas preenchidas na linha i */
    int n_rows;
} SheetValues;

void sheet_values_free(SheetValues *sv);

typedef struct {
    char access_token[2048];
    char spreadsheet_id[128];
} SheetsClient;

/* Equivalente a gc.open(nome): busca via Drive API v3 o arquivo de nome
 * exato e tipo application/vnd.google-apps.spreadsheet, e preenche
 * client->spreadsheet_id. Retorna 0 em sucesso. */
int drive_localizar_planilha(const char *access_token, const char *nome, char *out_id, size_t out_size);

/* Lê um intervalo (ex.: "RecLog" ou "RecLog!A1:D") — equivalente a
 * ws.get_all_values(). O chamador deve chamar sheet_values_free() depois. */
int sheets_get_values(const SheetsClient *cli, const char *range, SheetValues *out);

/* Escreve values (n_rows x n_cols, retangular) a partir de range (ex. "Aba!A1"),
 * com valueInputOption=USER_ENTERED (igual ao value_input_option do gspread
 * usado no bot — necessário para o Sheets interpretar "'0123100" como texto
 * e não subtrair o apóstrofo, e números como número mesmo vindo como string). */
int sheets_update_values(const SheetsClient *cli, const char *range,
                          char ***values, int n_rows, int n_cols);

/* Acrescenta linhas ao final de uma aba — equivalente a ws.append_rows(). */
int sheets_append_rows(const SheetsClient *cli, const char *sheet_name,
                        char ***values, int n_rows, int n_cols);

/* Limpa todo o conteúdo de uma aba — equivalente a ws.clear(). */
int sheets_clear(const SheetsClient *cli, const char *sheet_name);

/* add_worksheet / del_worksheet / rename via spreadsheets.batchUpdate —
 * usados pela escrita atômica (salvar_aba_atomica). */
int sheets_add_worksheet(const SheetsClient *cli, const char *title, int rows, int cols);
int sheets_delete_worksheet_by_title(const SheetsClient *cli, const char *title);
int sheets_rename_worksheet(const SheetsClient *cli, const char *old_title, const char *new_title);

/* Retorna 1 se a aba existe, 0 se não (usado para replicar o padrão
 * try: sh.worksheet(x) / except: sh.add_worksheet(...) do Python). */
int sheets_worksheet_existe(const SheetsClient *cli, const char *title);

#endif
