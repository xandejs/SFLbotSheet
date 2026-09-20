#include "sheets.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <cjson/cJSON.h>

#define SHEETS_BASE "https://sheets.googleapis.com/v4/spreadsheets"
#define DRIVE_BASE  "https://www.googleapis.com/drive/v3/files"

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    dynbuf_append((DynBuf *)userdata, (const char *)ptr, size * nmemb);
    return size * nmemb;
}

/* HTTP genérico com Authorization: Bearer + corpo JSON opcional.
 * method: "GET", "POST", "PUT", "DELETE" (mesmos verbos que a lib requests
 * do Python usa por baixo do gspread). Retorna o HTTP status code, e
 * preenche `out` com o corpo da resposta. */
static long http_call(const char *method, const char *url, const char *access_token,
                       const char *json_body, DynBuf *out) {
    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    struct curl_slist *headers = NULL;
    char auth_header[2100];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", access_token);
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    if (strcmp(method, "POST") == 0) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body ? json_body : "");
    } else if (strcmp(method, "PUT") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body ? json_body : "");
    } else if (strcmp(method, "DELETE") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    /* GET é o default do curl, nada a fazer */

    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    if (res == CURLE_OK) curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    else fprintf(stderr, "❌ curl falhou: %s\n", curl_easy_strerror(res));

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return code;
}

static char *url_escape(const char *s) {
    CURL *curl = curl_easy_init();
    char *escaped = curl_easy_escape(curl, s, 0);
    char *result = strdup(escaped);
    curl_free(escaped);
    curl_easy_cleanup(curl);
    return result;
}

void sheet_values_free(SheetValues *sv) {
    if (!sv || !sv->rows) return;
    for (int i = 0; i < sv->n_rows; i++) {
        for (int j = 0; j < sv->n_cols[i]; j++) free(sv->rows[i][j]);
        free(sv->rows[i]);
    }
    free(sv->rows);
    free(sv->n_cols);
    sv->rows = NULL;
    sv->n_cols = NULL;
    sv->n_rows = 0;
}

int drive_localizar_planilha(const char *access_token, const char *nome, char *out_id, size_t out_size) {
    char query[512];
    snprintf(query, sizeof(query),
        "name='%s' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false", nome);
    char *q_esc = url_escape(query);

    char url[700];
    snprintf(url, sizeof(url), "%s?q=%s&fields=files(id,name)", DRIVE_BASE, q_esc);
    free(q_esc);

    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("GET", url, access_token, NULL, &resp);

    int ok = -1;
    if (code == 200) {
        cJSON *root = cJSON_Parse(resp.data);
        cJSON *files = root ? cJSON_GetObjectItemCaseSensitive(root, "files") : NULL;
        if (cJSON_IsArray(files) && cJSON_GetArraySize(files) > 0) {
            cJSON *first = cJSON_GetArrayItem(files, 0);
            cJSON *id = cJSON_GetObjectItemCaseSensitive(first, "id");
            if (cJSON_IsString(id)) {
                snprintf(out_id, out_size, "%s", id->valuestring);
                ok = 0;
            }
        } else {
            fprintf(stderr, "❌ Planilha '%s' não encontrada (ou sem acesso da service account)\n", nome);
        }
        if (root) cJSON_Delete(root);
    } else {
        fprintf(stderr, "❌ Drive API retornou HTTP %ld: %s\n", code, resp.data);
    }
    dynbuf_free(&resp);
    return ok;
}

int sheets_get_values(const SheetsClient *cli, const char *range, SheetValues *out) {
    char *range_esc = url_escape(range);
    char url[700];
    snprintf(url, sizeof(url), "%s/%s/values/%s?valueRenderOption=FORMATTED_VALUE",
             SHEETS_BASE, cli->spreadsheet_id, range_esc);
    free(range_esc);

    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("GET", url, cli->access_token, NULL, &resp);

    out->rows = NULL; out->n_cols = NULL; out->n_rows = 0;
    int ok = -1;

    if (code == 200) {
        cJSON *root = cJSON_Parse(resp.data);
        cJSON *values = root ? cJSON_GetObjectItemCaseSensitive(root, "values") : NULL;
        if (cJSON_IsArray(values)) {
            int nr = cJSON_GetArraySize(values);
            out->rows = calloc((size_t)nr, sizeof(char **));
            out->n_cols = calloc((size_t)nr, sizeof(int));
            out->n_rows = nr;
            for (int i = 0; i < nr; i++) {
                cJSON *row = cJSON_GetArrayItem(values, i);
                int nc = cJSON_GetArraySize(row);
                out->rows[i] = calloc((size_t)nc, sizeof(char *));
                out->n_cols[i] = nc;
                for (int j = 0; j < nc; j++) {
                    cJSON *cell = cJSON_GetArrayItem(row, j);
                    out->rows[i][j] = strdup(cJSON_IsString(cell) ? cell->valuestring : "");
                }
            }
        }
        /* values ausente = aba vazia, n_rows fica 0 (igual a [] no Python) */
        ok = 0;
        if (root) cJSON_Delete(root);
    } else {
        fprintf(stderr, "❌ get_values('%s') retornou HTTP %ld: %s\n", range, code, resp.data);
    }
    dynbuf_free(&resp);
    return ok;
}

static char *montar_json_values(char ***values, int n_rows, int n_cols_max, const char *range_field) {
    cJSON *root = cJSON_CreateObject();
    if (range_field) cJSON_AddStringToObject(root, "range", range_field);
    cJSON_AddStringToObject(root, "majorDimension", "ROWS");
    cJSON *arr = cJSON_AddArrayToObject(root, "values");
    for (int i = 0; i < n_rows; i++) {
        cJSON *row = cJSON_CreateArray();
        for (int j = 0; j < n_cols_max; j++) {
            cJSON_AddItemToArray(row, cJSON_CreateString(values[i][j] ? values[i][j] : ""));
        }
        cJSON_AddItemToArray(arr, row);
    }
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return s;
}

int sheets_update_values(const SheetsClient *cli, const char *range,
                          char ***values, int n_rows, int n_cols) {
    char *range_esc = url_escape(range);
    char url[700];
    snprintf(url, sizeof(url), "%s/%s/values/%s?valueInputOption=USER_ENTERED",
             SHEETS_BASE, cli->spreadsheet_id, range_esc);
    free(range_esc);

    char *body = montar_json_values(values, n_rows, n_cols, range);
    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("PUT", url, cli->access_token, body, &resp);
    free(body);

    int ok = (code == 200) ? 0 : -1;
    if (ok != 0) fprintf(stderr, "❌ update_values('%s') retornou HTTP %ld: %s\n", range, code, resp.data);
    dynbuf_free(&resp);
    return ok;
}

int sheets_append_rows(const SheetsClient *cli, const char *sheet_name,
                        char ***values, int n_rows, int n_cols) {
    char *name_esc = url_escape(sheet_name);
    char url[700];
    snprintf(url, sizeof(url),
        "%s/%s/values/%s:append?valueInputOption=USER_ENTERED&insertDataOption=INSERT_ROWS",
        SHEETS_BASE, cli->spreadsheet_id, name_esc);
    free(name_esc);

    char *body = montar_json_values(values, n_rows, n_cols, NULL);
    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("POST", url, cli->access_token, body, &resp);
    free(body);

    int ok = (code == 200) ? 0 : -1;
    if (ok != 0) fprintf(stderr, "❌ append_rows('%s') retornou HTTP %ld: %s\n", sheet_name, code, resp.data);
    dynbuf_free(&resp);
    return ok;
}

int sheets_clear(const SheetsClient *cli, const char *sheet_name) {
    char *name_esc = url_escape(sheet_name);
    char url[700];
    snprintf(url, sizeof(url), "%s/%s/values/%s:clear", SHEETS_BASE, cli->spreadsheet_id, name_esc);
    free(name_esc);

    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("POST", url, cli->access_token, "{}", &resp);
    int ok = (code == 200) ? 0 : -1;
    dynbuf_free(&resp);
    return ok;
}

/* Busca metadados (sheets.properties) para mapear título -> sheetId
 * numérico, exigido pelas requests de batchUpdate (rename/delete não
 * aceitam título diretamente, só o ID interno da aba). */
static int obter_sheet_id_por_titulo(const SheetsClient *cli, const char *title, int *out_id) {
    char url[512];
    snprintf(url, sizeof(url), "%s/%s?fields=sheets.properties", SHEETS_BASE, cli->spreadsheet_id);

    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("GET", url, cli->access_token, NULL, &resp);
    int ok = -1;

    if (code == 200) {
        cJSON *root = cJSON_Parse(resp.data);
        cJSON *sheets_arr = root ? cJSON_GetObjectItemCaseSensitive(root, "sheets") : NULL;
        if (cJSON_IsArray(sheets_arr)) {
            int n = cJSON_GetArraySize(sheets_arr);
            for (int i = 0; i < n; i++) {
                cJSON *entry = cJSON_GetArrayItem(sheets_arr, i);
                cJSON *props = cJSON_GetObjectItemCaseSensitive(entry, "properties");
                cJSON *t = cJSON_GetObjectItemCaseSensitive(props, "title");
                cJSON *id = cJSON_GetObjectItemCaseSensitive(props, "sheetId");
                if (cJSON_IsString(t) && strcmp(t->valuestring, title) == 0 && cJSON_IsNumber(id)) {
                    *out_id = id->valueint;
                    ok = 0;
                    break;
                }
            }
        }
        if (root) cJSON_Delete(root);
    }
    dynbuf_free(&resp);
    return ok;
}

int sheets_worksheet_existe(const SheetsClient *cli, const char *title) {
    int id;
    return obter_sheet_id_por_titulo(cli, title, &id) == 0;
}

static int batch_update(const SheetsClient *cli, const char *requests_json_array) {
    char url[512];
    snprintf(url, sizeof(url), "%s/%s:batchUpdate", SHEETS_BASE, cli->spreadsheet_id);

    char body[2048];
    snprintf(body, sizeof(body), "{\"requests\":%s}", requests_json_array);

    DynBuf resp; dynbuf_init(&resp);
    long code = http_call("POST", url, cli->access_token, body, &resp);
    int ok = (code == 200) ? 0 : -1;
    if (ok != 0) fprintf(stderr, "❌ batchUpdate retornou HTTP %ld: %s\n", code, resp.data);
    dynbuf_free(&resp);
    return ok;
}

int sheets_add_worksheet(const SheetsClient *cli, const char *title, int rows, int cols) {
    char req[1024];
    snprintf(req, sizeof(req),
        "[{\"addSheet\":{\"properties\":{\"title\":\"%s\","
        "\"gridProperties\":{\"rowCount\":%d,\"columnCount\":%d}}}}]",
        title, rows, cols);
    return batch_update(cli, req);
}

int sheets_delete_worksheet_by_title(const SheetsClient *cli, const char *title) {
    int id;
    if (obter_sheet_id_por_titulo(cli, title, &id) != 0) return -1; /* não existe — ok, ignora (mesmo comportamento do try/except do Python) */
    char req[256];
    snprintf(req, sizeof(req), "[{\"deleteSheet\":{\"sheetId\":%d}}]", id);
    return batch_update(cli, req);
}

int sheets_rename_worksheet(const SheetsClient *cli, const char *old_title, const char *new_title) {
    int id;
    if (obter_sheet_id_por_titulo(cli, old_title, &id) != 0) return -1;
    char req[512];
    snprintf(req, sizeof(req),
        "[{\"updateSheetProperties\":{\"properties\":{\"sheetId\":%d,\"title\":\"%s\"},\"fields\":\"title\"}}]",
        id, new_title);
    return batch_update(cli, req);
}
