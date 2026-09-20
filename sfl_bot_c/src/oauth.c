#include "oauth.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cjson/cJSON.h>
#include <curl/curl.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/bio.h>

/* Mesmos escopos usados em bot_4_24.py:
 *   escopo = ["https://spreadsheets.google.com/feeds",
 *             "https://www.googleapis.com/auth/drive"]
 * A API v4 aceita o escopo canônico .../auth/spreadsheets no lugar do
 * legado /feeds — mantemos os dois por segurança de compatibilidade. */
#define ESCOPOS "https://www.googleapis.com/auth/spreadsheets " \
                "https://www.googleapis.com/auth/drive"

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
    DynBuf *b = (DynBuf *)userdata;
    dynbuf_append(b, (const char *)ptr, size * nmemb);
    return size * nmemb;
}

int carregar_service_account(const char *json_path, ServiceAccount *sa) {
    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "❌ Não foi possível abrir %s\n", json_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t lidos = fread(buf, 1, (size_t)sz, f);
    buf[lidos] = '\0';
    fclose(f);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) {
        fprintf(stderr, "❌ credenciais_robo.json inválido (JSON malformado)\n");
        return -1;
    }

    const cJSON *email = cJSON_GetObjectItemCaseSensitive(root, "client_email");
    const cJSON *key    = cJSON_GetObjectItemCaseSensitive(root, "private_key");
    const cJSON *uri     = cJSON_GetObjectItemCaseSensitive(root, "token_uri");

    if (!cJSON_IsString(email) || !cJSON_IsString(key)) {
        fprintf(stderr, "❌ credenciais_robo.json sem client_email/private_key\n");
        cJSON_Delete(root);
        return -1;
    }

    snprintf(sa->client_email, sizeof(sa->client_email), "%s", email->valuestring);
    snprintf(sa->private_key_pem, sizeof(sa->private_key_pem), "%s", key->valuestring);
    snprintf(sa->token_uri, sizeof(sa->token_uri), "%s",
             cJSON_IsString(uri) ? uri->valuestring : "https://oauth2.googleapis.com/token");

    cJSON_Delete(root);
    return 0;
}

/* Assina `data` (header.payload) com a chave RSA privada em PEM, RS256
 * (RSASSA-PKCS1-v1_5 + SHA-256) — exigido pelo Google para JWT bearer
 * assertions. Retorna base64url da assinatura (aloca com malloc). */
static char *assinar_rs256(const char *data, const char *private_key_pem) {
    BIO *bio = BIO_new_mem_buf(private_key_pem, -1);
    EVP_PKEY *pkey = PEM_read_bio_PrivateKey(bio, NULL, NULL, NULL);
    BIO_free(bio);
    if (!pkey) {
        fprintf(stderr, "❌ Falha ao carregar chave privada RSA (PEM inválido)\n");
        return NULL;
    }

    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    unsigned char sig[512];
    size_t sig_len = sizeof(sig);
    char *result = NULL;

    if (EVP_DigestSignInit(mdctx, NULL, EVP_sha256(), NULL, pkey) == 1 &&
        EVP_DigestSignUpdate(mdctx, data, strlen(data)) == 1 &&
        EVP_DigestSignFinal(mdctx, sig, &sig_len) == 1) {
        result = base64url_encode(sig, sig_len);
    } else {
        fprintf(stderr, "❌ Falha ao assinar JWT (EVP_DigestSign)\n");
    }

    EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    return result;
}

/* Monta o JWT completo: base64url(header) + "." + base64url(claims),
 * assina, e retorna "header.claims.assinatura" (aloca com malloc). */
static char *montar_jwt(const ServiceAccount *sa) {
    long iat = agora_unix();
    long exp = iat + 3600; /* Google exige exp <= iat + 1h */

    const char *header_json = "{\"alg\":\"RS256\",\"typ\":\"JWT\"}";
    char claims_json[1024];
    snprintf(claims_json, sizeof(claims_json),
        "{\"iss\":\"%s\",\"scope\":\"%s\",\"aud\":\"%s\",\"exp\":%ld,\"iat\":%ld}",
        sa->client_email, ESCOPOS, sa->token_uri, exp, iat);

    char *h_b64 = base64url_encode((const unsigned char *)header_json, strlen(header_json));
    char *c_b64 = base64url_encode((const unsigned char *)claims_json, strlen(claims_json));

    char signing_input[2048];
    snprintf(signing_input, sizeof(signing_input), "%s.%s", h_b64, c_b64);

    char *sig_b64 = assinar_rs256(signing_input, sa->private_key_pem);
    free(h_b64);
    free(c_b64);
    if (!sig_b64) return NULL;

    char *jwt = malloc(strlen(signing_input) + strlen(sig_b64) + 2);
    sprintf(jwt, "%s.%s", signing_input, sig_b64);
    free(sig_b64);
    return jwt;
}

int obter_access_token(const ServiceAccount *sa, char *out_token, size_t out_size) {
    char *jwt = montar_jwt(sa);
    if (!jwt) return -1;

    CURL *curl = curl_easy_init();
    if (!curl) { free(jwt); return -1; }

    char post_fields[3072];
    snprintf(post_fields, sizeof(post_fields),
        "grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=%s", jwt);
    free(jwt);

    DynBuf resp;
    dynbuf_init(&resp);

    curl_easy_setopt(curl, CURLOPT_URL, sa->token_uri);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_fields);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    int ok = -1;
    if (res == CURLE_OK && http_code == 200) {
        cJSON *root = cJSON_Parse(resp.data);
        const cJSON *tok = root ? cJSON_GetObjectItemCaseSensitive(root, "access_token") : NULL;
        if (cJSON_IsString(tok)) {
            snprintf(out_token, out_size, "%s", tok->valuestring);
            ok = 0;
        }
        if (root) cJSON_Delete(root);
    } else {
        fprintf(stderr, "❌ Token endpoint retornou HTTP %ld: %s\n", http_code, resp.data);
    }

    dynbuf_free(&resp);
    return ok;
}
