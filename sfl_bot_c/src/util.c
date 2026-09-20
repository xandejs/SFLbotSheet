#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <openssl/evp.h>

void dynbuf_init(DynBuf *b) {
    b->cap = 4096;
    b->len = 0;
    b->data = malloc(b->cap);
    b->data[0] = '\0';
}

void dynbuf_append(DynBuf *b, const char *src, size_t n) {
    if (b->len + n + 1 > b->cap) {
        while (b->len + n + 1 > b->cap) b->cap *= 2;
        b->data = realloc(b->data, b->cap);
    }
    memcpy(b->data + b->len, src, n);
    b->len += n;
    b->data[b->len] = '\0';
}

void dynbuf_free(DynBuf *b) {
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

/* Réplica de limpar_numero(texto) do bot_4_24.py:
 *   - "" ou NULL -> 0.0
 *   - se tem '.' E ',' -> assume '.' é separador de milhar (padrão BR:
 *     "1.234,56"), remove os pontos e troca ',' por '.'
 *   - senão, só troca ',' por '.' (aceita tanto "1234,56" quanto "1234.56")
 *   - múltiplos '.' residuais -> mantém só o último como decimal
 *   - qualquer falha de parse -> 0.0 (nunca propaga erro, igual ao original)
 */
double limpar_numero(const char *texto) {
    if (texto == NULL || texto[0] == '\0') return 0.0;

    char buf[128];
    size_t n = strlen(texto);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, texto, n);
    buf[n] = '\0';

    /* trim */
    char *start = buf;
    while (*start && isspace((unsigned char)*start)) start++;
    char *end = start + strlen(start) - 1;
    while (end > start && isspace((unsigned char)*end)) *end-- = '\0';

    int tem_ponto = strchr(start, '.') != NULL;
    int tem_virgula = strchr(start, ',') != NULL;

    char out[128];
    size_t oi = 0;
    for (char *p = start; *p && oi < sizeof(out) - 1; p++) {
        if (*p == '.' && tem_ponto && tem_virgula) {
            continue; /* ponto é separador de milhar, descarta */
        } else if (*p == ',') {
            out[oi++] = '.';
        } else {
            out[oi++] = *p;
        }
    }
    out[oi] = '\0';

    /* Se sobrou mais de um '.', mantém só o último como decimal
     * (equivalente ao "".join(partes[:-1]) + "." + partes[-1]" do Python) */
    int n_pontos = 0;
    for (size_t i = 0; i < oi; i++) if (out[i] == '.') n_pontos++;
    if (n_pontos > 1) {
        char fixed[128];
        size_t fi = 0;
        int visto = 0;
        for (int i = (int)oi - 1; i >= 0; i--) {
            if (out[i] == '.') {
                visto++;
                if (visto == 1) { fixed[fi++] = '.'; continue; }
                else continue; /* descarta pontos extras à esquerda */
            }
            fixed[fi++] = out[i];
        }
        fixed[fi] = '\0';
        /* está invertido, reverte */
        for (size_t i = 0; i < fi / 2; i++) {
            char t = fixed[i];
            fixed[i] = fixed[fi - 1 - i];
            fixed[fi - 1 - i] = t;
        }
        strcpy(out, fixed);
    }

    char *endptr = NULL;
    double v = strtod(out, &endptr);
    if (endptr == out) return 0.0; /* nada foi convertido -> parse falhou */
    return v;
}

char *base64url_encode(const unsigned char *input, size_t len) {
    /* EVP_EncodeBlock não insere quebras de linha (diferente de PEM_write),
     * então dá pra converter direto para o alfabeto URL-safe. */
    size_t out_cap = 4 * ((len + 2) / 3) + 1;
    unsigned char *tmp = malloc(out_cap);
    int written = EVP_EncodeBlock(tmp, input, (int)len);

    char *result = malloc((size_t)written + 1);
    int ri = 0;
    for (int i = 0; i < written; i++) {
        unsigned char c = tmp[i];
        if (c == '=') continue;          /* remove padding */
        else if (c == '+') result[ri++] = '-';
        else if (c == '/') result[ri++] = '_';
        else result[ri++] = (char)c;
    }
    result[ri] = '\0';
    free(tmp);
    return result;
}

long agora_unix(void) {
    return (long)time(NULL);
}
