#ifndef SFL_UTIL_H
#define SFL_UTIL_H

#include <stddef.h>

/* Buffer dinâmico simples para respostas HTTP (equivalente ao que o
 * requests/gspread fazem por baixo dos panos em Python). */
typedef struct {
    char *data;
    size_t len;
    size_t cap;
} DynBuf;

void dynbuf_init(DynBuf *b);
void dynbuf_append(DynBuf *b, const char *src, size_t n);
void dynbuf_free(DynBuf *b);

/* Equivalente a limpar_numero() do bot_4_24.py: trata separador decimal
 * BR (vírgula) vs. US (ponto), e string vazia/inválida vira 0.0 — mesma
 * semântica "nunca lança exceção" do original. */
double limpar_numero(const char *texto);

/* base64url sem padding — necessário para montar o JWT do Service Account
 * (header.payload assinados em RS256, RFC 7515). OpenSSL não tem uma
 * função pronta para "base64url sem padding", só base64 padrão, então
 * fazemos a conversão de charset (+/ -> -_) e removemos o '=' à mão. */
char *base64url_encode(const unsigned char *input, size_t len);

/* Timestamp UNIX atual, usado para exp/iat do JWT. */
long agora_unix(void);

#endif
