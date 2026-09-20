#ifndef SFL_OAUTH_H
#define SFL_OAUTH_H

#include <stddef.h>

typedef struct {
    char client_email[256];
    char private_key_pem[4096]; /* chave RSA privada, formato PEM, do JSON de credenciais */
    char token_uri[256];
} ServiceAccount;

/* Lê o credenciais_robo.json (mesmo arquivo usado pelo ServiceAccountCredentials
 * do Python) e extrai client_email / private_key / token_uri. */
int carregar_service_account(const char *json_path, ServiceAccount *sa);

/* Monta o JWT assinado (RS256) com os escopos do Sheets+Drive, troca por um
 * access_token via grant_type=jwt-bearer (RFC 7523), e retorna o token em
 * out_token (buffer fornecido pelo chamador, min. 2048 bytes).
 * Retorna 0 em sucesso, -1 em erro. */
int obter_access_token(const ServiceAccount *sa, char *out_token, size_t out_size);

#endif
