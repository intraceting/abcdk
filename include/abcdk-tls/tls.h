/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_TLS_TLS_H
#define ABCDK_TLS_TLS_H

#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/socket.h"
#include "abcdk-util/epollex.h"
#include "abcdk-util/openssl.h"

__BEGIN_DECLS

#ifndef HEADER_SSL_H
typedef struct ssl_ctx_st SSL_CTX;
#endif //HEADER_SSL_H

/**/
#define ABCDK_TLS_EVENT_CONNECT 0x0001
#define ABCDK_TLS_EVENT_INPUT 0x0010
#define ABCDK_TLS_EVENT_OUTPUT 0x0020
#define ABCDK_TLS_EVENT_CLOSE 0x1000

int abcdk_tls_set_timeout(uint64_t tls, time_t timeout);

int abcdk_tls_get_address(uint64_t tls, abcdk_sockaddr_t addr[2]);

ssize_t abcdk_tls_read(uint64_t tls, void *buf, size_t size);

int abcdk_tls_read_watch(uint64_t tls);

ssize_t abcdk_tls_write(uint64_t tls, void *buf, size_t size);

int abcdk_tls_write_watch(uint64_t tls);

void abcdk_tls_shutdown(uint64_t tls);

//void abcdk_tls_close(uint64_t tls);

void abcdk_tls_loop(void (*event_cb)(uint64_t tls, uint32_t events, void *opaque));

int abcdk_tls_listen(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque);

int abcdk_tls_connect(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque);

__END_DECLS

#endif //ABCDK_TLS_TLS_H