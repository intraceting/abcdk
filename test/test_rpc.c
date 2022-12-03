/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

void test_rpc_request_cb(abcdk_comm_node_t *rpc, uint64_t mid, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(rpc, sockname_str, peername_str);

    //  printf("Server(%s -> %s): ", sockname_str, peername_str);

    uint64_t a = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6);
    uint64_t b = atoll((char *)data);

    //       printf("%lu-%lu=%lu",a,b,a-b);

    //    usleep(rand()%10000+1000);

    abcdk_rpc_request(rpc, data, len, NULL, 1);
    abcdk_rpc_response(rpc, mid, data, len);
    

    //   printf("\n");
}

void test_rpc_request2_cb(abcdk_comm_node_t *rpc, uint64_t mid, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(rpc, sockname_str, peername_str);

    //  printf("Client(%s -> %s): ", sockname_str, peername_str);

    //   printf(" %s\n",(char*)data);

    //  printf("\n");
}

#ifdef HAVE_OPENSSL
int _abcdk_test_verify_callback(int ok, X509_STORE_CTX *x509_store)
{

    char *subject, *issuer;
    int err, depth;
    X509 *cert;
    X509_NAME *sname, *iname;

    cert = X509_STORE_CTX_get_current_cert(x509_store);
    err = X509_STORE_CTX_get_error(x509_store);
    depth = X509_STORE_CTX_get_error_depth(x509_store);

    sname = X509_get_subject_name(cert);

    if (sname)
    {
        subject = X509_NAME_oneline(sname, NULL, 0);
        if (subject == NULL)
        {
            abcdk_log_printf(LOG_WARNING, "X509_NAME_oneline() failed");
        }
    }
    else
    {
        subject = NULL;
    }

    iname = X509_get_issuer_name(cert);

    if (iname)
    {
        issuer = X509_NAME_oneline(iname, NULL, 0);
        if (issuer == NULL)
        {
            abcdk_log_printf(LOG_WARNING, "X509_NAME_oneline() failed");
        }
    }
    else
    {
        issuer = NULL;
    }

    abcdk_log_printf(LOG_INFO,
                     "verify:%d, error:%d, depth:%d, "
                     "subject:\"%s\", issuer:\"%s\"",
                     ok, err, depth,
                     subject ? subject : "(none)",
                     issuer ? issuer : "(none)");

    if (subject)
    {
        OPENSSL_free(subject);
    }

    if (issuer)
    {
        OPENSSL_free(issuer);
    }

    return 1;
}
#endif

int abcdk_test_rpc(abcdk_tree_t *args)
{
    signal(SIGPIPE, NULL);

    abcdk_comm_t *ctx = abcdk_comm_start(-1,-1);

    SSL_CTX *server_ssl_ctx = NULL;
    SSL_CTX *client_ssl_ctx[4] = {NULL};

#ifdef HAVE_OPENSSL

    const char *cafile = abcdk_option_get(args, "--ca-file", 0, NULL);
    const char *capath = abcdk_option_get(args, "--ca-path", 0, NULL);
    const char *cert = abcdk_option_get(args, "--cert-file", 0, NULL);
    const char *key = abcdk_option_get(args, "--key-file", 0, NULL);
    const char *cert2 = abcdk_option_get(args, "--cert2-file", 0, NULL);
    const char *key2 = abcdk_option_get(args, "--key2-file", 0, NULL);

    if (cert && key)
    {
        server_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, cafile, capath, capath?2:0);

        abcdk_openssl_ssl_ctx_load_crt(server_ssl_ctx, cert,key,NULL);

        //  SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, _abcdk_test_verify_callback);

        SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, _abcdk_test_verify_callback);
        // SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);

    }

    if (cert2 && key2)
    {
        for (int i = 0; i < 4; i++)
        {
            client_ssl_ctx[i] = abcdk_openssl_ssl_ctx_alloc(0, cafile, capath, capath?2:0);

            abcdk_openssl_ssl_ctx_load_crt(client_ssl_ctx[i], cert2, key2, NULL);

            //      SSL_CTX_set_verify(client_ssl_ctx[i], SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, _abcdk_test_verify_callback);
            //   SSL_CTX_set_verify(client_ssl_ctx[i], SSL_VERIFY_PEER, _abcdk_test_verify_callback);
            //   SSL_CTX_set_verify(client_ssl_ctx[i], SSL_VERIFY_PEER, NULL);
        }
    }

#endif // HAVE_OPENSSL

    const char *sunpath = "/tmp/test_rpc.sock";
    unlink(sunpath);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_t addr2 = {0};

    const char *listen_p = abcdk_option_get(args, "--listen", 0, "0.0.0.0:12345");
    abcdk_sockaddr_from_string(&addr, listen_p, 0);
    // addr.family = AF_UNIX;
    // strncpy(addr.addr_un.sun_path,sunpath,108);

    abcdk_comm_node_t *rpc_listen = abcdk_rpc_alloc(ctx, 3333,0);

    abcdk_rpc_callback_t listencb = {NULL,NULL, test_rpc_request_cb};
    abcdk_rpc_listen(rpc_listen, server_ssl_ctx, &addr, &listencb);

    const char *connect_p = abcdk_option_get(args, "--connect", 0, "127.0.0.1:12345");
    abcdk_sockaddr_from_string(&addr2, connect_p, 0);
    // addr2.family = AF_UNIX;
    // strncpy(addr2.addr_un.sun_path,sunpath,108);

    for (int j = 0; j < 10; j++)
    {

        int nn = 4;
        abcdk_comm_node_t *rpc_client[40] = {NULL};
        for (int i = 0; i < nn; i++)
        {
            rpc_client[i] = abcdk_rpc_alloc(ctx, 3333,0);
            abcdk_rpc_callback_t clientcb = {NULL, NULL, test_rpc_request2_cb};
            abcdk_rpc_connect(rpc_client[i], client_ssl_ctx[i], &addr2, &clientcb);
        }

        //  sleep(10);

        uint64_t d = 0, s = 0;
        s = abcdk_clock(d, &d);

#pragma omp parallel for num_threads(nn)
        for (int i = 0; i < 10000; i++)
        {
#ifdef _OPENMP
            omp_get_thread_num();
#endif
            int id = i%nn;

            uint64_t d = 0, s = 0;
            s = abcdk_clock(d, &d);

            int len = 1000;
            char *req = (char *)abcdk_heap_alloc(len);
            abcdk_message_t *rsp = NULL;

            sprintf(req, "%lu", abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6));

            abcdk_rpc_request(rpc_client[id], req, len, &rsp, 1);

            if (rsp)
            {

                // printf("%d=%s\n",i,(char*)abcdk_message_data(rsp));

                abcdk_message_unref(&rsp);
            }
            else
            {
                printf("Pipe(%d) %s timeout\n", id, req);
            }

            abcdk_heap_free(req);

            s = abcdk_clock(d, &d);

            //  printf("[%d]:s = %lu,d = %lu\n",i,s,d);
        }

        s = abcdk_clock(d, &d);

        printf("s = %lu,d = %lu\n", s, d);

        for (int i = 0; i < nn; i++)
        {
            abcdk_comm_set_timeout(rpc_client[i],1);
            abcdk_comm_unref(&rpc_client[i]);
        }
    }

    //   abcdk_rpc_set_timeout(rpc_listen,1);

    //  abcdk_rpc_unref(&rpc_listen);
    while (getchar() != 'Q')
        ;

    abcdk_comm_unref(&rpc_listen);
    abcdk_comm_stop(&ctx);

    return 0;
}