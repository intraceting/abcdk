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

#ifdef HAVE_OPENSSL
static int _test_pem_password_cb(char *buf, int size, int rwflag, void *userdata)
{
   int c = scanf("%s",buf);

   assert(c > 0);

   return strlen(buf);
}

#endif //#ifdef HAVE_OPENSSL

int abcdk_test_pem(abcdk_option_t *args)
{
#ifdef HAVE_OPENSSL
    int iserver = abcdk_option_get_int(args,"--server",0,0);
    const char *ca_file_p = abcdk_option_get(args, "--ca-file", 0, NULL);
    const char *ca_path_p = abcdk_option_get(args, "--ca-path", 0, NULL);
    const char *crt_file_p = abcdk_option_get(args, "--crt-file", 0, NULL);
    const char *key_file_p = abcdk_option_get(args, "--key-file", 0, NULL);

    abcdk_object_t *passwd = NULL;
    EVP_PKEY *key = abcdk_openssl_key_load(key_file_p,&passwd);
    assert(key != NULL);


    fprintf(stderr,"passwd: %s\n",passwd?passwd->pstrs[0]:"no passwd");

    SSL_CTX *ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(iserver,ca_file_p,ca_path_p,crt_file_p,key_file_p,passwd?passwd->pstrs[0]:NULL);
    assert(ssl_ctx != NULL);

    abcdk_openssl_ssl_ctx_free(&ssl_ctx);

    abcdk_object_unref(&passwd);
    EVP_PKEY_free(key);

#endif //#ifdef HAVE_OPENSSL
    return 0;
}