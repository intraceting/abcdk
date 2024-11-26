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
    const char *key_file_p = abcdk_option_get(args, "--key-file", 0, NULL);

    FILE *fp = fopen(key_file_p,"r");
    assert(fp != NULL);

    EVP_PKEY *key = PEM_read_PrivateKey(fp,NULL,_test_pem_password_cb,NULL);
    assert(key != NULL);

    EVP_PKEY_free(key);

    fclose(fp);
#endif //#ifdef HAVE_OPENSSL
    return 0;
}