/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_OPENSSL_H
#define ABCDK_OPENSSL_OPENSSL_H

#include "abcdk/util/general.h"


#ifdef HAVE_OPENSSL
#include <openssl/opensslconf.h>
#include <openssl/opensslv.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/x509_vfy.h>

#if !defined(OPENSSL_NO_SHA) && (!defined(OPENSSL_NO_SHA0) || !defined(OPENSSL_NO_SHA1))
#include <openssl/sha.h>
#endif //!defined(OPENSSL_NO_SHA) && (!defined(OPENSSL_NO_SHA0) || !defined(OPENSSL_NO_SHA1))

#ifndef OPENSSL_NO_RSA
#include <openssl/rsa.h>
#endif //OPENSSL_NO_RSA

#ifndef OPENSSL_NO_AES
#include <openssl/aes.h>
#endif //OPENSSL_NO_AES

#ifndef OPENSSL_NO_HMAC
#include <openssl/hmac.h>
#endif //OPENSSL_NO_HMAC

#endif //HAVE_OPENSSL

/**/
#ifndef OPENSSL_VERSION_NUMBER
#define STACK_OF(type) struct stack_st_##type
typedef struct stack_st_X509 stack_st_X509;
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct bio_st BIO;
typedef struct bignum_st BIGNUM;
typedef struct evp_pkey_st EVP_PKEY;
typedef struct evp_cipher_st EVP_CIPHER;
typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_md_st EVP_MD;
typedef struct x509_st X509;
typedef struct X509_crl_st X509_CRL;
typedef struct x509_store_ctx_st X509_STORE_CTX;
typedef struct x509_store_st X509_STORE;
typedef struct rsa_st RSA;
typedef struct hmac_ctx_st HMAC_CTX;
#define SSL_read(f,b,s) 0
#define SSL_write(f,b,s) 0
#define BIO_read(f,b,s) 0
#define BIO_write(f,b,s) 0
#endif //OPENSSL_VERSION_NUMBER

#endif //ABCDK_OPENSSL_OPENSSL_H