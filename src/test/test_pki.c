/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

static void _make_cert(EVP_PKEY **pkey, X509 **cert, const char *cn, const char *org, int ca_or_not, X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
    *pkey = abcdk_openssl_pki_generate_pkey(4096);

    ASN1_INTEGER *serial = abcdk_openssl_pki_generate_serial(120);
    assert(serial != NULL);

    abcdk_option_t *opt = abcdk_option_alloc("--");

    abcdk_option_fset(opt, "--ca", "%d", ca_or_not);
    abcdk_option_set(opt, "--pathlen", "3");

    *cert = abcdk_openssl_pki_generate_cert(*pkey, serial, cn, org, opt, issuer_cert, issuer_pkey);

    abcdk_option_free(&opt);
    abcdk_openssl_ai_free(&serial);

    char pkey_file[100] = {0};
    char cert_file[100] = {0};

    sprintf(pkey_file, "/tmp/%s-%s.prikey.pem", cn, org);
    sprintf(cert_file, "/tmp/%s-%s.cert.pem", cn, org);

    abcdk_object_t *pkey_txt = abcdk_openssl_pki_export_pkey(*pkey, 0, "1234", 4);
    abcdk_dump(pkey_file, pkey_txt->pstrs[0], pkey_txt->sizes[0]);
    abcdk_object_unref(&pkey_txt);

    abcdk_object_t *cert_txt = abcdk_openssl_pki_export_cert(*cert);
    abcdk_dump(cert_file, cert_txt->pstrs[0], cert_txt->sizes[0]);
    abcdk_object_unref(&cert_txt);
}

static void _save_crl(X509_CRL *crl,const char *file, X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
    abcdk_option_t *opt = abcdk_option_alloc("--");

    abcdk_option_fset(opt, "--last-update-days", "%d", 0);
    abcdk_option_fset(opt, "--next-update-days", "%d", 90);

    abcdk_openssl_pki_update_crl(crl,opt,issuer_cert,issuer_pkey);
    abcdk_option_free(&opt);

    abcdk_object_t *crl_txt = abcdk_openssl_pki_export_crl(crl);
    abcdk_dump(file, crl_txt->pstrs[0], crl_txt->sizes[0]);
    abcdk_object_unref(&crl_txt);
}

int abcdk_test_pki(abcdk_option_t *args)
{
    EVP_PKEY *root_key = NULL;
    X509 *root_cert = NULL;

    _make_cert(&root_key,&root_cert,"haha","haha",1,NULL,NULL);

    EVP_PKEY *int_key = NULL;
    X509 *int_cert = NULL;

    _make_cert(&int_key,&int_cert,"hehe","hehe",1,root_cert,root_key);

    EVP_PKEY *leaf_key = NULL;
    X509 *leaf_cert = NULL;

    _make_cert(&leaf_key,&leaf_cert,"dada","dada",0,int_cert,int_key);

    EVP_PKEY *leaf2_key = NULL;
    X509 *leaf2_cert = NULL;

    _make_cert(&leaf2_key,&leaf2_cert,"dada2","dada2",0,int_cert,int_key);

    EVP_PKEY *leaf3_key = NULL;
    X509 *leaf3_cert = NULL;

    _make_cert(&leaf3_key,&leaf3_cert,"dada3","dada3",0,int_cert,int_key);

    X509_CRL *crl = NULL;

 ///   abcdk_openssl_pki_revoke_cert(&crl,leaf3_cert,CRL_REASON_KEY_COMPROMISE,root_cert,root_key);
    int chk = abcdk_openssl_pki_revoke_cert(&crl,leaf3_cert,CRL_REASON_KEY_COMPROMISE,int_cert,int_key);
    assert(chk == 0);
    chk = abcdk_openssl_pki_revoke_cert(&crl,leaf3_cert,CRL_REASON_KEY_COMPROMISE,int_cert,int_key);
    assert(chk == -2);

    _save_crl(crl,"/tmp/crl-01.pem",int_cert,int_key);

  //  abcdk_openssl_pki_revoke_cert(&crl,leaf2_cert,CRL_REASON_KEY_COMPROMISE,root_cert,root_key);

     chk = abcdk_openssl_pki_revoke_cert(&crl,leaf2_cert,CRL_REASON_KEY_COMPROMISE,int_cert,int_key);
     assert(chk == 0);
     chk = abcdk_openssl_pki_revoke_cert(&crl,leaf2_cert,CRL_REASON_KEY_COMPROMISE,int_cert,int_key);
     assert(chk == -2);

    _save_crl(crl,"/tmp/crl-02.pem",int_cert,int_key);

    abcdk_openssl_x509_CRL_free(&crl);


    X509_free(root_cert);
    abcdk_openssl_evp_pkey_free(&root_key);

    X509_free(int_cert);
    abcdk_openssl_evp_pkey_free(&int_key);

    X509_free(leaf_cert);
    abcdk_openssl_evp_pkey_free(&leaf_key);

    X509_free(leaf2_cert);
    abcdk_openssl_evp_pkey_free(&leaf2_key);
}