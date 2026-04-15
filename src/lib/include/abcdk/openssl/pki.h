/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_PKI_H
#define ABCDK_OPENSSL_PKI_H

#include "abcdk/util/option.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

EVP_PKEY *abcdk_openssl_pki_generate_pkey(int bits);

ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits);

X509 *abcdk_openssl_pki_issue_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *cn, const char *org, int ca_or_not, abcdk_option_t *opt,
                                   X509 *issuer_cert, EVP_PKEY *issuer_pkey);

__END_DECLS

#endif //ABCDK_OPENSSL_PKI_H