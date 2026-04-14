/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_PKI_H
#define ABCDK_OPENSSL_PKI_H

#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

EVP_PKEY *abcdk_openssl_pki_generate_pkey(int bits);

ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits);

int abcdk_openssl_pki_add_ext(X509 *cert, int nid, const char *value);

__END_DECLS

#endif //ABCDK_OPENSSL_PKI_H