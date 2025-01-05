/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_OPENSSL_BIO_H
#define ABCDK_OPENSSL_BIO_H

#include "abcdk/openssl/darknet.h"

__BEGIN_DECLS

#ifdef HEADER_BIO_H

/**销毁。*/
void abcdk_openssl_BIO_destroy(BIO **bio);

/**
 * 创建兼容Darknet的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_Darknet(RSA *rsa_ctx, int use_pubkey);

/**
 * 创建兼容Darknet的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_Darknet_form_file(const char *rsa_file, int pubkey);


#endif //HEADER_BIO_H

__END_DECLS

#endif //ABCDK_OPENSSL_BIO_H