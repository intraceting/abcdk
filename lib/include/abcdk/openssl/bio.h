/*
 * This file is part of ABCDK.
 * 
 * MIT License
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
 * 创建兼容Marknet的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_Darknet(int scheme, const uint8_t *key,size_t size);

/**
 * 创建兼容Marknet的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_Darknet_form_file(int scheme,const char *file);


#endif //HEADER_BIO_H

__END_DECLS

#endif //ABCDK_OPENSSL_BIO_H