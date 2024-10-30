/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_OPENSSL_BIO_H
#define ABCDK_OPENSSL_BIO_H

#include "abcdk/ssl/maskssl.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

#ifdef HEADER_BIO_H

/**
 * 设置关联句柄。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_openssl_BIO_set_fd(BIO *bio, int fd);

/**
 * 获取关联句柄。
 */
int abcdk_openssl_BIO_get_fd(BIO *bio);

/**销毁。*/
void abcdk_openssl_BIO_destroy(BIO **bio);

/**
 * 创建兼容MaskSSL的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_MaskSSL(int scheme, const uint8_t *key,size_t size);

/**
 * 创建兼容MaskSSL的BIO环境。
*/
BIO *abcdk_openssl_BIO_s_MaskSSL_form_file(int scheme,const char *file);


#endif //HEADER_BIO_H

__END_DECLS

#endif //ABCDK_OPENSSL_BIO_H