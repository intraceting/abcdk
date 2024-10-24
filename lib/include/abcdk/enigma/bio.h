/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_ENIGMA_BIO_H
#define ABCDK_ENIGMA_BIO_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/object.h"
#include "abcdk/enigma/ssl.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

#ifdef HEADER_BIO_H

/**
 * 设置关联句柄。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_enigma_BIO_set_fd(BIO *bio, int fd);

/**
 * 获取关联句柄。
 */
int abcdk_enigma_BIO_get_fd(BIO *bio);

/**销毁。*/
void abcdk_enigma_BIO_destroy(BIO **bio);

/**
 * 创建兼容Enigma的SSL的BIO环境。
*/
BIO *abcdk_enigma_BIO_s_SSL(const char *file);


#endif //HEADER_BIO_H

__END_DECLS

#endif //ABCDK_ENIGMA_BIO_H