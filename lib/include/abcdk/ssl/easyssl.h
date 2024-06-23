/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SSL_EASYSSL_H
#define ABCDK_SSL_EASYSSL_H

#include "abcdk/util/general.h"
#include "abcdk/util/enigma.h"
#include "abcdk/util/object.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/sha256.h"
#include "abcdk/util/mmap.h"
#include "abcdk/ssl/openssl.h"

__BEGIN_DECLS

/**简单的SSL通讯。 */
typedef struct _abcdk_easyssl abcdk_easyssl_t;

/**
 * 方案。
 */
typedef enum _abcdk_easyssl_scheme
{
    /*ENIGMA加密机。*/
    ABCDK_EASYSSL_SCHEME_ENIGMA = 1,
#define ABCDK_EASYSSL_SCHEME_ENIGMA ABCDK_EASYSSL_SCHEME_ENIGMA

}abcdk_easyssl_scheme_t;


/**
 * 销毁。
 * 
 * @warning 不会销毁被关联的句柄。
*/
void abcdk_easyssl_destroy(abcdk_easyssl_t **ctx);

/**
 * 创建。
 * 
 * @param [in] key 密钥。
 * @param [in] size 密钥长度(字节)。
 * @param [in] scheme 方案。
 * @param [in] salt 盐长度。<= 256 有效。
 * 
*/
abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *key,size_t size,uint32_t scheme,size_t salt);

/**
 * 创建。
 * 
 * @param [in] file 密钥文件。
 * @param [in] scheme 方案。
 * @param [in] salt 盐长度。<= 256 有效。
 * 
*/
abcdk_easyssl_t *abcdk_easyssl_create_from_file(const char *file,uint32_t scheme,size_t salt);

/**
 * 设置关联句柄。
 * 
 * @param [in] flag 标志。0 读写，1 读，2 写。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd,int flag);

/**
 * 获取关联句柄。
 * 
 * @return >=0 成功(旧的句柄)，< 0  失败(未设置或读写句柄不一致)。
*/
int abcdk_easyssl_get_fd(abcdk_easyssl_t *ctx,int flag);

/**
 * 发送数据。
 * 
 * @warning 重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已经发送的长度，= 0 连接已经关闭或断开，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_easyssl_write(abcdk_easyssl_t *ctx,const void *data,size_t size);

/**
 * 接收数据。
 * 
 * @return > 0 已接收的长度，= 0 连接已经关闭或断开(缓存未清空前不会返回此值)，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_easyssl_read(abcdk_easyssl_t *ctx,void *data,size_t size);

#ifdef HEADER_BIO_H

#if OPENSSL_VERSION_NUMBER < 0x10100000L
void *BIO_get_data(BIO* bio);
void BIO_set_data(BIO* bio,void *ptr);
#endif //OPENSSL_VERSION_NUMBER < 0x10100000L

/**
 * 销毁。
 * 
 */
void abcdk_easyssl2BIO_destroy(BIO **ctx);

/**
 * 创建兼容的BIO环境。
*/
BIO *abcdk_easyssl2BIO_create_from_file(const char *file,uint32_t scheme,size_t salt);

#endif //HEADER_BIO_H

__END_DECLS

#endif //ABCDK_SSL_EASYSSL_H


