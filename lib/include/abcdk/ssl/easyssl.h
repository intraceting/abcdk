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

__BEGIN_DECLS

/** 简单的SSL通讯。 */
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
 * 设置关联句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd,int writer);

/**
 * 获取关联句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_easyssl_get_fd(abcdk_easyssl_t *ctx,int writer);

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


__END_DECLS

#endif //ABCDK_SSL_EASYSSL_H


