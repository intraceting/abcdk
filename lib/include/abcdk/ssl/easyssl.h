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

__BEGIN_DECLS

/** 简单的SSL通讯。 */
typedef struct _abcdk_easyssl abcdk_easyssl_t;

/**
 * 方案。
 */
typedef enum _abcdk_easyssl_scheme
{
    /*ENIGMA加密机，标准的。*/
    ABCDK_EASYSSL_SCHEME_ENIGMA_NORMAL = 1,
#define ABCDK_EASYSSL_SCHEME_ENIGMA_NORMAL ABCDK_EASYSSL_SCHEME_ENIGMA_NORMAL

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
 * @param [in] flag 标志。
 * 
*/
abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *key,size_t size,uint32_t scheme);

/**
 * 设置关联句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd);

/**
 * 获取关联句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_easyssl_get_fd(abcdk_easyssl_t *ctx);

/**
 * 发送数据。
 * 
 * @return > 0 已经发送的长度(包括缓存未发送完成的)，= 0 连接已经关闭或断开，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_easyssl_send(abcdk_easyssl_t *ctx,const void *data,size_t size);

/**
 * 接收数据。
 * 
 * @return > 0 已接收的长度，= 0 连接已经关闭或断开(缓存未清空前不会返回此值)，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_easyssl_recv(abcdk_easyssl_t *ctx,void *data,size_t size);


__END_DECLS

#endif //ABCDK_SSL_EASYSSL_H


