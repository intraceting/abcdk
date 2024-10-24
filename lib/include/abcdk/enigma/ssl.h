/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_ENIGMA_SSL_H
#define ABCDK_ENIGMA_SSL_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/sha256.h"
#include "abcdk/util/mmap.h"
#include "abcdk/enigma/enigma.h"

__BEGIN_DECLS

/**基于enigma的SSL通讯。 */
typedef struct _abcdk_enigma_ssl abcdk_enigma_ssl_t;


/**
 * 销毁。
 * 
 * @warning 不会销毁被关联的句柄。
*/
void abcdk_enigma_ssl_destroy(abcdk_enigma_ssl_t **ctx);

/** 创建。*/
abcdk_enigma_ssl_t *abcdk_enigma_ssl_create(const uint8_t *key,size_t size);

/** 创建。*/
abcdk_enigma_ssl_t *abcdk_enigma_ssl_create_from_file(const char *file);

/**
 * 设置关联句柄。
 * 
 * @param [in] flag 标志。0 读写，1 读，2 写。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_enigma_ssl_set_fd(abcdk_enigma_ssl_t *ctx,int fd,int flag);

/**
 * 获取关联句柄。
 * 
 * @return >=0 成功(旧的句柄)，< 0  失败(未设置或读写句柄不一致)。
*/
int abcdk_enigma_ssl_get_fd(abcdk_enigma_ssl_t *ctx,int flag);

/**
 * 发送数据。
 * 
 * @warning 重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已经发送的长度，= 0 连接已经关闭或断开，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_enigma_ssl_write(abcdk_enigma_ssl_t *ctx,const void *data,size_t size);

/**
 * 接收数据。
 * 
 * @return > 0 已接收的长度，= 0 连接已经关闭或断开(缓存未清空前不会返回此值)，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_enigma_ssl_read(abcdk_enigma_ssl_t *ctx,void *data,size_t size);


__END_DECLS

#endif //ABCDK_ENIGMA_SSL_H


