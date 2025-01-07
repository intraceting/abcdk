/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_DARKNET_H
#define ABCDK_OPENSSL_DARKNET_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/sha256.h"
#include "abcdk/util/mmap.h"
#include "abcdk/util/hash.h"
#include "abcdk/util/receiver.h"
#include "abcdk/openssl/cipher.h"

__BEGIN_DECLS

#ifdef OPENSSL_VERSION_NUMBER

/**简单的安全套接字。*/
typedef struct _abcdk_openssl_darknet abcdk_openssl_darknet_t;

/**
 * 销毁。
 * 
 * @warning 不会销毁被关联的句柄。
*/
void abcdk_openssl_darknet_destroy(abcdk_openssl_darknet_t **ctx);

/**
 * 创建。
 * 
 * @note 内部会复制RSA环境，不会影响外部的使用。
 * 
 * @param [in] use_pubkey 使用公钥。!0 是，0 否。
 * 
*/
abcdk_openssl_darknet_t *abcdk_openssl_darknet_create(RSA *rsa_ctx, int use_pubkey);

/**创建。*/
abcdk_openssl_darknet_t *abcdk_openssl_darknet_create_from_file(const char *rsa_file, int pubkey);

/**
 * 设置关联句柄。
 * 
 * @param [in] flag 标志。0 读写，1 读，2 写。
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_openssl_darknet_set_fd(abcdk_openssl_darknet_t *ctx,int fd,int flag);

/**
 * 获取关联句柄。
 * 
 * @return >=0 成功(旧的句柄)，< 0  失败(未设置或读写句柄不一致)。
*/
int abcdk_openssl_darknet_get_fd(abcdk_openssl_darknet_t *ctx,int flag);

/**
 * 发送数据。
 * 
 * @warning 重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已经发送的长度，= 0 连接已经关闭或断开，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_openssl_darknet_write(abcdk_openssl_darknet_t *ctx,const void *data,size_t size);

/**
 * 接收数据。
 * 
 * @return > 0 已接收的长度，= 0 连接已经关闭或断开(缓存未清空前不会返回此值)，< 0 失败(非阻塞管道有效)。
*/
ssize_t abcdk_openssl_darknet_read(abcdk_openssl_darknet_t *ctx,void *data,size_t size);

#endif //OPENSSL_VERSION_NUMBER

__END_DECLS


#endif //ABCDK_OPENSSL_DARKNET_H