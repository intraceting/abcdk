/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_REDIS_UTIL_H
#define ABCDK_REDIS_UTIL_H

#include "abcdk/redis/redis.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS

/**
 * 
*/
void abcdk_redis_reply_dump(FILE *fp, redisReply *rep);

/**
 * 断开连接.
*/
void abcdk_redis_disconnect(redisContext **ctx);

/**
 * 连接redis服务器.
 * 
 * @param server 服务器地地址(域名或IP).
 * @param port 服务器的端口.
 * @param timeout 超时.> 0 有效.
 * 
 * @return !NULL(0) 成功(句柄), NULL(0) 失败.
*/
redisContext *abcdk_redis_connect(const char *server,uint16_t port,time_t timeout);

/**
 * 授权码验证.
 * 
 * @param auth 授权字符串.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_redis_auth(redisContext *ctx,const char *auth);

/**
 * 设置授权码.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_redis_set_auth(redisContext *ctx,const char *auth);

/**
 * 获取授权码.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_redis_get_auth(redisContext *ctx,char auth[128]);


__END_DECLS

#endif //ABCDK_REDIS_UTIL_H