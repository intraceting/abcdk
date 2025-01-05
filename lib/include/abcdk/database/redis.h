/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
*/
#ifndef ABCDK_DATABASE_REDIS_H
#define ABCDK_DATABASE_REDIS_H

#include "abcdk/util/general.h"

#ifdef HAVE_REDIS
#include <hiredis/hiredis.h>
#endif //HAVE_REDIS

__BEGIN_DECLS

#ifdef __HIREDIS_H

/**
 * 
*/
void abcdk_redis_reply_dump(FILE *fp, redisReply *rep);

/**
 * 断开连接。
*/
void abcdk_redis_disconnect(redisContext **ctx);

/**
 * 连接redis服务器。
 * 
 * @param server 服务器地地址(域名或IP)。
 * @param port 服务器的端口。
 * @param timeout 超时。> 0 有效。
 * 
 * @return !NULL(0) 成功(句柄)，NULL(0) 失败。
*/
redisContext *abcdk_redis_connect(const char *server,uint16_t port,time_t timeout);

/**
 * 授权码验证。
 * 
 * @param auth 授权字符串。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_redis_auth(redisContext *ctx,const char *auth);

/**
 * 设置授权码。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_redis_set_auth(redisContext *ctx,const char *auth);

/**
 * 获取授权码。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_redis_get_auth(redisContext *ctx,char auth[128]);

#endif //__HIREDIS_H

__END_DECLS

#endif //ABCDK_DATABASE_REDIS_H