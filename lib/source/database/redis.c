/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
*/
#include "abcdk/database/redis.h"

#ifdef __HIREDIS_H

void abcdk_redis_reply_dump(FILE *fp, redisReply *reply)
{
    static char *str_type[]={"","STRING","ARRAY","INTEGER","NIL","STATUS","ERROR"};

    assert(fp != NULL && reply != NULL);

    if (reply->type == REDIS_REPLY_STRING)
        fprintf(fp, "\"%s\"\n", reply->str);
    else if (reply->type == REDIS_REPLY_INTEGER)
        fprintf(fp, "%lld\n", reply->integer);
    else if (reply->type == REDIS_REPLY_STATUS)
        fprintf(fp, "%s\n", reply->str);
    else if (reply->type == REDIS_REPLY_ERROR)
        fprintf(fp, "(%s) %s\n",str_type[reply->type],reply->str);
    else if (reply->type == REDIS_REPLY_NIL)
        fprintf(fp, "(%s)\n", str_type[reply->type]);
    else if (reply->type == REDIS_REPLY_ARRAY)
    {
        for (size_t i = 0; i < reply->elements; i++)
        {
            fprintf(fp, "%zu) ", i+1);
            abcdk_redis_reply_dump(fp, reply->element[i]);
        }
    }

}

void abcdk_redis_disconnect(redisContext **ctx)
{
    redisContext *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    redisFree(ctx_p);
}

redisContext *abcdk_redis_connect(const char *server, uint16_t port, time_t timeout)
{
    struct timeval tv;

    assert(server != NULL && port > 0 && port < 65536 && timeout > 0);
    
    tv.tv_sec = timeout / 1000;
    tv.tv_usec = (timeout % 1000) * 1000;

    return redisConnectWithTimeout(server, port, tv);
}

int abcdk_redis_auth(redisContext *ctx, const char *auth)
{
    redisReply *reply = NULL;
    int chk = 0;

    assert(ctx != NULL && auth != NULL);

    reply = redisCommand(ctx, "auth %s", auth);
    if (!reply || reply->type == REDIS_REPLY_ERROR)
        chk = -1;

    //abcdk_redis_reply_dump(stderr,reply);

    if(reply)
        freeReplyObject(reply);

    return chk;
}

int abcdk_redis_set_auth(redisContext *ctx,const char *auth)
{
    redisReply *reply = NULL;
    int chk = 0;

    assert(ctx != NULL && auth != NULL);

    reply = redisCommand(ctx, "config set requirepass %s", auth);
    if (!reply || reply->type == REDIS_REPLY_ERROR)
        chk = -1;

    //abcdk_redis_reply_dump(stderr,reply);

    if(reply)
        freeReplyObject(reply);

    return chk;
}

int abcdk_redis_get_auth(redisContext *ctx,char auth[128])
{
    redisReply *reply = NULL;
    int chk = 0;

    assert(ctx != NULL && auth != NULL);

    reply = redisCommand(ctx, "config get requirepass");
    if (!reply || reply->type == REDIS_REPLY_ERROR)
        chk = -1;

    //abcdk_redis_reply_dump(stderr,reply);

    strncpy(auth,reply->element[1]->str ,ABCDK_MIN(reply->element[1]->len,128));

    if(reply)
        freeReplyObject(reply);

    return chk;
}

#endif //__HIREDIS_H