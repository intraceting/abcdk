/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/thread.h"

int abcdk_thread_create(abcdk_thread_t *ctx,int joinable)
{
    int err = -1;
    pthread_attr_t attr;
    char name[17] = {0};
  
    assert(ctx);
    assert(ctx->routine);

    abcdk_thread_getname(name);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,(joinable?PTHREAD_CREATE_JOINABLE:PTHREAD_CREATE_DETACHED));

    err = pthread_create(&ctx->handle,&attr,ctx->routine,ctx->opaque);
    if(err == 0)
        abcdk_thread_setname(name);

    pthread_attr_destroy(&attr);

    return err;
}


int abcdk_thread_join(abcdk_thread_t *ctx)
{
    int err = -1;
    pthread_attr_t attr;
    int detachstate = -1;

    assert(ctx);

    err = pthread_getattr_np(ctx->handle,&attr);
    if (err != 0)
        return err;

    err = pthread_attr_getdetachstate(&attr,&detachstate);
    if (err != 0)
        return err;
    
    pthread_attr_destroy(&attr);

    if (detachstate == PTHREAD_CREATE_JOINABLE)
        err = pthread_join(ctx->handle, &ctx->result);

    return err;
}

int abcdk_thread_setname(const char* fmt,...)
{
    int err = -1;
    char name[17] = {0};

    assert(fmt && fmt[0]);

    va_list vaptr;
    va_start(vaptr, fmt);
    vsnprintf(name,16,fmt,vaptr);
    va_end(vaptr);

    err = pthread_setname_np(pthread_self(),name);

    return err;
}

int abcdk_thread_getname(char name[16])
{
    int err = -1;

    assert(name);

    err = pthread_getname_np(pthread_self(),name,16);

    return err; 
}

int abcdk_thread_leader_vote(volatile pthread_t *tid)
{
    pthread_t self_tid = pthread_self();

    if(abcdk_atomic_compare_and_swap(tid, 0, self_tid))
        return 0;

    return -1;
}

int abcdk_thread_leader_test(const volatile pthread_t *tid)
{
    pthread_t self_tid = pthread_self();

    if(abcdk_atomic_compare((volatile pthread_t *)tid,self_tid))
        return 0;

    return -1;
}

int abcdk_thread_leader_quit(volatile pthread_t *tid)
{
    pthread_t self_tid = pthread_self();

    if(abcdk_atomic_compare_and_swap(tid, self_tid, 0))
        return 0;

    return -1;
}
