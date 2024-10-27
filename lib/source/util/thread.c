/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/thread.h"

int abcdk_thread_create(abcdk_thread_t *ctx,int joinable)
{
    pthread_attr_t attr;
    char name[17] = {0};
    int chk = -1;
  
    assert(ctx);
    assert(ctx->routine);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,(joinable?PTHREAD_CREATE_JOINABLE:PTHREAD_CREATE_DETACHED));
    chk = pthread_create(&ctx->handle,&attr,ctx->routine,ctx->opaque);
    pthread_attr_destroy(&attr);

    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_thread_join(abcdk_thread_t *ctx)
{
    pthread_attr_t attr;
    int detachstate = -1;
    int chk = -1;

    assert(ctx);

    pthread_attr_init(&attr);
    chk = pthread_getattr_np(ctx->handle,&attr);
    if (chk != 0)
        goto final;

    chk = pthread_attr_getdetachstate(&attr,&detachstate);
    if (chk != 0)
        goto final;

    if (detachstate == PTHREAD_CREATE_JOINABLE)
        chk = pthread_join(ctx->handle, &ctx->result);

final:

    pthread_attr_destroy(&attr);

    return chk;
}

int abcdk_thread_create_group(int count, abcdk_thread_t *ctxs, int joinable)
{
    int chk, num = 0;

    assert(count > 0 && ctxs != NULL);

    for (int i = 0; i < count; i++)
    {
        chk = abcdk_thread_create(ctxs + i, joinable);
        if (chk != 0)
            break;

        num += 1;
    }

    return num;
}

#if !defined(__ANDROID__)

int abcdk_thread_setaffinity(pthread_t tid, int cpus[])
{
    long nps = 1;
    cpu_set_t mark;
    int chk;

    CPU_ZERO(&mark);

    nps = sysconf(_SC_NPROCESSORS_ONLN);

    for (int i = 0; i < nps; i++)
    {
        /*-1 结束。*/
        if (cpus[i] < 0)
            break;

        /*检查ID范围。*/
        if (cpus[i] >= nps)
            continue;

        /*检查是否已存在。*/
        if(CPU_ISSET(cpus[i], &mark))
            continue;

        /*记录。*/
        CPU_SET(cpus[i], &mark);
    }

    chk = pthread_setaffinity_np(tid, sizeof(mark), &mark);

    return chk;
}

int abcdk_thread_setaffinity2(pthread_t tid,int cpu)
{
    int cpus[2] = {-1};

    cpus[0] = cpu;
    cpus[1] = -1;

    return abcdk_thread_setaffinity(tid,cpus);
}

#endif //__ANDROID__

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

void abcdk_thread_setname(const char *fmt, ...)
{
    char name[18] = {0};

    va_list args;
    va_start(args, fmt);
    vsnprintf(name, 17, fmt, args);
    va_end(args);

    pthread_setname_np(pthread_self(), name);
}
