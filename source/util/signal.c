/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/signal.h"

void abcdk_sigwaitinfo(const abcdk_signal_t *sig, time_t timeout)
{
    sigset_t old;
    struct timespec tout;
    siginfo_t info;
    int chk;

    assert(sig);

    chk = pthread_sigmask(SIG_BLOCK, &sig->signals, &old);
    if (chk != 0)
        return;

    while (1)
    {
        if (timeout >= 0)
        {
            tout.tv_sec = timeout / 1000;
            tout.tv_nsec = (timeout % 1000) * 1000000;
            chk = sigtimedwait(&sig->signals, &info, &tout);
        }
        else
        {
            chk = sigwaitinfo(&sig->signals, &info);
        }

        if (chk == -1)
            break;
        
        if(sig->signal_cb)
            chk = sig->signal_cb(&info,sig->opaque);

        /* -1 终止。 */
        if(chk == -1)
            break;
    }
    
    pthread_sigmask(SIG_BLOCK,&old,NULL);

    return;
}

void* _abcdk_sigwaitinfo_routine(void *opaque)
{
    abcdk_signal_t *sig = (abcdk_signal_t *)opaque;

    abcdk_sigwaitinfo(sig,-1);

    /*这里要释放，不然会有内存泄漏的问题。*/
    abcdk_heap_free(sig);
}

void abcdk_sigwaitinfo_async(const abcdk_signal_t *sig)
{
    abcdk_signal_t *sig_cp = NULL;
    abcdk_thread_t td = {0};
    int chk;

    assert(sig != NULL);

    /*在线程中使用，因此需要复制对象。*/
    sig_cp = abcdk_heap_alloc(sizeof(abcdk_signal_t));
    memcpy(sig_cp,sig,sizeof(abcdk_signal_t));

    td.opaque = sig_cp;
    td.routine = _abcdk_sigwaitinfo_routine;

    chk = abcdk_thread_create(&td,0);

    /*如果线程未能创建，则要释放申请的内存，不然会造成内存泄漏。*/
    if(chk != 0)
        abcdk_heap_free(sig_cp);
}