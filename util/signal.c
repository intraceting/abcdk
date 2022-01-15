/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "util/signal.h"

void abcdk_sigwaitinfo(abcdk_signal_t *sig, time_t timeout)
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