/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/signal.h"

int abcdk_signal_wait(const sigset_t *sigs, siginfo_t *info, time_t timeout)
{
    struct timespec tout;
    int chk;

    assert(sigs != NULL && info != NULL);

    while (1)
    {
        if (timeout >= 0)
        {
            tout.tv_sec = timeout / 1000;
            tout.tv_nsec = (timeout % 1000) * 1000000;
            chk = sigtimedwait(sigs, info, &tout);
        }
        else
        {
            chk = sigwaitinfo(sigs, info);
        }

        if (chk == -1 && errno == EINTR)
            continue;
        else
            break;
    }

    return chk;
}

void abcdk_signal_fill(sigset_t *sigs,int sigdel,...)
{
    assert(sigs != NULL);

    sigemptyset(sigs);
    sigfillset(sigs);

    va_list vaptr;
    va_start(vaptr, sigdel);
    for(;;)
    {
        if (sigdel == -1)
            break;

        sigdelset(sigs, sigdel);

        /*遍历后续的。*/
        sigdel = va_arg(vaptr, int);
    }
    va_end(vaptr);
}

int abcdk_signal_block(const sigset_t *news,sigset_t *olds)
{
    int chk;

    assert(news != NULL || olds != NULL);

    chk = pthread_sigmask(SIG_BLOCK, news, olds);
    if (chk != 0)
        return -1;

    return 0;
}
