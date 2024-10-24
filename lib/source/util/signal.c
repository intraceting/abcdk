/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/signal.h"

void abcdk_signal_set(sigset_t *sigs,int op, int sig,...)
{
    assert(sigs != NULL);

    va_list vaptr;
    va_start(vaptr, sig);
    for(;;)
    {
        if (sig == -1)
            break;

        if(op)
            sigdelset(sigs, sig);
        else 
            sigaddset(sigs, sig);

        /*遍历后续的。*/
        sig = va_arg(vaptr, int);
    }
    va_end(vaptr);
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

        abcdk_signal_set(sigs,1, sigdel,-1);

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

int abcdk_signal_wait(siginfo_t *info, const sigset_t *sigs, time_t timeout)
{
    sigset_t in_sigs = {0},*sigs_p = NULL;
    struct timespec tout;
    int chk;

    assert(info != NULL);

    sigs_p = (sigset_t*)sigs;
    if(!sigs_p)
    {
        abcdk_signal_fill(&in_sigs,SIGTRAP,SIGKILL,SIGSEGV, SIGSTOP,SIGCHLD, -1);
        sigs_p = &in_sigs;
    }

    while (1)
    {
        if (timeout >= 0)
        {
            tout.tv_sec = timeout / 1000;
            tout.tv_nsec = (timeout % 1000) * 1000000;
            chk = sigtimedwait(sigs_p, info, &tout);
        }
        else
        {
            chk = sigwaitinfo(sigs_p, info);
        }

        if (chk > 0)
            return chk;
        else if (chk == -1)
        {
            if (errno == EINTR)
                continue;
            else if (errno == EAGAIN)
                return 0;
            else 
                return -1;
        }
    }

    return 0;
}