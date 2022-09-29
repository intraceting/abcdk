/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include "abcdk/util/thread.h"
#include "abcdk/util/crc32.h"
#include "abcdk/util/clock.h"
#include "abcdk/util/signal.h"

void* specific_cb(void* args)
{
    abcdk_thread_setname("haha");

    printf("dot:%lu\n",abcdk_clock_dot(NULL));

    uint32_t sum = abcdk_crc32_sum("abc",3,0);

    printf("sun=%u,%08X\n",sum,sum);

    printf("step:%lu\n",abcdk_clock_step(NULL));

    sum = abcdk_crc32_sum("abc",3,sum);

    printf("sun=%u,%08X\n",sum,sum);

    printf("step:%lu\n",abcdk_clock_step(NULL));

    sleep(3);

    printf("dot:%lu\n",abcdk_clock_dot(NULL));

    return NULL;
}

int signal_cb(const siginfo_t *info, void *opaque)
{
    printf("signo=%d,errno=%d,code =%d\n",info->si_signo,info->si_errno,info->si_code);

    switch(info->si_code)
    {
        case SI_USER:
        {
            printf("pid=%d,uid=%d\n",info->si_pid,info->si_uid);
        }
        break;
    }

    return 1;
}

int main(int argc, char **argv)
{
    abcdk_thread_setname("hehe");

    abcdk_mutex_t m;
    abcdk_mutex_init2(&m,1);

    abcdk_mutex_lock(&m,0);

    abcdk_mutex_unlock(&m);

    abcdk_mutex_destroy(&m);

    abcdk_thread_t p;
    p.routine = specific_cb;
    abcdk_thread_create(&p,1);
    abcdk_thread_join(&p);

    abcdk_signal_t sig;
    sigfillset(&sig.signals);
    sig.signal_cb = signal_cb;
    sig.opaque = NULL;
     
    abcdk_sigwaitinfo(&sig,-1);

    return 0;
}