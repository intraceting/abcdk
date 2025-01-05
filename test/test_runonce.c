/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int lock_fd = -1;

static int _runonce_process_cb(void *opaque)
{
    abcdk_trace_printf(LOG_DEBUG,"%s:begin",__FUNCTION__);

    int chk = flock(lock_fd, LOCK_SH | LOCK_NB);
    assert(chk == 0);

    sleep(50);

    abcdk_trace_printf(LOG_DEBUG,"%s:end",__FUNCTION__);

    return 1;
}

int abcdk_test_runonce(abcdk_option_t *args)
{
    int chk,pid;

    lock_fd = abcdk_open("/tmp/test.runonce.pid",1,0,1);

    while (1)
    {
        chk = abcdk_proc_singleton_lock(lock_fd, &pid);
        assert(chk == 0);

        chk = flock(lock_fd, LOCK_EX | LOCK_NB);
        assert(chk == 0);

        int exitcode=0,sigcode=999;
        chk = abcdk_proc_subprocess(_runonce_process_cb, NULL,&exitcode,&sigcode);
        if (chk == 0)
            break;

        abcdk_trace_printf(LOG_DEBUG,"exitcode(%d),sigcode(%d)\n",exitcode,sigcode);
    }

    abcdk_closep(&lock_fd);

    return 0;
}