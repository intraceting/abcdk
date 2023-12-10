/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/daemon.h"

int abcdk_daemon(int background, abcdk_daemon_entry_cb entry_cb, abcdk_daemon_monitor_cb monitor_cb, void *opaque)
{
    pid_t cid = -1, wcid = -1, sid = -1;
    int wstatus;
    int chk;

    if (background)
    {
        chk = daemon(1, 0);
        if (chk != 0)
            return -1;
    }

RESTART:

    /*恢复默认的状态值。*/
    wstatus = chk = 0;

    cid = fork();
    if (cid == 0)
    {
        /*创建一个新会话并脱离终端控制。*/
        sid = setsid();
        if (sid < 0)
            exit(126);

        abcdk_reopen(STDIN_FILENO, "/dev/null", 0, 0, 0);
        abcdk_reopen(STDOUT_FILENO, "/dev/null", 0, 0, 0);
        abcdk_reopen(STDERR_FILENO, "/dev/null", 0, 0, 0);

        chk = entry_cb(opaque);
        exit(chk & 0x7F);
    }

    wcid = waitpid(cid, &wstatus, 0);
    if (wcid < 0)
        return -2;

    chk = monitor_cb(wstatus,opaque);
    if (chk < 0)
        return 0;

    sleep(chk);
    goto RESTART;
}