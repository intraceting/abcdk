/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/exec.h"

int abcdk_exec(const char *cmd, char *const *args, char *const *envs,
               uid_t uid, gid_t gid, const char *rpath, const char *wpath)
{
    int chk;

    assert(cmd != NULL && args != NULL && envs != NULL);

    if (uid != 0)
    {
        chk = setuid(uid);
        if (chk != 0)
            return -2;
    }

    if (gid != 0)
    {
        chk = setgid(gid);
        if (chk != 0)
            return -3;
    }

    if (rpath)
    {
        chk = chroot(rpath);
        if (chk != 0)
            return -4;
    }

    if (wpath)
    {
        chk = chdir(wpath);
        if (chk != 0)
            return -5;
    }

    chk = execve(cmd, args, envs);

    /*执行到这里时，表示出错了。*/
    return chk;
}