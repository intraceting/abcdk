/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/popen.h"

pid_t abcdk_popen(const char *cmdline, char *const *envs, uid_t uid,
                  gid_t gid, const char *rpath, const char *wpath,
                  int *stdin_fd, int *stdout_fd, int *stderr_fd)
{
    char * args[5] = {NULL};

    assert(cmdline != NULL);

    args[0] = "sh";
    args[1] = "-c";
    args[2] = (char*)cmdline;
    args[3] = NULL;

    return abcdk_system("/bin/sh", args, envs, uid, gid, rpath, wpath, stdin_fd, stdout_fd, stderr_fd);
}
