/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/exec.h"

pid_t abcdk_fork(abcdk_fork_process_cb process_cb, void *opaque,
                 int *stdin_fd, int *stdout_fd, int *stderr_fd)
{
    pid_t child = -1, sid = -1;
    int out2in_fd[2] = {-1, -1};
    int in2out_fd[2] = {-1, -1};
    int in2err_fd[2] = {-1, -1};
    int chk;

    assert(process_cb != NULL);

    if (pipe(out2in_fd) != 0)
        goto error;

    if (pipe(in2out_fd) != 0)
        goto error;

    if (pipe(in2err_fd) != 0)
        goto error;

    child = fork();
    if (child < 0)
        goto error;

    if (child == 0)
    {
        /*创建一个新会话并脱离终端控制.*/
        sid = setsid();
        if (sid < 0)
            exit(126);

        if (stdin_fd)
            dup2(out2in_fd[0], STDIN_FILENO);
        else
            abcdk_reopen(STDIN_FILENO, "/dev/null", 0, 0, 0);

        abcdk_closep(&out2in_fd[1]);
        abcdk_closep(&out2in_fd[0]);

        if (stdout_fd)
            dup2(in2out_fd[1], STDOUT_FILENO);
        else
            abcdk_reopen(STDOUT_FILENO, "/dev/null", 1, 0, 0);

        abcdk_closep(&in2out_fd[0]);
        abcdk_closep(&in2out_fd[1]);

        if (stderr_fd)
            dup2(in2err_fd[1], STDERR_FILENO);
        else
            abcdk_reopen(STDERR_FILENO, "/dev/null", 1, 0, 0);

        abcdk_closep(&in2err_fd[0]);
        abcdk_closep(&in2err_fd[1]);

        /*执行子进程处理流程.*/
        chk = process_cb(opaque);
        exit(chk & 0x7F);
    }
    else
    {
        /*
         * 关闭不需要的句柄.
         */
        abcdk_closep(&out2in_fd[0]);
        abcdk_closep(&in2out_fd[1]);
        abcdk_closep(&in2err_fd[1]);

        if (stdin_fd)
            *stdin_fd = out2in_fd[1];
        else
            abcdk_closep(&out2in_fd[1]);

        if (stdout_fd)
            *stdout_fd = in2out_fd[0];
        else
            abcdk_closep(&in2out_fd[0]);

        if (stderr_fd)
            *stderr_fd = in2err_fd[0];
        else
            abcdk_closep(&in2err_fd[0]);

        return child;
    }

error:

    abcdk_closep(&out2in_fd[0]);
    abcdk_closep(&out2in_fd[1]);
    abcdk_closep(&in2out_fd[0]);
    abcdk_closep(&in2out_fd[1]);
    abcdk_closep(&in2err_fd[0]);
    abcdk_closep(&in2err_fd[1]);

    return -1;
}

typedef struct _abcdk_system_param
{
    const char *filename;
    char *const *args;
    char *const *envs;
    uid_t uid;
    gid_t gid;
    const char *rpath;
    const char *wpath;
} abcdk_system_param_t;

int _abcdk_system_process_cb(void *opaque)
{
    abcdk_system_param_t *param_p = (abcdk_system_param_t *)opaque;
    int chk;

    if (param_p->uid != 0)
    {
        chk = setuid(param_p->uid);
        if (chk != 0)
            return 125;
    }

    if (param_p->gid != 0)
    {
        chk = setgid(param_p->gid);
        if (chk != 0)
            return 124;
    }

    if (param_p->rpath)
    {
        chk = chroot(param_p->rpath);
        if (chk != 0)
            return 123;
    }

    if (param_p->wpath)
    {
        chk = chdir(param_p->wpath);
        if (chk != 0)
            return 122;
    }

    chk = execve(param_p->filename, param_p->args, (param_p->envs ? param_p->envs : environ));
    if (chk != 0)
        return errno;

    /*正常情况下, 永远也不可能到这里.*/
    return 121;
}

pid_t abcdk_system(const char *filename, char *const *args, char *const *envs,
                   uid_t uid, gid_t gid, const char *rpath, const char *wpath,
                   int *stdin_fd, int *stdout_fd, int *stderr_fd)
{
    abcdk_system_param_t param = {0};

    assert(filename != NULL && args != NULL);

    param.filename = filename;
    param.args = args;
    param.envs = envs;
    param.uid = uid;
    param.gid = gid;
    param.rpath = rpath;
    param.wpath = wpath;

    return abcdk_fork(_abcdk_system_process_cb, &param, stdin_fd, stdout_fd, stderr_fd);
}

pid_t abcdk_popen(const char *cmdline, char *const *envs, uid_t uid,
                  gid_t gid, const char *rpath, const char *wpath,
                  int *stdin_fd, int *stdout_fd, int *stderr_fd)
{
    char * args[5] = {NULL};

    assert(cmdline != NULL);

    args[0] = "/bin/sh";//不会执行, 仅做为第一个参数传给/bin/sh.
    args[1] = "-c";
    args[2] = (char*)cmdline;
    args[3] = NULL;

    return abcdk_system("/bin/sh", args, envs, uid, gid, rpath, wpath, stdin_fd, stdout_fd, stderr_fd);
}
