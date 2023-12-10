/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/exec.h"

int abcdk_exec(const char *filename, char *const *args, char *const *envs,
               uid_t uid, gid_t gid, const char *rpath, const char *wpath)
{
    int chk;

    assert(filename != NULL && args != NULL);

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

    chk = execve(filename, args, (envs ? envs : environ));

    /*执行到这里时，表示出错了。*/
    return chk;
}

pid_t abcdk_exec_new(const char *filename, char *const *args, char *const *envs,
                     uid_t uid, gid_t gid, const char *rpath, const char *wpath,
                     int *stdin_fd, int *stdout_fd, int *stderr_fd)
{
    pid_t child = -1,sid = -1;
    int out2in_fd[2] = {-1, -1};
    int in2out_fd[2] = {-1, -1};
    int in2err_fd[2] = {-1, -1};
    int chk;

    assert(filename != NULL && args != NULL);

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
        /*创建一个新会话并脱离终端控制。*/
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

        chk = abcdk_exec(filename,args,envs,uid,gid,rpath,wpath);

        /*也许永远也不可能到这里.*/
        _exit(chk);
    }
    else
    {
        /*
        * 关闭不需要的句柄。
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