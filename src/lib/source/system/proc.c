/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/system/proc.h"

char *abcdk_proc_pathfile(char *buf)
{
    assert(buf);

    if (readlink("/proc/self/exe", buf, PATH_MAX) == -1)
        return NULL;

    return buf;
}

char *abcdk_proc_dirname(char *buf, const char *append)
{
    char *tmp = NULL;

    assert(buf);

    tmp = abcdk_heap_alloc(PATH_MAX);
    if (!tmp)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    if (abcdk_proc_pathfile(tmp))
    {
        abcdk_dirname(buf, tmp);

        if (append)
            abcdk_dirdir(buf, append);
    }
    else
    {
        /* 这里的覆盖不会影响调用者.*/
        buf = NULL;
    }

    abcdk_heap_freep((void **)&tmp);

    return buf;
}

char *abcdk_proc_basename(char *buf)
{
    char *tmp = NULL;

    assert(buf);

    tmp = abcdk_heap_alloc(PATH_MAX);
    if (!tmp)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    if (abcdk_proc_pathfile(tmp))
    {
        abcdk_basename(buf, tmp);
    }
    else
    {
        /*这里的覆盖不会影响调用者.*/
        buf = NULL;
    }

    abcdk_heap_freep((void **)&tmp);

    return buf;
}

int abcdk_proc_singleton_lock(int pid_fd, int* pid)
{
    char buf[16] = {0};
    int chk;

    assert(pid_fd >= 0);

    /* 通过尝试加独占锁来确定是否程序已经运行.*/
    chk = flock(pid_fd, LOCK_EX | LOCK_NB);
    if (chk != 0)
    {
        if (pid)
        {
            /*从PID文件中读取锁定进程的PID. */
            lseek(pid_fd, 0, SEEK_SET);
            abcdk_read(pid_fd, buf, 15);

            if (abcdk_strtype(buf, isdigit))
                *pid = atoi(buf);
            else
                *pid = -1;
        }

        /* 锁定失败, 已经被其它进程锁定.*/
        return -1;
    }

    /*进程PID以十进制文本格式写入文件, 例: 2021*/
    snprintf(buf, 15, "%-d", getpid());

    /*清空, 写入文件.*/
    ftruncate(pid_fd, 0);
    abcdk_write(pid_fd, buf, strlen(buf));
    fsync(pid_fd);

    /*进程ID就是自己.*/
    if (pid)
        *pid = getpid();

    /*锁定成功.*/
    return 0;
}

int abcdk_proc_singleton_kill(int pid_fd , int signum)
{
    char buf[16] = {0};
    int pid = -1;
    int chk;

    /*通过尝试加独占锁来确定是否程序已经运行.*/
    chk = flock(pid_fd, LOCK_EX | LOCK_NB);
    if (chk == 0)
    {
        /*锁定成功, 表示进程已经结束, 因此在返回前必须先解锁.*/
        flock(pid_fd, LOCK_UN);
        return -1;
    }

    /*从PID文件中读取锁定进程的PID. */
    lseek(pid_fd, 0, SEEK_SET);
    abcdk_read(pid_fd, buf, 15);

    if (abcdk_strtype(buf, isdigit))
        pid = atoi(buf);
    else
        pid = -1;

    if (pid < 0)
        return -1;

    kill(pid, signum);
    return 0;
}

pid_t abcdk_proc_popen(int *stdin_fd, int *stdout_fd, int *stderr_fd, const char *cmd, ...)
{
    pid_t pid = -1;

    assert(cmd != NULL);

    va_list ap;
    va_start(ap, cmd);

    pid = abcdk_proc_vpopen(stdin_fd,stdout_fd,stderr_fd,cmd,ap);

    va_end(ap);
 
    return pid;
}

pid_t abcdk_proc_vpopen(int *stdin_fd, int *stdout_fd, int *stderr_fd, const char *cmd, va_list ap)
{
    char *buf = NULL;
    pid_t pid = -1;

    assert(cmd != NULL);

    buf = abcdk_heap_alloc(40*1024);
    if(!buf)
        goto ERR;

    vsnprintf(buf,40*1024,cmd,ap);

    abcdk_trace_printf(LOG_DEBUG,ABCDK_GETTEXT("popen: %s"),buf);

    pid = abcdk_popen(buf, NULL, 0, 0, NULL, NULL, stdin_fd, stdout_fd, stderr_fd);
    if (pid < 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("'%s'执行失败."));

        goto ERR;
    }

    abcdk_heap_freep((void**)&buf);
    return pid;

ERR:

    abcdk_heap_freep((void**)&buf);
    return -1; 
}

int abcdk_proc_shell(int *exitcode , int *sigcode,const char *cmd,...)
{
    int chk;

    assert(cmd != NULL);

    va_list ap;
    va_start(ap, cmd);

    chk = abcdk_proc_vshell(exitcode,sigcode,cmd,ap);

    va_end(ap);
 
    return chk;
}

int abcdk_proc_vshell(int *exitcode , int *sigcode,const char *cmd,va_list ap)
{
    pid_t pid = -1;

    assert(cmd != NULL);

    pid = abcdk_proc_vpopen(NULL,NULL,NULL,cmd,ap);
    if(pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, exitcode, sigcode);
    return 0;
}


int abcdk_proc_signal_block(const sigset_t *news, sigset_t *olds)
{
    sigset_t default_sigs = {0},*p = NULL;

    ABCDK_TRACE_ASSERT(getpid() == abcdk_gettid(),ABCDK_GETTEXT("仅限主线程调用."));

    /*阻塞信号.*/
    abcdk_signal_fill(&default_sigs, SIGTRAP, SIGKILL, SIGSEGV, SIGSTOP, -1);

    p = (sigset_t*)(news ? news : &default_sigs);

    return abcdk_signal_block(p, olds);
}

int abcdk_proc_signal_wait(siginfo_t *info, time_t timeout)
{
    return abcdk_signal_wait(info,NULL, timeout);
}

int abcdk_proc_wait_exit_signal(time_t timeout)
{
    siginfo_t info = {0};
    int chk;

RETRY: 

    chk = abcdk_proc_signal_wait(&info, timeout);
    if (chk == 0)
        goto CHECK_RETRY;
    else if (chk < 0)
        return -1;

    abcdk_trace_siginfo( LOG_WARNING, &info);

    if (SIGILL == info.si_signo || SIGTERM == info.si_signo || SIGINT == info.si_signo || SIGQUIT == info.si_signo)
        return 1;
    
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("终止进程, 请按Ctrl+c组合键或发送SIGTERM(15)信号.例: kill -s 15 %d\n"), getpid());

CHECK_RETRY:

    /*如果死等就重试.*/
    if (timeout < 0)
        goto RETRY;
    else
        return 0;
}

int abcdk_proc_subprocess(abcdk_fork_process_cb process_cb, void *opaque,int *exitcode, int *sigcode)
{
    pid_t cid = -1, cid_chk = -1, cid_pgid = 0x7fffffff;
    int chk;

    assert(process_cb != NULL);

    cid = abcdk_fork(process_cb, opaque, NULL, NULL, NULL);
    if (cid < 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("无法创建子进程, 资源不足.\n"));
        return -1;
    }
    
    /*使子进程成为进程组的组长, 以便后续可以通过进程组管理.*/
    setpgid(cid, cid);

    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("创建子进程(PID=%d)完成, 等待其运行结束.\n"), cid);

    for (;;)
    {
        /*查看终止信号.*/
        chk = abcdk_proc_wait_exit_signal(300);
        if (chk != 0)
        {
            /*通知组内所有子进程退出.*/
            kill(-cid, 15);

            abcdk_waitpid(cid, 0, exitcode, sigcode);

            chk = 0;//父进程收到终止信号, 返回"正常结束".
            break;
        }
        else
        {
            /*查看子进程状态.*/
            cid_chk = abcdk_waitpid(cid, WNOHANG, exitcode, sigcode);
            if (cid_chk < 0)
            {
                chk = -2;//子进程不存在, 返回"已结束或被终止".
                break;
            }
        }
    }

    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("子进程(PID=%d)已终止.\n"), cid);

    return chk;
}

int abcdk_proc_subprocess2(const char *cmdline,int *exitcode, int *sigcode)
{
    pid_t cid = -1, cid_chk = -1, cid_pgid = 0x7fffffff;
    int chk;

    assert(cmdline != NULL);

    cid = abcdk_proc_popen(NULL, NULL, NULL, cmdline);
    if (cid < 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("父进程无法创建子进程, 结束守护服务.\n"));
        return -1;
    }
    
    /*使子进程成为进程组的组长, 以便后续可以通过进程组管理.*/
    setpgid(cid, cid);

    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("创建子进程(PID=%d)完成, 等待其运行结束.\n"), cid);

    for (;;)
    {
        /*查看终止信号.*/
        chk = abcdk_proc_wait_exit_signal(300);
        if (chk != 0)
        {
            /*通知组内所有子进程退出.*/
            kill(-cid, 15);

            abcdk_waitpid(cid, 0, exitcode, sigcode);

            chk = 0;//父进程收到终止信号, 返回"正常结束".
            break;
        }
        else
        {
            /*查看子进程状态.*/
            cid_chk = abcdk_waitpid(cid, WNOHANG, exitcode, sigcode);
            if (cid_chk < 0)
            {
                chk = -2;//子进程不存在, 返回"已结束或被终止".
                break;
            }
        }
    }

    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("子进程(PID=%d)已终止.\n"), cid);

    return chk;
}