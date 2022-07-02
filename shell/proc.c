/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "shell/proc.h"

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
        /* 这里的覆盖不会影响调用者。*/
        buf = NULL;
    }

    abcdk_heap_free2((void **)&tmp);

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
        /*这里的覆盖不会影响调用者。*/
        buf = NULL;
    }

    abcdk_heap_free2((void **)&tmp);

    return buf;
}

int abcdk_proc_singleton(const char *lockfile,int* pid)
{
    int fd = -1;
    char strpid[16] = {0};

    assert(lockfile);

    fd = abcdk_open(lockfile, 1, 0, 1);
    if (fd < 0)
        return -1;

    /* 通过尝试加独占锁来确定是否程序已经运行。*/
    if (flock(fd, LOCK_EX | LOCK_NB) == 0)
    {
        /* PID可视化，便于阅读。*/
        snprintf(strpid,15,"%d",getpid());

        /* 清空。*/
        ftruncate(fd, 0);

        /*写入文件。*/
        abcdk_write(fd,strpid,strlen(strpid));
        fsync(fd);

        /*进程ID就是自己。*/
        if(pid)
           *pid = getpid();

        /* 走到这里返回锁定文件的句柄。*/
        return fd;
    }

    /* 程序已经运行，进程ID需要从锁定文件中读取。 */
    if(pid)
    {
        abcdk_read(fd,strpid,12);

        if(abcdk_strtype(strpid,isdigit))
            *pid = atoi(strpid);
        else
            *pid = -1;
    }

    /* 独占失败，关闭句柄，返回-1。*/
    abcdk_closep(&fd);
    ABCDK_ERRNO_AND_RETURN1(EPERM,-1);
}
