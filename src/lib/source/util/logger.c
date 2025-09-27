/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/logger.h"

/** 日志接口。*/
struct _abcdk_logger
{
    /** 文件锁。*/
    abcdk_mutex_t *locker;

    /**PID。 */
    pid_t pid;

    /** 文件句柄。*/
    int fd;

    /** 文件名(包括路径)。*/
    char name[PATH_MAX];

    /** 分段文件名(包括路径)。*/
    char segment_name[PATH_MAX];

    /** 分段数量。*/
    size_t segment_max;
    
    /** 分段大小(MB)。*/
    size_t segment_size;

    /** 
     * 复制到stderr。
     * 是：!0
     * 否： 0
    */
    int copy2stderr;

    /** 
     * 复制到syslog。
     * 是：!0
     * 否： 0
    */
    int copy2syslog;

    /** 
     * 掩码。
     * 
     * 见syslog。
    */
    volatile uint32_t mask;

};// abcdk_logger_t;

void abcdk_logger_close(abcdk_logger_t **ctx)
{
    abcdk_logger_t *ctx_p = NULL;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_closep(&ctx_p->fd);
    abcdk_mutex_destroy(&ctx_p->locker);
    abcdk_heap_free(ctx_p);
}

abcdk_logger_t *abcdk_logger_open(const char *name,const char *segment_name,size_t segment_max,size_t segment_size,int copy2syslog,int copy2stderr)
{
    abcdk_logger_t *ctx = NULL;

    assert(name != NULL);

    ctx = abcdk_heap_alloc(sizeof(abcdk_logger_t));
    if(!ctx)
        return NULL;
    
    ctx->pid = -1;
    ctx->fd = -1;
    ctx->segment_size = segment_size * 1024 * 1024;
    ctx->segment_max = segment_max;
    ctx->copy2stderr = copy2stderr;
    ctx->copy2syslog = copy2syslog;
    ctx->mask = 0xFFFFFFFF;
    ctx->locker = abcdk_mutex_create();

    /*复制文件名。*/
    strncpy(ctx->name, name, PATH_MAX);

    /*复制或构造分段文件名。*/
    if (segment_name)
    {
        if (segment_name[0] == '/')
            strncpy(ctx->segment_name, segment_name, PATH_MAX);
        else
        {
            abcdk_dirname(ctx->segment_name,ctx->name);
            abcdk_dirdir(ctx->segment_name,segment_name);
        }
    }

    return ctx;
}

abcdk_logger_t *abcdk_logger_open2(const char *path,const char *name, const char *segment_name, size_t segment_max, size_t segment_size, int copy2syslog, int copy2stderr)
{
    char pathfile[PATH_MAX] = {0};

    abcdk_dirdir(pathfile,path);
    abcdk_dirdir(pathfile,name);

    return abcdk_logger_open(pathfile,segment_name,segment_max,segment_size,copy2syslog,copy2stderr);
}

/** 检查日志类型。*/
#define ABCDK_LOGGER_TYPE_CHECK(t) ((t) >= ABCDK_LOGGER_TYPE_ERROR && (t) < ABCDK_LOGGER_TYPE_MAX)

void abcdk_logger_mask(abcdk_logger_t *ctx, int type, ...)
{
    uint32_t mask = 0;

    assert(ctx != NULL);

    va_list vaptr;
    va_start(vaptr, type);
    for (;;)
    {
        if (!ABCDK_LOGGER_TYPE_CHECK(type))
            break;

        mask |= (1 << type);

        /*遍历后续的。*/
        type = va_arg(vaptr, int);
    }
    va_end(vaptr);

    /*覆盖现有的。*/
    abcdk_atomic_store(&ctx->mask, mask);
}

static int _abcdk_logger_segment(const char *src, const char *dst, int max)
{
    char tmp[PATH_MAX] = {0};
    char tmp2[PATH_MAX] = {0};
    int chk;

    assert(src != NULL && dst != NULL && max > 0);

    /*依次修改分段文件编号。*/
    for (int i = max; i > 0; i--)
    {
        /*编号较大的分段文件。*/
        snprintf(tmp2, PATH_MAX, dst, i);

        /*删除编号最大的分段文件。*/
        if (i == max)
        {
            if (access(tmp2, F_OK) == 0)
            {
                chk = remove(tmp2);
                if (chk != 0)
                    return -1;
            }
        }

        /*编号较小的分段文件。*/
        if (i > 1)
            snprintf(tmp, PATH_MAX, dst, i - 1);
        else
            strncpy(tmp, src, PATH_MAX);

        /*跳过不存在的分段文件。*/
        if (access(tmp, F_OK) != 0)
            continue;

        chk = rename(tmp,tmp2);
        if (chk != 0)
            return -1;
    }

    return 0;
}

static void _abcdk_logger_dump2file(abcdk_logger_t *ctx,const char *str)
{
    struct stat attr;
    int chk;

open_log_file:

    if (ctx->fd < 0 || ctx->pid != getpid())
    {
        abcdk_mkdir(ctx->name, 0755);

        ctx->pid = getpid();

        abcdk_closep(&ctx->fd);
        ctx->fd = abcdk_open(ctx->name, 1, 0, 1);

        if (ctx->fd >= 0)
            fchmod(ctx->fd, 0644);
    }

    if (ctx->fd < 0)
        return;

    chk = fstat(ctx->fd, &attr);
    if (chk != 0)
        return;

    if (ctx->segment_name[0] &&
        ctx->segment_max > 0 &&
        ctx->segment_size > 0 &&
        attr.st_size >= ctx->segment_size)
    {
        abcdk_closep(&ctx->fd);
        _abcdk_logger_segment(ctx->name, ctx->segment_name, ctx->segment_max);
        goto open_log_file;
    }
    else
    {
        /*加锁。*/
        flock(ctx->fd,LOCK_EX);

        /*在末尾追加。*/
        lseek(ctx->fd, 0, SEEK_END);

        /*写，内部会保正写完。如果写不完，就是出错或没空间了。*/
        abcdk_write(ctx->fd, str, strlen(str));

        /*解锁。*/
        flock(ctx->fd,LOCK_UN);

#if 0
        /*落盘，非常慢。*/
        fsync(ctx->fd);
#endif
    }
}

static void _abcdk_logger_dump(void *opaque,int type, const char* str)
{
    abcdk_logger_t *ctx = (abcdk_logger_t *)opaque;

    /*如果不需要记录，直接跳过。*/
    if (!(abcdk_atomic_load(&ctx->mask) & (1 << type)))
        return;

    /*记录到文件。*/
    _abcdk_logger_dump2file(ctx, str);

    /*可能需要复制到stderr。*/
    if (ctx->copy2stderr)
        fprintf(stderr, "%s", str);

    /*可能需要复制到syslog。*/
    if (ctx->copy2syslog)
        syslog(type, "%s", str);
}

void abcdk_logger_output(abcdk_logger_t *ctx, int type, const char *str)
{
    assert(ctx != NULL && str != NULL);

    /*加锁，确保每个线程写操作不被打断。*/
    abcdk_mutex_lock(ctx->locker, 1);

    /*输出。*/
    abcdk_trace_output(type,str,_abcdk_logger_dump,ctx);
    
    /*解锁，给其它线程写入的机会。*/
    abcdk_mutex_unlock(ctx->locker);
}

void abcdk_logger_vprintf(abcdk_logger_t *ctx, int type, const char *fmt, va_list ap)
{
    char buf[8000] = {0};

    assert(ctx != NULL && fmt != NULL);

    vsnprintf(buf, 8000, fmt, ap);

    abcdk_logger_output(ctx,type,buf);
}

void abcdk_logger_printf(abcdk_logger_t *ctx, int type, const char *fmt, ...)
{
    assert(ctx != NULL && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_logger_vprintf(ctx, type, fmt, ap);
    va_end(ap);
}

void abcdk_logger_proxy(void *opaque,int type, const char* str)
{
    abcdk_logger_t *ctx = (abcdk_logger_t *)opaque;

    abcdk_logger_output(ctx,type,str);
}
