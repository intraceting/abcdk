/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/log/logger.h"
#include "abcdk/shell/file.h"
#include "abcdk/shell/proc.h"

/** 日志接口。*/
struct _abcdk_logger
{
    /** 文件锁。*/
    abcdk_mutex_t *locker;

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

    /** 缓存。*/
    abcdk_object_t *buf;
    size_t bufpos;

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
    abcdk_object_unref(&ctx_p->buf);

    abcdk_heap_free(ctx_p);
}

abcdk_logger_t *abcdk_logger_open(const char *name,const char *segment_name,size_t segment_max,size_t segment_size,int copy2syslog,int copy2stderr)
{
    abcdk_logger_t *ctx = NULL;

    assert(name != NULL);

    ctx = abcdk_heap_alloc(sizeof(abcdk_logger_t));
    if(!ctx)
        return NULL;
    
    ctx->fd = -1;
    ctx->segment_size = segment_size * 1024 * 1024;
    ctx->segment_max = segment_max;
    ctx->copy2stderr = copy2stderr;
    ctx->copy2syslog = copy2syslog;
    ctx->mask = 0xFFFFFFFF;
    ctx->buf = NULL;
    ctx->bufpos = 0;

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

int _abcdk_logger_segment(const char *src, const char *dst, int max)
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

void _abcdk_logger_flush(abcdk_logger_t *ctx)
{
    struct stat attr;
    int chk;

open_log_file:

    if (ctx->fd < 0)
    {
        abcdk_mkdir(ctx->name, 0666);
        ctx->fd = abcdk_open(ctx->name, 1, 0, 1);
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
        /*在末尾追加。*/
        lseek(ctx->fd, 0, SEEK_END);

        /*写，内部会保正写完。如果写不完，就是出错或没空间了。*/
        abcdk_write(ctx->fd, ctx->buf->pstrs[0], ctx->bufpos);
        ctx->bufpos = 0;

#if 0
        /*落盘，非常慢。*/
        fsync(ctx->fd);
#endif
    }
}

void abcdk_logger_puts(abcdk_logger_t *ctx, int type, const char *str)
{
    uint64_t ts = 0;
    struct tm tm;
    char name[NAME_MAX] = {0};
    int hdrlen = 0;
    size_t bufpos;
    char c;
    int chk;

    assert(ctx != NULL && ABCDK_LOGGER_TYPE_CHECK(type) && str != NULL);

    /*如果不需要记录，直接跳过。*/
    if (!(abcdk_atomic_load(&ctx->mask) & (1 << type)))
        return;

    /*加锁，确保每个线程写操作不被打断。*/
    abcdk_mutex_lock(ctx->locker, 1);

    /*没有缓存，申请一个。*/
    if(!ctx->buf)
    {
        ctx->buf = abcdk_object_alloc2(ABCDK_MIN((1 * 1024 * 1024UL),ctx->segment_size));
        if (!ctx->buf)
           return;
    }

    /*获取自然时间。*/
    ts = abcdk_time_clock2kind_with(CLOCK_REALTIME, 6);
    abcdk_time_sec2tm(&tm, ts / 1000000UL, 0);

    /*获进程或线程名称。*/
#ifndef __USE_GNU
    abcdk_proc_basename(name);
#else //__USE_GNU
    abcdk_thread_getname(pthread_self(),name);
#endif //__USE_GNU

    /*格式化行的头部：时间、PID、进程名字*/
    hdrlen = snprintf(ctx->buf->pstrs[0], ctx->buf->sizes[0], "%04d%02d%02d%02d%02d%02d.%06llu p%d %s: ",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ts % 1000000UL, getpid(), name);

next_line:

    if(!*str)
        goto final;

    /*从头部之后开始。*/
    ctx->bufpos = hdrlen;

next_char:

    if (*str)
    {
        /*读一个字符。*/
        c = *str++;

        /*回车符转成换行符。*/
        c = (c == '\r' ? '\n' : c);
    }
    else
    {
        /*未尾没有换行符，自动添加。*/
        c = '\n';
    }

    /*跳过所有空行。*/
    if (c == '\n' && ctx->bufpos == hdrlen)
        goto next_line;

    /*追加字符。*/
    ctx->buf->pstrs[0][ctx->bufpos++] = c;

    /*缓存已满时自动添加换行符。*/
    if (ctx->bufpos == ctx->buf->sizes[0] - 2)
        ctx->buf->pstrs[0][ctx->bufpos++] = c = '\n';

    /* 当前字符是换行时落盘，否则仅缓存。*/
    if (c != '\n')
        goto next_char;

    /*结束符。*/
    ctx->buf->pstrs[0][ctx->bufpos] = '\0';
    
    /*可能需要复制到stderr。*/
    if (ctx->copy2stderr)
        fprintf(stderr, "%s", ctx->buf->pstrs[0]);

    /*可能需要复制到syslog。*/
    if (ctx->copy2syslog)
        syslog(type, "p%d: %s", getpid(), ctx->buf->pstrs[0] + hdrlen);

    /*记录并清理缓存。*/
    _abcdk_logger_flush(ctx);

    /*下一行。*/
    goto next_line;

final:

    /*解锁，给其它线程写入的机会。*/
    abcdk_mutex_unlock(ctx->locker);
}


void abcdk_logger_vprintf(abcdk_logger_t *ctx, int type, const char *fmt, va_list ap)
{
    char buf[16000] = {0};

    assert(ctx != NULL && ABCDK_LOGGER_TYPE_CHECK(type) && fmt != NULL);

    vsnprintf(buf, 16000, fmt, ap);
    abcdk_logger_puts(ctx, type, buf);
}

void abcdk_logger_printf(abcdk_logger_t *ctx, int type, const char *fmt, ...)
{
    assert(ctx != NULL && ABCDK_LOGGER_TYPE_CHECK(type) && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_logger_vprintf(ctx, type, fmt, ap);
    va_end(ap);
}

void abcdk_logger_dump_siginfo(abcdk_logger_t *ctx, int type, siginfo_t *info)
{
    assert(ctx != NULL && ABCDK_LOGGER_TYPE_CHECK(type) && info != NULL);

    if (SI_USER == info->si_code)
        abcdk_logger_printf(ctx, type, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);
    else
        abcdk_logger_printf(ctx, type, "signo(%d),errno(%d),code(%d)\n", info->si_signo, info->si_errno, info->si_code);
}

void abcdk_logger_from_trace(void *opaque,int type, const char* fmt, va_list vp)
{
    abcdk_logger_t *ctx = (abcdk_logger_t *)opaque;

    abcdk_logger_vprintf(ctx,type,fmt,vp);
}
