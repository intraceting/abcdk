/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/log/log.h"
#include "abcdk/util/object.h"
#include "abcdk/util/uri.h"
#include "abcdk/shell/proc.h"

/** 日志接口。*/
typedef struct _abcdk_log
{
    /** 
     * 状态。
     * 
     * 0：未初始化。
     * 1：正在。
     * 2：完成。
     * 
    */
    volatile int status;

    /** 线程KEY。*/
    pthread_key_t buf_key;

    /** 文件锁。*/
    abcdk_mutex_t mutex;

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
     * !0： 是，
     * 0： 否。
    */
    int copy2stderr;

    /** 
     * 复制到syslog。
     * !0： 是，
     * 0： 否。
    */
    int copy2syslog;

    /** 
     * 掩码。
     * 
     * 见syslog。
    */
    volatile uint32_t mask;

} abcdk_log_t;

abcdk_log_t *_abcdk_log_ctx()
{
    static abcdk_log_t ctx = {0};
    return &ctx;
}

void _abcdk_log_buf_destroy(void *opaque)
{
    abcdk_object_t *buf_p = (abcdk_object_t *)opaque;
    abcdk_object_unref(&buf_p);
}

void _abcdk_log_close()
{
    abcdk_log_t *ctx = NULL;

    ctx = _abcdk_log_ctx();

    abcdk_closep(&ctx->fd);
    abcdk_mutex_unlock(&ctx->mutex);
    pthread_key_delete(ctx->buf_key);
}

void abcdk_log_open(const char *name,const char *segment_name,size_t segment_max,size_t segment_size,int copy2syslog,int copy2stderr)
{
    abcdk_log_t *ctx = NULL;

    assert(name != NULL);

    ctx = _abcdk_log_ctx();

    /*初始化一次。*/
    if(!abcdk_atomic_compare_and_swap(&ctx->status,0,1))
        return;

    /*注册关闭函数。*/
    atexit(_abcdk_log_close);

    pthread_key_create(&ctx->buf_key, _abcdk_log_buf_destroy);
    abcdk_mutex_init2(&ctx->mutex,0);
    ctx->fd = -1;

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

    ctx->segment_size = segment_size * 1024 * 1024;
    ctx->segment_max = segment_max;
    ctx->copy2stderr = copy2stderr;
    ctx->copy2syslog = copy2syslog;
    ctx->mask = 0xFFFFFFFF;

    /*完成。*/
    abcdk_atomic_store(&ctx->status,2);

}

void abcdk_log_mask(int type, ...)
{
    abcdk_log_t *ctx = NULL;
    uint32_t mask = 0;

    ctx = _abcdk_log_ctx();

    if (!abcdk_atomic_compare(&ctx->status, 2))
        return;

    va_list vaptr;
    va_start(vaptr, type);
    for (;;)
    {
        if (type < ABCDK_LOG_ERROR || type >= ABCDK_LOG_MAX)
            break;

        mask |= (1 << type);

        /*遍历后续的。*/
        type = va_arg(vaptr, int);
    }
    va_end(vaptr);

    /*覆盖现有的。*/
    abcdk_atomic_store(&ctx->mask, mask);
}

abcdk_object_t *_abcdk_log_get_buffer()
{
    abcdk_log_t *ctx = NULL;
    abcdk_object_t *buf_p = NULL;

    ctx = _abcdk_log_ctx();

    buf_p = pthread_getspecific(ctx->buf_key);
    if (!buf_p)
    {
        /*没有缓存，申请一个。*/
        buf_p = abcdk_object_alloc2(ABCDK_MIN((16 * 1024 * 1024),ctx->segment_size));
        if (buf_p)
            pthread_setspecific(ctx->buf_key, buf_p);
    }

    return buf_p;
}

void abcdk_log_vprintf(int type, const char *fmt, va_list ap)
{
    abcdk_log_t *ctx = NULL;
    abcdk_object_t *buf_p = NULL;
    uint64_t ts = 0;
    struct tm tm;
    char name[17] = {0};
    int len,len2;
    struct stat attr;
    int chk;

    assert(fmt != NULL);

    ctx = _abcdk_log_ctx();

    if (!abcdk_atomic_compare(&ctx->status, 2))
        return;

    /*不知道什么类型，直接跳过。*/
    if (type < ABCDK_LOG_ERROR || type >= ABCDK_LOG_MAX)
        return;

    /*如果不需要记录，直接跳过。*/
    if (!(abcdk_atomic_load(&ctx->mask) & (1 << type)))
        return;

    /*获取自然时间。*/
    ts = abcdk_time_clock2kind_with(CLOCK_REALTIME, 6);
    abcdk_time_sec2tm(&tm,ts/1000000,0);

    /*获取线程名称。*/
    abcdk_thread_getname(pthread_self(),name);
    
    /*获取缓存。*/
    buf_p = _abcdk_log_get_buffer();
    if (!buf_p)
        return;

    /*格式化。*/
    len = snprintf(buf_p->pstrs[0], buf_p->sizes[0], "%04d%02d%02d%02d%02d%02d.%06lu p%d %s: ",
                   tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ts % 1000000, getpid(), name);
    len2 = vsnprintf(buf_p->pstrs[0] + len, buf_p->sizes[0] - len, fmt, ap);

    /*可能还需要追加换行符。*/
    if (buf_p->pstrs[0][len + len2 - 1] != '\n')
        buf_p->pstrs[0][len + len2++] = '\n'; //长度会+1。

    /*结束符。*/
    buf_p->pstrs[0][len + len2] = '\0';
    

abcdk_mutex_lock(&ctx->mutex,1);

    /*可能需要复制到stderr。*/
    if (ctx->copy2stderr)
        fprintf(stderr, "%s", buf_p->pstrs[0]);

    /*可能需要复制到syslog。*/
    if (ctx->copy2syslog)
        syslog(type, "p%d %s: %s", getpid(), name, buf_p->pstrs[0] + len);

    /*写文件。*/
open_log:

    if (ctx->fd < 0)
    {
        abcdk_mkdir(ctx->name,0666);
        ctx->fd = abcdk_open(ctx->name, 1, 0, 1);
    }

    if (ctx->fd < 0)
        goto final;

    chk = fstat(ctx->fd, &attr);
    if (chk != 0)
        goto final;

    if (ctx->segment_name[0] &&
        ctx->segment_max > 0 &&
        ctx->segment_size > 0 &&
        attr.st_size >= ctx->segment_size)
    {
        abcdk_closep(&ctx->fd);
        abcdk_file_segment(ctx->name, ctx->segment_name, ctx->segment_max);
        goto open_log;
    }
    else
    {   
        /*在末尾追加。*/
        lseek(ctx->fd,0,SEEK_END);

        /*写，内部会保正写完。如果写不完，就是出错或没空间了。*/
        abcdk_write(ctx->fd,buf_p->pstrs[0],len+len2);
        
#if 0
        /*落盘，非常慢。*/
        fsync(ctx->fd);
#endif
    }

final:

    abcdk_mutex_unlock(&ctx->mutex);
}

void abcdk_log_printf(int type, const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(type, fmt, ap);
    va_end(ap);
}