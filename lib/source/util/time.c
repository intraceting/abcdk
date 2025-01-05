/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/time.h"


uint64_t abcdk_time_clock2kind(struct timespec *ts, uint8_t precision)
{
    uint64_t kind = 0;
    uint64_t p = 0;

    assert(ts);

    if (precision <= 9 && precision >= 1)
    {
        p = powl(10, precision);

        kind = ts->tv_sec * p;
        kind += ts->tv_nsec / (1000000000UL / p);
    }
    else
    {
        kind = ts->tv_sec;
    }

    return kind;
}

uint64_t abcdk_time_clock2kind_with(clockid_t id,uint8_t precision)
{
    int chk;
    struct timespec ts = {0};

    if (clock_gettime(id, &ts) != 0)
        return 0;

    return abcdk_time_clock2kind(&ts,precision);
}

struct tm *abcdk_time_local2utc(struct tm *dst, const struct tm *src, int reverse)
{
    time_t sec = 0;

    assert(dst && src);

    if (reverse)
    {
        sec = timegm((struct tm*)src);
        localtime_r(&sec,dst);
    }
    else
    {
        sec = timelocal((struct tm *)src);
        gmtime_r(&sec, dst);
    }

    return dst;
}

struct tm* abcdk_time_get(struct tm* tm,int utc)
{
    struct timespec ts = {0};
    int chk;

    assert(tm != NULL);

    chk = clock_gettime(CLOCK_REALTIME,&ts);

    return abcdk_time_sec2tm(tm,ts.tv_sec,utc);
}

struct tm *abcdk_time_sec2tm(struct tm *tm, time_t sec, int utc)
{
    assert(tm != NULL);

    return (utc ? gmtime_r(&sec, tm) : localtime_r(&sec, tm));
}

time_t abcdk_time_diff(struct tm *t1, struct tm *t0, int utc)
{
    time_t b = 0, e = 0;
    double d = 0.0;

    assert(t1 != NULL && t0 != NULL);

    if (utc)
    {
        b = timegm(t0);
        e = timegm(t1);
    }
    else
    {
        b = timelocal(t0);
        e = timelocal(t1);
    }

    d = difftime(e, b);

    return (time_t)d;
}

time_t abcdk_time_diff2(const char *t1, const char *t0, int utc)
{
    struct tm b = {0}, e = {0};

    strptime(t0,"%Y-%m-%dT%H:%M:%SZ",&b);
    strptime(t1,"%Y-%m-%dT%H:%M:%SZ",&e);

    return abcdk_time_diff(&e,&b,utc);
}

int _abcdk_time_format_init(void *opaque)
{
    pthread_key_t *key_p = (pthread_key_t*)opaque;

    pthread_key_create(key_p,abcdk_heap_free);
    return 0;
}

const char *abcdk_time_format(const char *fmt, const struct tm *tm, locale_t loc)
{
    static volatile int init_status = 0;
    static pthread_key_t key = -1;

    struct tm tmp;
    char *buf;
    int chk;

    if(!fmt)
        fmt = "%Y-%m-%d %H:%M:%S";

    /*如果未输入时间，则使用UTC时间。*/
    if(!tm)
    {
        abcdk_time_get(&tmp,1);
        return abcdk_time_format(fmt,&tmp,loc);
    }

    chk = abcdk_once(&init_status,_abcdk_time_format_init,&key);
    if(chk < 0)
        return NULL;

    buf = pthread_getspecific(key);
    if(!buf)
    {
        buf = abcdk_heap_alloc(PATH_MAX);
        if(!buf)
            return NULL;

        pthread_setspecific(key,buf);
    }
    
    if(loc)
        chk = strftime_l(buf,PATH_MAX,fmt,tm,loc);    
    else 
        chk = strftime(buf,PATH_MAX,fmt,tm);
    buf[chk] = '\0';

    return buf;
}

const char *abcdk_time_format_gmt(const struct tm *tm, locale_t loc)
{
    return abcdk_time_format("%a, %d %b %Y %H:%M:%S GMT", tm, loc);
}
