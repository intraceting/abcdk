/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/time.h"

uint64_t abcdk_time_clock2kind(struct timespec *ts, uint8_t precision)
{
    static const uint64_t powers[] = {
        1ULL,         // 0
        10ULL,        // 1
        100ULL,       // 2
        1000ULL,      // 3
        10000ULL,     // 4
        100000ULL,    // 5
        1000000ULL,   // 6
        10000000ULL,  // 7
        100000000ULL, // 8
        1000000000ULL // 9
    };

    assert(ts);

    if (precision >= 1 && precision <= 9)
    {
        uint64_t p = powers[precision];
        return ((uint64_t)ts->tv_sec * p) + ((uint64_t)ts->tv_nsec * p / 1000000000ULL);
    }

    return (uint64_t)ts->tv_sec;
}

uint64_t abcdk_time_clock2kind_with(clockid_t id,uint8_t precision)
{
    int chk;
    struct timespec ts = {0};

    if (clock_gettime(id, &ts) != 0)
        return 0;

    return abcdk_time_clock2kind(&ts,precision);
}

uint64_t abcdk_time_realtime(uint8_t precision)
{
    return abcdk_time_clock2kind_with(CLOCK_REALTIME,precision);
}

uint64_t abcdk_time_systime(uint8_t precision)
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC,precision);
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

time_t abcdk_time_tm2sec(struct tm *tm, int utc)
{
    assert(tm != NULL);

    return (utc ? timegm(tm) : mktime(tm));
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
        b = mktime(t0);
        e = mktime(t1);
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

static pthread_once_t _abcdk_time_format_key_init_status = PTHREAD_ONCE_INIT;
static pthread_key_t _abcdk_time_format_key = 0xffffffff;

void _abcdk_time_format_key_init()
{
    pthread_key_create(&_abcdk_time_format_key,abcdk_heap_free);
}

const char *abcdk_time_format(const char *fmt, const struct tm *tm, locale_t loc)
{
    struct tm tmp;
    char *buf;
    int chk;

    if(!fmt)
        fmt = "%Y-%m-%d %H:%M:%S";

    /*如果未输入时间, 则使用UTC时间.*/
    if(!tm)
    {
        abcdk_time_get(&tmp,1);
        return abcdk_time_format(fmt,&tmp,loc);
    }

    /*初始化一次.*/
    chk = pthread_once(&_abcdk_time_format_key_init_status,_abcdk_time_format_key_init);
    assert(chk == 0);

    buf = pthread_getspecific(_abcdk_time_format_key);
    if(!buf)
    {
        buf = abcdk_heap_alloc(PATH_MAX);
        if(!buf)
            return NULL;

        pthread_setspecific(_abcdk_time_format_key,buf);
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
