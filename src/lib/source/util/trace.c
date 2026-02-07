/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/trace.h"

static void *g_trace_dump_cb_opaque = NULL;
static abcdk_trace_dump_cb g_trace_dump_cb_func = NULL;

void abcdk_trace_redirect(abcdk_trace_dump_cb cb, void *opaque)
{
    g_trace_dump_cb_func = cb;
    g_trace_dump_cb_opaque = opaque;
}

void abcdk_trace_output(int type, const char *str)
{
    char buf[8000] = {0};
    uint64_t ts = 0;
    struct tm tm;
    char name[NAME_MAX] = {0};
    int hdrlen = 0;
    size_t bufpos;
    char c;
    int chk;

    assert(str != NULL);

    /*获取自然时间.*/
    ts = abcdk_time_realtime(6);
    abcdk_time_sec2tm(&tm, ts / 1000000UL, 0);

    /*获线程名称.*/
    abcdk_thread_getname(pthread_self(), name);

    /*格式化行的头部: TIME, PID, TYPE, NAME*/
    hdrlen = snprintf(buf, sizeof(buf), "[%04d%02d%02dT%02d%02d%02d.%06llu][%d][%d][%s]: ",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ts % 1000000UL, getpid(), type, name);

next_line:

    if (!*str)
        return;

    /*从头部之后开始.*/
    bufpos = hdrlen;

next_char:

    if (*str)
    {
        /*读一个字符.*/
        c = *str++;

        /*回车符转成换行符.*/
        c = (c == '\r' ? '\n' : c);
    }
    else
    {
        /*未尾没有换行符, 自动添加.*/
        c = '\n';
    }

    /*跳过所有空行.*/
    if (c == '\n' && bufpos == hdrlen)
        goto next_line;

    /*追加字符.*/
    buf[bufpos++] = c;

    /*缓存已满时自动添加换行符.*/
    if (bufpos == sizeof(buf) - 2)
        buf[bufpos++] = c = '\n';

    /* 当前字符是换行时落盘, 否则仅缓存.*/
    if (c != '\n')
        goto next_char;

    /*结束符.*/
    buf[bufpos] = '\0';

    if (g_trace_dump_cb_func)
        g_trace_dump_cb_func(g_trace_dump_cb_opaque, type, buf);
    else
        fprintf(stderr, "%s", buf);

    /*下一行.*/
    goto next_line;
}

void abcdk_trace_vprintf(int type, const char *fmt, va_list ap)
{
    char buf[8000] = {0};

    assert(fmt != NULL);

    vsnprintf(buf, 8000, fmt, ap);

    abcdk_trace_output(type, buf);
}

void abcdk_trace_printf(int type, const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_vprintf(type, fmt, vp);
    va_end(vp);
}

void abcdk_trace_siginfo(int type, siginfo_t *info)
{
    assert(info != NULL);

    if (SI_USER == info->si_code)
        abcdk_trace_printf(type, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);
    else
        abcdk_trace_printf(type, "signo(%d),errno(%d),code(%d)\n", info->si_signo, info->si_errno, info->si_code);
}