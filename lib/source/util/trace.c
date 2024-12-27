/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/trace.h"

static void _abcdk_trace_log_syslog(void *opaque,int type, const char *fmt, va_list vp)
{
    vsyslog(type, fmt, vp);
}

/*全局的*/
static void *g_trace_log_opaque = NULL;
static abcdk_trace_log_cb g_trace_log_func = _abcdk_trace_log_syslog;

void abcdk_trace_set_log(abcdk_trace_log_cb cb,void *opaque)
{
    assert(cb);

    g_trace_log_func = cb;
    g_trace_log_opaque = opaque;
}

void abcdk_trace_voutput(int type, const char* fmt, va_list vp)
{
    assert(ABCDK_TRACE_TYPE_CHECK(type) && fmt != NULL);

    if(g_trace_log_func)
        g_trace_log_func(g_trace_log_opaque,type, fmt, vp);
}

void abcdk_trace_output(int type, const char *fmt, ...)
{
    assert(ABCDK_TRACE_TYPE_CHECK(type) && fmt != NULL);

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_voutput(type, fmt, vp);
    va_end(vp);
}

void abcdk_trace_output_siginfo(int type, siginfo_t *info)
{
    assert(ABCDK_TRACE_TYPE_CHECK(type) && info != NULL);

    if (SI_USER == info->si_code)
        abcdk_trace_output(type, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);
    else
        abcdk_trace_output(type, "signo(%d),errno(%d),code(%d)\n", info->si_signo, info->si_errno, info->si_code);
}