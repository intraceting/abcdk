/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "util/log.h"

void abcdk_log_open(const char *ident, int level, int copy2stderr)
{
    openlog(ident, LOG_CONS | LOG_PID | (copy2stderr ? LOG_PERROR : 0), LOG_USER);

    setlogmask(LOG_UPTO(level));
}

void abcdk_log_vprintf(int priority, const char *fmt, va_list ap)
{
    char buf[4096] = {0};

    assert(fmt);

    /*获取线程名称。*/
    abcdk_thread_getname(buf);

    /*如果线程已命名，则拼接在行首。*/
    snprintf(buf + strlen(buf), 4080, "%s%s", (*buf ? ": " : ""), fmt);

    vsyslog(priority,buf,ap);
}

void abcdk_log_printf(int priority, const char *fmt, ...)
{
    assert(fmt);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(priority,fmt, ap);
    va_end(ap);
}