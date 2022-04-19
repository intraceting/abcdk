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
    char buf[2048] = {0};

    assert(fmt);

    /*获取线程名称。*/
    abcdk_thread_getname(buf);

    /*
     * 是否接接线程名称，由以下策略决定。
     * 1：如果线程已经命名，则拼接在最前面。
     * 2：如果线程未命名，则不拼接。
    */
    snprintf(buf + strlen(buf), 2032, "%s%s", (*buf ? ": " : ""), fmt);

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