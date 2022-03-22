/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_LOG_H
#define ABCDK_UTIL_LOG_H

#include "util/general.h"
#include "util/thread.h"

__BEGIN_DECLS

/**
 * 日志初始化。
 * 
 * 只能执行一次。
 * 
 * @param ident NULL(0) 进程名做为标识，!NULL(0) 自定义标识。
 * @param level 记录级别。LOG_*宏定义在syslog.h文件中。
 * @param copy2stderr 0 仅记录，!0 复制到stderr。
 * 
 */
void abcdk_log_open(const char *ident,int level,int copy2stderr);

/**
 * 格式化输出日志。
 * 
 * @note 自动获取线程名字添加到行首。
 * 
 * @param priority 优先级。LOG_*宏定义在syslog.h文件中。
 * @param fmt 格式化字符串。最长支持2032字节。
 * 
*/
void abcdk_log_vprintf(int priority,const char *fmt,va_list ap);

/**
 * 格式化输出日志。
*/
void abcdk_log_printf(int priority,const char *fmt,...);


__END_DECLS

#endif //ABCDK_UTIL_LOG_H
