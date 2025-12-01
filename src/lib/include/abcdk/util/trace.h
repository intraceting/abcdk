/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_TRACE_H
#define ABCDK_UTIL_TRACE_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"

__BEGIN_DECLS

/**简单轨迹输出接口.*/

/**输出回调函数.*/
typedef void (*abcdk_trace_output_cb)(void *opaque,int type, const char *str);

/**字符串输出.*/
void abcdk_trace_output(int type, const char *str, abcdk_trace_output_cb cb, void *opaque);

/**
 * 格式化输出重定向.
 * 
 * @note 如果未设置则使用默认的.
*/
void abcdk_trace_printf_redirect(abcdk_trace_output_cb cb,void *opaque);

/**格式化输出.*/
void abcdk_trace_vprintf(int type, const char* fmt, va_list ap);

/**格式化输出.*/
void abcdk_trace_printf(int type, const char* fmt,...);

/**信号格式化输出.*/
void abcdk_trace_printf_siginfo(int type, siginfo_t *info);

/**断言提示. */
#define ABCDK_TRACE_ASSERT(expr, tips) \
    ((expr) ? (void)(0) : ({abcdk_trace_printf(LOG_ERR,"%s(%d): %s\n",__FUNCTION__, __LINE__,#tips);abort(); }))

__END_DECLS


#endif //ABCDK_UTIL_TRACE_H