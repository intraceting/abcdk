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


/** 轨迹类型。*/
typedef enum _abcdk_trace_type
{
    /** 错误。*/
    ABCDK_TRACE_ERROR = LOG_ERR,
#define ABCDK_TRACE_ERROR ABCDK_TRACE_ERROR

    /** 警告。*/
    ABCDK_TRACE_WARN = LOG_WARNING,
#define ABCDK_TRACE_WARN ABCDK_TRACE_WARN

    /** 重要。*/
    ABCDK_TRACE_INFO = LOG_INFO,
#define ABCDK_TRACE_INFO ABCDK_TRACE_INFO

    /** 调式。*/
    ABCDK_TRACE_DEBUG = LOG_DEBUG,
#define ABCDK_TRACE_DEBUG ABCDK_TRACE_DEBUG

    /** 最大值。*/
    ABCDK_TRACE_MAX = 32
#define ABCDK_TRACE_MAX ABCDK_TRACE_MAX
} abcdk_trace_type_t;

/** 检查轨迹类型。*/
#define ABCDK_TRACE_TYPE_CHECK(t) ((t) >= ABCDK_TRACE_ERROR && (t) < ABCDK_TRACE_MAX)

/** 输出回调函数。*/
typedef void (*abcdk_trace_output_cb)(void *opaque,int type, const char *str);

/**字符串输出。*/
void abcdk_trace_output(int type, const char *str, abcdk_trace_output_cb cb, void *opaque);

/**
 * 设置格式化输出回调函数。
 * 
 * @note 如果未设置则使用默认的。
*/
void abcdk_trace_printf_set_callback(abcdk_trace_output_cb cb,void *opaque);

/**格式化输出。*/
void abcdk_trace_vprintf(int type, const char* fmt, va_list ap);

/**格式化输出。*/
void abcdk_trace_printf(int type, const char* fmt,...);

/**信号格式化输出。*/
void abcdk_trace_printf_siginfo(int type, siginfo_t *info);

__END_DECLS


#endif //ABCDK_UTIL_TRACE_H