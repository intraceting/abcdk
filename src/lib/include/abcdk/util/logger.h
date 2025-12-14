/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_UTIL_LOGGER_H
#define ABCDK_UTIL_LOGGER_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/util/object.h"
#include "abcdk/util/uri.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/trace.h"


__BEGIN_DECLS

/** 简单的记录器.*/
typedef struct _abcdk_logger abcdk_logger_t;

/** 日志类型.*/
typedef enum _abcdk_logger_type
{
    /** 错误.*/
    ABCDK_LOGGER_TYPE_ERROR = LOG_ERR,
#define ABCDK_LOGGER_TYPE_ERROR ABCDK_LOGGER_TYPE_ERROR

    /** 警告.*/
    ABCDK_LOGGER_TYPE_WARN = LOG_WARNING,
#define ABCDK_LOGGER_TYPE_WARN ABCDK_LOGGER_TYPE_WARN

    /** 重要.*/
    ABCDK_LOGGER_TYPE_INFO = LOG_INFO,
#define ABCDK_LOGGER_TYPE_INFO ABCDK_LOGGER_TYPE_INFO

    /** 调式.*/
    ABCDK_LOGGER_TYPE_DEBUG = LOG_DEBUG,
#define ABCDK_LOGGER_TYPE_DEBUG ABCDK_LOGGER_TYPE_DEBUG

    /** 最大值.*/
    ABCDK_LOGGER_TYPE_MAX = 32
#define ABCDK_LOGGER_TYPE_MAX ABCDK_LOGGER_TYPE_MAX
} abcdk_logger_type_t;


/**
 * 关闭.
 */
void abcdk_logger_close(abcdk_logger_t **ctx);

/**
 * 打开.
 *
 * @param [in] name 文件名(包括路径).
 * @param [in] segment_max 分段数量, 0 不分段.
 * @param [in] segment_size 分段大小(MB), 0 不分段.
 * @param [in] copy2syslog 复制到syslog. !0 是, 0 否.
 * @param [in] copy2stderr 复制到stderr. !0 是, 0 否.
 *
 */
abcdk_logger_t *abcdk_logger_open(const char *name, size_t segment_max, size_t segment_size, int copy2syslog, int copy2stderr);

/**打开.*/
abcdk_logger_t *abcdk_logger_open2(const char *path, const char *filename, size_t segment_max, size_t segment_size, int copy2syslog, int copy2stderr);

/**
 * 设置掩码.
 *
 * @param [in] type 类型.
 * @param [in] type,... 更多的类型, -1 结束.
 */
void abcdk_logger_mask(abcdk_logger_t *ctx, int type, ...);

/**
 * 输出字符串.
 *
 * @param [in] type 类型.
 * @param [in] str 字符串.
 *
 */
void abcdk_logger_output(abcdk_logger_t *ctx, int type, const char *str);

/**
 * 格式化输出.
 * 
 * @note 最大支持8000个字符(包括结束符).
 *
 * @param [in] type 类型.
 * @param [in] fmt 格式化字符串. @see vprintf
 * @param [in] ap 可变参数.
 *
 */
void abcdk_logger_vprintf(abcdk_logger_t *ctx, int type, const char *fmt, va_list ap);

/**
 * 格式化输出.
 * 
 * @note 最大支持8000个字符(包括结束符).
 * 
 */
void abcdk_logger_printf(abcdk_logger_t *ctx, int type, const char *fmt, ...);

/**
 * 代理.
 * 
 * @note 接收转发来的日志.
 */
void abcdk_logger_proxy(void *opaque, int type, const char *str);

__END_DECLS

#endif // ABCDK_UTIL_LOGGER_H
