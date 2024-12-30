/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_LOG_LOGGER_H
#define ABCDK_LOG_LOGGER_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/util/object.h"
#include "abcdk/util/uri.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/trace.h"


__BEGIN_DECLS

/** 简单的记录器。*/
typedef struct _abcdk_logger abcdk_logger_t;

/** 日志类型。*/
typedef enum _abcdk_logger_type
{
    /** 错误。*/
    ABCDK_LOGGER_ERROR = LOG_ERR,
#define ABCDK_LOGGER_ERROR ABCDK_LOGGER_ERROR

    /** 警告。*/
    ABCDK_LOGGER_WARN = LOG_WARNING,
#define ABCDK_LOGGER_WARN ABCDK_LOGGER_WARN

    /** 重要。*/
    ABCDK_LOGGER_INFO = LOG_INFO,
#define ABCDK_LOGGER_INFO ABCDK_LOGGER_INFO

    /** 调式。*/
    ABCDK_LOGGER_DEBUG = LOG_DEBUG,
#define ABCDK_LOGGER_DEBUG ABCDK_LOGGER_DEBUG

    /** 最大值。*/
    ABCDK_LOGGER_MAX = 32
#define ABCDK_LOGGER_MAX ABCDK_LOGGER_MAX
} abcdk_logger_type_t;

/** 检查日志类型。*/
#define ABCDK_LOGGER_TYPE_CHECK(t) ((t) >= ABCDK_LOGGER_ERROR && (t) < ABCDK_LOGGER_MAX)

/**
 * 关闭。
 */
void abcdk_logger_close(abcdk_logger_t **ctx);

/**
 * 打开。
 *
 * @code
 * //打开日志。
 * abcdk_logger_open("/tmp/abcdk-log/abcdk.log","abcdk.%d.log", 10, 10, 0, 1);
 * @endcode
 *
 * @param [in] name 文件名(包括路径)。
 * @param [in] segment_name 分段文件名(包括路径)，NULL(0) 不分段。注：分段文件名仅支持一个数值格式控制符。如：%d 。
 * @param [in] segment_max 分段数量，0 不分段。
 * @param [in] segment_size 分段大小(MB)，0 不分段。
 * @param [in] copy2syslog 复制到syslog。!0 是，0 否。
 * @param [in] copy2stderr 复制到stderr。!0 是，0 否。
 *
 */
abcdk_logger_t *abcdk_logger_open(const char *name, const char *segment_name, size_t segment_max, size_t segment_size, int copy2syslog, int copy2stderr);

/**
 * 打开。
 * 
 * @param [in] path 路径。
 * 
*/
abcdk_logger_t *abcdk_logger_open2(const char *path,const char *name, const char *segment_name, size_t segment_max, size_t segment_size, int copy2syslog, int copy2stderr);

/**
 * 设置掩码。
 *
 * @param [in] type 类型。
 * @param [in] type,... 更多的类型，-1 结束。
 */
void abcdk_logger_mask(abcdk_logger_t *ctx, int type, ...);

/**
 * 输出字符串。
 *
 * @param [in] type 类型。
 * @param [in] str 字符串。
 *
 */
void abcdk_logger_output(abcdk_logger_t *ctx, int type, const char *str);

/**
 * 格式化输出。
 * 
 * @note 最大支持8000个字符(包括结束符)。
 *
 * @param [in] type 类型。
 * @param [in] fmt 格式化字符串。@see vprintf
 * @param [in] ap 可变参数。
 *
 */
void abcdk_logger_vprintf(abcdk_logger_t *ctx, int type, const char *fmt, va_list ap);

/**
 * 格式化输出。
 * 
 * @note 最大支持8000个字符(包括结束符)。
 * 
 */
void abcdk_logger_printf(abcdk_logger_t *ctx, int type, const char *fmt, ...);


/**
 * 轨迹日志函数。
*/
void abcdk_logger_from_trace(void *opaque,int type, const char* str);


__END_DECLS

#endif // ABCDK_LOG_LOGGER_H
