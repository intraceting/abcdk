/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_LOG_LOG_H
#define ABCDK_LOG_LOG_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/shell/file.h"

__BEGIN_DECLS

/** 日志类型。*/
typedef enum _abcdk_log_type
{
    /** 错误。*/
    ABCDK_LOG_ERROR = LOG_ERR,
#define ABCDK_LOG_ERROR ABCDK_LOG_ERROR

    /** 警告。*/
    ABCDK_LOG_WARN = LOG_WARNING,
#define ABCDK_LOG_WARN ABCDK_LOG_WARN

    /** 重要。*/
    ABCDK_LOG_INFO = LOG_INFO,
#define ABCDK_LOG_INFO ABCDK_LOG_INFO

    /** 调式。*/
    ABCDK_LOG_DEBUG = LOG_DEBUG,
#define ABCDK_LOG_DEBUG ABCDK_LOG_DEBUG

    /** 最大值。*/
    ABCDK_LOG_MAX = 32
#define ABCDK_LOG_MAX ABCDK_LOG_MAX
} abcdk_log_type_t;

/**
 * 初始化。
 * 
 * @note 分段文件名仅支持一个数值格式控制符。
 *
 * @param [in] name 文件名(包括路径)。
 * @param [in] segment_name 分段文件名(包括路径)，NULL(0) 不分段。
 * @param [in] segment_max 分段数量，0 不分段。
 * @param [in] segment_size 分段大小(MB)，0 不分段。
 * @param [in] copy2syslog 复制到syslog。!0 是，0 否。
 * @param [in] copy2stderr 复制到stderr。!0 是，0 否。
 * 
 * @code
 * //打开日志。
 * abcdk_log_open("/tmp/abcdk-log/abcdk.log","abcdk.%d.log", 10, 10, 0, 1);
 * @endcode
 *
*/
void abcdk_log_open(const char *name,const char *segment_name,size_t segment_max,size_t segment_size, int copy2syslog,int copy2stderr);

/**
 * 设置掩码。
 *
 * @param [in] type 类型。
 * @param [in] type,... 更多的类型，-1 结束。
 */
void abcdk_log_mask(int type, ...);

/**
 * 格式化输出。
 *
 * @param [in] type 类型。
 * @param [in] fmt 格式化字符串。@see vprintf
 * @param [in] ap 可变参数。
 *
 */
void abcdk_log_vprintf(int type, const char *fmt, va_list ap);

/**
 * 格式化输出。
 */
void abcdk_log_printf(int type, const char *fmt, ...);

__END_DECLS

#endif // ABCDK_LOG_LOG_H
