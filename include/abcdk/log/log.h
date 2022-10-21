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
 * @warning 如果在其它接调用之后才进行初始化，将使用参数默认值。
 * @warning 如果未指定收货地址，将尝试从环境变量(ABCDK_LOG_CONSIGNEE)中获取。
 * @code
 * IPV4:PORT
 * IPV6,PORT
 * [IPV6]:PORT
 * @endcode
 *
 * @param [in] consignee 收货地址。
 * @param [in] service 服务ID。默认：1
 * @param [in] copy2syslog 是否复制到syslog。默认：否
 *
*/
void abcdk_log_open(const char *consignee,uint16_t service, int copy2syslog);

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
