/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_LOG_LOG_H
#define ABCDK_LOG_LOG_H

#include "util/general.h"

__BEGIN_DECLS

/**
 * 收件人环境变量名字。
 *
 * @code
 * IPV4:PORT
 * IPV6,PORT
 * [IPV6]:PORT
 * @endcode
 */
#define ABCDK_LOG_CONSIGNEE "ABCDK_LOG_CONSIGNEE"

/** 日志类型。*/
typedef enum _abcdk_log_type
{
    /** 错误。*/
    ABCDK_LOG_ERROR = 0,
#define ABCDK_LOG_ERROR ABCDK_LOG_ERROR

    /** 警告。*/
    ABCDK_LOG_WARN = 1,
#define ABCDK_LOG_WARN ABCDK_LOG_WARN

    /** 重要。*/
    ABCDK_LOG_INFO = 2,
#define ABCDK_LOG_INFO ABCDK_LOG_INFO

    /** 调式。*/
    ABCDK_LOG_DEBUG = 3,
#define ABCDK_LOG_DEBUG ABCDK_LOG_DEBUG

    /** 最大值。*/
    ABCDK_LOG_MAX = 32
#define ABCDK_LOG_MAX ABCDK_LOG_MAX
} abcdk_log_type_t;

/**
 * 关闭。
*/
void abcdk_log_close();

/**
 * 初始化。
 *
 * @warning 如果在其它接调用之后才进行初始化，将不会起作用。
 *
 * @param [in] consignee 收货地址。默认：本机。
 *
 */
void abcdk_log_open(const char *consignee);

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
