/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_SERIALPORT_H
#define ABCDK_UTIL_SERIALPORT_H

#include "abcdk/util/general.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/** 串口通讯对象。*/
typedef struct _abcdk_serialport abcdk_serialport_t;

/**
 * 串口通讯选项。
 */
typedef enum _abcdk_serialport_option
{
    /** 指令间隔(毫秒).*/
    ABCDK_SERIALPORT_OPT_INTERVAL = 1,
#define ABCDK_SERIALPORT_OPT_INTERVAL ABCDK_SERIALPORT_OPT_INTERVAL

} abcdk_serialport_option_t;

/** 销毁串口通讯对象。*/
void abcdk_serialport_destroy(abcdk_serialport_t **com);

/** 
 * 创建串口通讯对象。
 * 
 * @return !NULL(0) 成功(通讯对象指针)，NULL(0) 失败。
*/
abcdk_serialport_t *abcdk_serialport_create();

/**
 * 绑定句柄。
 * 
 * @warning 会强制给绑定名柄添加异步标志。
 * 
 * @return 旧的句柄。
*/
int abcdk_serialport_attach(abcdk_serialport_t *ctx,int fd);

/**
 * 分离句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_serialport_detach(abcdk_serialport_t *ctx);

/**
 * 设置选项。
 * 
 * @param 0 成功，-1 失败。
*/
int abcdk_serialport_set_option(abcdk_serialport_t *ctx,int opt,...);

/**
 * 获取选项。
 * 
 * @param 0 成功，-1 失败。
*/
int abcdk_serialport_get_option(abcdk_serialport_t *ctx,int opt,...);

/**
 * 传输数据。
 * 
 * @param [in] out 输出数据，NULL(0) 忽略。
 * @param [in] outlen 输出数据长度，<= 0 忽略。
 * @param [out] in 输入数据，NULL(0) 忽略。
 * @param [in] inlen 输入数据长度，<= 0 忽略。
 * @param [in] timeout 超时(毫秒)。
 * @param [in] magic 起始码，NULL(0) 忽略。注：仅对输入有效。
 * @param [in] mglen 起始码长度，<= 0 忽略。注：仅对输入有效。
 * 
 * @param 0 成功，-1 失败(或超时)。
*/
int abcdk_serialport_transfer(abcdk_serialport_t *ctx, const void *out, size_t outlen, void *in, size_t inlen,
                              time_t timeout, const void *magic, size_t mglen);

__END_DECLS

#endif //ABCDK_UTIL_SERIALPORT_H