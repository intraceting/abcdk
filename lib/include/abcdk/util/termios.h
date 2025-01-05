/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_TERMIOS_H
#define ABCDK_UTIL_TERMIOS_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 设置终端句柄属性。
 * 
 * @param now 新属性的指针，NULL(0)忽略。
 * @param old 旧属性的指针，NULL(0)忽略。
 * 
 * @return 0 成功，-1 失败。
 * 
*/
int abcdk_tcattr_option(int fd, const struct termios *now, struct termios *old);

/**
 * 终端禁用行缓冲和回显。
 * 
 * @param old 旧属性的指针，NULL(0)忽略。
 * 
 * @return 0 成功，-1 失败。
 * 
 */
int abcdk_tcattr_cbreak(int fd,struct termios *old);

/** 设置串口参数
 * 
 * @param baudrate 波特率。
 * @param bits 数据位，5、6、7、8。默认：8。
 * @param parity 效验位，1 奇校验，2 偶校验，其它 无校验。默认：无校验。
 * @param stop 停止位，1位或2位。默认：1位。
 * @param old 旧属性的指针，NULL(0)忽略。
 * 
 * @return 0 成功，-1 失败。
 * 
 */
int abcdk_tcattr_serial(int fd, int baudrate, int bits, int parity, int stop,struct termios *old);

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
int abcdk_tcattr_transfer(int fd, const void *out, size_t outlen, void *in, size_t inlen,
                          time_t timeout, const void *magic, size_t mglen);

__END_DECLS

#endif //ABCDK_UTIL_TERMIOS_H