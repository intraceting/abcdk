/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_TERMIOS_H
#define ABCDK_UTIL_TERMIOS_H

#include "abcdk-util/general.h"

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
 * @param speed 波特率。
 * @param bits 数据位，5、6、7、8。默认：8。
 * @param parity 效验位，O 奇校验，E 偶校验，其它 无校验。默认：无。
 * @param stop 停止位，1或2。默认：1。
 * @param old 旧属性的指针，NULL(0)忽略。
 * 
 * @return 0 成功，-1 失败。
 * 
 */
int abcdk_tcattr_serial(int fd, int speed, int bits, int parity, int stop,struct termios *old);

__END_DECLS

#endif //ABCDK_UTIL_TERMIOS_H