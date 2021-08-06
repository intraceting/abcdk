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

__END_DECLS

#endif //ABCDK_UTIL_TERMIOS_H