/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_UTIL_GENERAL_H
#define ABCDK_UTIL_GENERAL_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"

__BEGIN_DECLS

/**
 * 数值对齐。
 *
 * @param align 对齐量。0,1是等价的。
 */
size_t abcdk_align(size_t size, size_t align);

/**
 * 二进制转十六进制。
 *
 * @param dst 十六进制数据的指针。可用空间至少是二进制数据长度的两倍。
 * @param src 二进制数的指针。
 * @param size 二进制数据的长度。
 * @param ABC 0 小写，!0 大写。
 *
 * @return !NULL(0) 成功(十六进制数据的指针)，NULL(0) 失败。
 */
char *abcdk_bin2hex(char *dst, const void *src, size_t size, int ABC);

/**
 * 十六进制转二进制。
 *
 * @param dst 二进制数据的指针。可用空间至少是十六进制数据长度的二分之一。
 * @param src 十六进制数的指针。
 * @param size 十六进制数据的长度。
 *
 * @return !NULL(0) 成功(二进制数据的指针)，NULL(0) 失败。
 */
void *abcdk_hex2bin(void *dst, const char *src, size_t size);

/**
 * 匹配环境变量。
 *
 * @code
 * name<delim>value
 * name<delim> value
 * @endcode
 *
 * @param [in] line 行数据。
 * @param [in] name 变量名称。
 * @param [in] delim 分割字符。
 *
 * @return !NULL(0) 成功(值的指针)，NULL(0) 失败。
 *
 */
const char *abcdk_match_env(const char *line, const char *name, uint8_t delim);

/**
 * 内存复制(1D)。
 *
 * @param [out] dst 目标地址。
 * @param [in] dst_offset 目标地址偏移量。
 * @param [in] src 源地址。
 * @param [in] src_offset 源地址偏移量。
 * @param [in] count 数量。
 *
 */
void abcdk_memcpy_1d(void *dst, size_t dst_offset, const void *src, size_t src_offset, size_t count);

/**
 * 内存复制(2D)。
 *
 * @param [out] dst 目标地址。
 * @param [in] dst_pitch 目标地址X方向步长(字节)。
 * @param [in] dst_x_bytes 目标地址X方向偏移量(字节)。
 * @param [in] dst_y 目标地址Y方向偏移量。
 * @param [in] src 源地址。
 * @param [in] src_pitch 源地址X方向步长(字节)。
 * @param [in] src_x_bytes 源地址X方向偏移量(字节)。
 * @param [in] src_y 源地址Y方向偏移量。
 * @param [in] roi_width_bytes 复制X方向宽度(字节)。
 * @param [in] roi_height 复制Y方向高度。
 *
 */
void abcdk_memcpy_2d(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y,
                     const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y,
                     size_t roi_width_bytes, size_t roi_height);

/**
 * 等待进程结束。
 *
 * @param [in] pid 进程PID或进程组ID。@see waitpid().
 * @param [in] options 选项。@see waitpid().
 * @param [out] exitcode 状态码。
 * @param [out] sigcode 信号。
 *
 * @return > 0 PID(已结束的进程PID)，0 无，< 0 无(PID无效)。
 */
pid_t abcdk_waitpid(pid_t pid, int options, int *exitcode, int *sigcode);

/**
 * 获取线程的进程PID。
 */
pid_t abcdk_gettid();

/**
 * 自增编号。
 *
 * @note 从1开始。
 * @note 全局的，多线程之间编号不保证连续性。
 */
uint64_t abcdk_sequence_num();

__END_DECLS

#endif // ABCDK_UTIL_GENERAL_H