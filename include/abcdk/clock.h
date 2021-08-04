/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_CLOCK_H
#define ABCDK_CLOCK_H

#include "abcdk/general.h"
#include "abcdk/thread.h"

__BEGIN_DECLS

/**
 * 计时器-重置
 * 
 * @warning 线程内有效。
*/
void abcdk_clock_reset();

/**
 * 计时器-打点
 * 
 * @warning 线程内有效。
 * 
 * @param step !NULL(0) 返回两次打点间隔时长(微秒)。NULL(0) 忽略。
 * 
 * @return 计时器启动/重置到当前打点间隔时长(微秒)。
 *
*/
uint64_t abcdk_clock_dot(uint64_t *step);

/**
 * 计时器-打点
 * 
 * @warning 线程内有效。
 * 
 * @param dot !NULL(0) 返回计时器启动或重置到当前打点间隔时长(微秒)。NULL(0) 忽略。
 * 
 * @return 两次打点间隔时长(微秒)。
 * 
*/
uint64_t abcdk_clock_step(uint64_t *dot);

__END_DECLS

#endif //ABCDK_CLOCK_H