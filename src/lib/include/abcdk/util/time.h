/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_TIME_H
#define ABCDK_UTIL_TIME_H

#include "abcdk/util/defs.h"
#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"

__BEGIN_DECLS

/**
 * 时间戳整形化。
 * 
 * 当精度为纳秒时，公元2444年之前或者时长544年之内有效。
 * 
 * @param precision 精度。0～9。
*/
uint64_t abcdk_time_clock2kind(struct timespec* ts,uint8_t precision);

/**
 * 获取时间戳并整形化。
 * 
 * @param id CLOCK_* in time.h
*/
uint64_t abcdk_time_clock2kind_with(clockid_t id ,uint8_t precision);

/**
 * 获取时间戳(自然时间,UTC)并整形化。
*/
uint64_t abcdk_time_realtime(uint8_t precision);

/**
 * 获取时间戳(系统时间)并整形化。
*/
uint64_t abcdk_time_systime(uint8_t precision);


/**
 * 获取自然时间。
 * 
 * @param utc 0 获取本地，!0 获取国际。
*/
struct tm* abcdk_time_get(struct tm* tm,int utc);

/**
 * 秒转自然时间。
 * 
 * @param utc 0 转本地，!0 转国际。
*/
struct tm* abcdk_time_sec2tm(struct tm* tm,time_t sec,int utc);

/**
 * 自然时间转秒。
 * 
 * @param utc 0 转本地，!0 转国际。
*/
time_t abcdk_time_tm2sec(struct tm* tm,int utc);

/**
 * 计算时间差。
 * 
 * @param utc 0 本地时间，!0 国际时间。
 * 
 * @return 返回相差的时长(秒)。
*/
time_t abcdk_time_diff(struct tm *t1, struct tm *t0, int utc);

/**
 * 计算时间差。
 * 
 * @note yyyy-mm-ddThh:MM:SSZ
 * 
 * @param utc 0 本地时间，!0 国际时间。
 * 
 * @return 返回相差的时长(秒)。
*/
time_t abcdk_time_diff2(const char *t1, const char *t0, int utc);

/**
 * 格式化时间。
 * 
 * @param [in] tm 时间，NULL(0) 获取UTC时间。
 * @param [in] loc 本地化设置，NULL(0) 使用全局设置。
*/
const char *abcdk_time_format(const char *fmt, const struct tm *tm, locale_t loc);

/**
 * 格式化时间(GMT)。
 * 
 * @param [in] tm 时间，NULL(0) 获取UTC时间。
 * @param [in] loc 本地化设置，NULL(0) 使用全局设置。
*/
const char *abcdk_time_format_gmt(const struct tm *tm, locale_t loc);

__END_DECLS

#endif //ABCDK_UTIL_TIME_H