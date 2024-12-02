/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_CPU_H
#define ABCDK_UTIL_CPU_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**
 * 获取CPUID。
 * 
 * @return > 0 有效长度。<= 0 不支持的CPU。
*/
int abcdk_get_cpuid(unsigned int buf[48]);

__END_DECLS

#endif //ABCDK_UTIL_CPU_H