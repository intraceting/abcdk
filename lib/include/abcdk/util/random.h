/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_RANDOM_H
#define ABCDK_UTIL_RANDOM_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/time.h"

__BEGIN_DECLS

/**
 * 产生一个随机数。
 * 
 * @param [in out] seed 随机种子。
 */
int64_t abcdk_rand(uint64_t *seed);

/** 产生一个随机数。*/
int64_t abcdk_rand_q();
#define abcdk_rand_number abcdk_rand_q

/**
 * 产生随机字符串。
 * 
 * @param [in] type 类型。0 所有可见字符，1 所有字母和数字，2 所在大写字母，3 所有小字字母，4 所有数字。
*/
char *abcdk_rand_string(char *buf,size_t size,int type);

__END_DECLS

#endif //ABCDK_UTIL_RANDOM_H