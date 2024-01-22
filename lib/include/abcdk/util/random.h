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

__END_DECLS

#endif //ABCDK_UTIL_RANDOM_H