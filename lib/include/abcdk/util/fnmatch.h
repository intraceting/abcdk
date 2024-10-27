/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_FNMATCH_H
#define ABCDK_UTIL_FNMATCH_H

#include "abcdk/util/defs.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * 智能匹配。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_fnmatch(const char *str,const char *pattern,int caseAb,int ispath);


__END_DECLS

#endif //ABCDK_UTIL_FNMATCH_H