/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_H2645_H
#define ABCDK_UTIL_H2645_H

#include "abcdk/util/defs.h"

__BEGIN_DECLS

/**
 * 查找起始码。
 * 
 * @param [in out] msize 起始码长度。
 * 
 * @return >=0 成功(当前位置到起始码开始的偏移量)，< 0 不存在。
*/
ssize_t abcdk_h2645_find_start_code(const void *b, const void *e,int *msize);

/**
 * 分包，拆包。 
 *
 */
const void *abcdk_h2645_packet_split(void **next,const void *e);

__END_DECLS

#endif //ABCDK_UTIL_H2645_H
