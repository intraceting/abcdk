/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_REGISTER_H
#define ABCDK_UTIL_REGISTER_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"

__BEGIN_DECLS

/** 
 * 简单的寄存器。
 * 
 * @note 进程内有效。
 * 
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |63      32|31      24|23      16|15      8|7      0|
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |64bits.............................................|
 * |          |32bits..................................|
 * |                                |16bits............|
 * |                                          |8bits...|
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * 
 * @param [in] type 类型，仅支持8位、16位、32位、64位四种类型。
 * @param [in] addr 地址，0~255有效。
*/
volatile void *abcdk_register(int type, uint8_t addr);


__END_DECLS

#endif //ABCDK_UTIL_REGISTER_H