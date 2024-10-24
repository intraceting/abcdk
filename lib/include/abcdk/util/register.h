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

/*设置寄存器的值。*/
#define abcdk_register64_set(A, V) abcdk_atomic_store((uint64_t *)abcdk_register(64, A), V)
#define abcdk_register32_set(A, V) abcdk_atomic_store((uint32_t *)abcdk_register(32, A), V)
#define abcdk_register16_set(A, V) abcdk_atomic_store((uint16_t *)abcdk_register(16, A), V)
#define abcdk_register8_set(A, V) abcdk_atomic_store((uint8_t *)abcdk_register(8, A), V)

/*获取寄存器的值。*/
#define abcdk_register64_get(A) abcdk_atomic_load((uint64_t *)abcdk_register(64, A))
#define abcdk_register32_get(A) abcdk_atomic_load((uint32_t *)abcdk_register(32, A))
#define abcdk_register16_get(A) abcdk_atomic_load((uint16_t *)abcdk_register(16, A))
#define abcdk_register8_get(A) abcdk_atomic_load((uint8_t *)abcdk_register(8, A))

__END_DECLS

#endif //ABCDK_UTIL_REGISTER_H