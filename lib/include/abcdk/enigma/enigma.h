/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_ENIGMA_ENIGMA_H
#define ABCDK_ENIGMA_ENIGMA_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/random.h"
#include "abcdk/util/object.h"
#include "abcdk/util/clock.h"

__BEGIN_DECLS

/** Enigma加密机。*/
typedef struct _abcdk_enigma abcdk_enigma_t;

/**
 * 制作字典。
 * 
 * @param [in out] seed 随机种子。
 * @param [in out] dict 字典表格。
 * @param [in] rows 转子数量。大于等于3有效。
 * @param [in] cols 通道数量。小于等256的偶数有效。
*/
void abcdk_enigma_mkdict(uint64_t *seed,uint8_t *dict,size_t rows, size_t cols);


/** 销毁。*/
void abcdk_enigma_free(abcdk_enigma_t **ctx);

/** 
 * 创建。
 * 
 * @param [in] dict 字典表格。
 * @param [in] rows 转子数量。
 * @param [in] cols 通道数量。
*/
abcdk_enigma_t *abcdk_enigma_create(const uint8_t *dict,size_t rows, size_t cols);

/** 
 * 创建。
 * 
 * @param [in] seed 随机种子。
 * @param [in] rows 转子数量。
 * @param [in] cols 通道数量。
*/
abcdk_enigma_t *abcdk_enigma_create2(uint64_t seed,size_t rows, size_t cols);

/** 
 * 创建。
 * 
 * @param [in] seed 随机种子(每个转子使用不同的种子)。
 * @param [in] rows 转子数量。
 * @param [in] cols 通道数量。
*/
abcdk_enigma_t *abcdk_enigma_create3(uint64_t seed[],size_t rows, size_t cols);

/** 
 * 获取转子指针。
 * 
 * @param [in] row  转子编号。
*/
uint8_t abcdk_enigma_getpos(abcdk_enigma_t *ctx,size_t row);

/** 
 * 设置转子指针。
 * 
 * @param [in] row 转子编号。
 * @param [in] pos 转子指针。
*/
uint8_t abcdk_enigma_setpos(abcdk_enigma_t *ctx,size_t row, uint8_t pos);

/**
 * 亮灯。
 * 
 * @note 加密和解密过程是相同的，输一个得到另一个。
 * 
*/
uint8_t abcdk_enigma_light(abcdk_enigma_t *ctx, uint8_t c);

/**
 * 批量亮灯。
*/
void abcdk_enigma_light_batch(abcdk_enigma_t *ctx,uint8_t *dst,const uint8_t *src,size_t size);


__END_DECLS

#endif //ABCDK_ENIGMA_ENIGMA_H
