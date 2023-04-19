/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_ENIGMA_H
#define ABCDK_UTIL_ENIGMA_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/random.h"

__BEGIN_DECLS

/** Enigma加密机。*/
typedef struct _abcdk_enigma abcdk_enigma_t;

/**
 * 制作字典。
 * 
 * @param [in out] seed 随机种子。
 * @param [in out] dict 字典表格。
 * @param [in] rows 字典行数。
 * @param [in] cols 字典列数。
*/
void abcdk_enigma_mkdict(uint64_t *seed,uint8_t *dict,size_t rows,size_t cols);


/** 销毁。*/
void abcdk_enigma_free(abcdk_enigma_t **ctx);

/** 
 * 创建。
 * 
 * @param [in out] dict 字典表格。
 * @param [in] rows 字典行数(转子的个数)。
 * @param [in] cols 字典列数(转子的通道)。
 * 
*/
abcdk_enigma_t *abcdk_enigma_create(const uint8_t *dict,size_t rows,size_t cols);

/** 
 * 获取转子指针。
 * 
 * @param [in] index  转子编号。0~255。
*/
uint8_t abcdk_enigma_getpos(abcdk_enigma_t *ctx,uint8_t rotor);

/** 
 * 设置转子指针。
 * 
 * @param [in] rotor 转子编号。0~255。
 * @param [in] pos  转子指针。
*/
uint8_t abcdk_enigma_setpos(abcdk_enigma_t *ctx,uint8_t rotor, uint8_t pos);

/**
 * 执行。
*/
void abcdk_enigma_execute(abcdk_enigma_t *ctx,void *dst,const void *src,size_t size);


__END_DECLS

#endif //ABCDK_UTIL_ENIGMA_H
