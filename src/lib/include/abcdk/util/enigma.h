/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
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
#include "abcdk/util/sha256.h"

__BEGIN_DECLS

/** Enigma加密机.*/
typedef struct _abcdk_enigma abcdk_enigma_t;

/** 销毁.*/
void abcdk_enigma_destroy(abcdk_enigma_t **ctx);

/** 
 * 创建.
 * 
 * @param [in] rows 转子数量.在2 - 128之间有效.
 * @param [in] cols 通道数量.在2 - 256之间的偶数有效.
*/
abcdk_enigma_t *abcdk_enigma_create(int rows, int cols);

/** 
 * 初始化.
 * 
 * @note 每个转子内字符的值不能出现重复.
 * @note 反射板内字符的值不能出现重复.
 * 
 * @param [in] rotors 转子.[ROWS * COLS] 数组.
 * @param [in] rboard 反射板.[COLS] 数组.
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_enigma_init(abcdk_enigma_t *ctx,uint8_t rotors[], uint8_t rboard[]);

/**
 * 亮灯.
 * 
 * @note 加密与解密的过程是一样的.
*/
uint8_t abcdk_enigma_light(abcdk_enigma_t *ctx, uint8_t c);
#define abcdk_enigma_update abcdk_enigma_light

/**
 * 亮灯(批量).
 * 
 * @note 加密与解密的过程是一样的.
*/
void abcdk_enigma_light_batch(abcdk_enigma_t *ctx,uint8_t *dst, const uint8_t *src,size_t size);
#define abcdk_enigma_update_batch abcdk_enigma_light_batch 


__END_DECLS

#endif //ABCDK_ENIGMA_ENIGMA_H
