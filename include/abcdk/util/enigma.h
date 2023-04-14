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

__BEGIN_DECLS

/** Enigma加密机。*/
typedef struct _abcdk_enigma abcdk_enigma_t;

/** 配置。*/
typedef struct _abcdk_enigma_config
{
    /** 字典数量。*/
    uint8_t count;
    
    /** 字典表格。*/
    const uint8_t dict[256][256];

    /** 反射器。*/
    uint8_t (*reflector_cb)(uint8_t s);
    
}abcdk_enigma_config_t;

/** 销毁。*/
void abcdk_enigma_free(abcdk_enigma_t **ctx);

/** 创建。*/
abcdk_enigma_t *abcdk_enigma_create(abcdk_enigma_config_t *cfg);

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
