/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ssl/easyssl.h"

/** 简单的SSL通讯。 */
struct _abcdk_easyssl
{
    /**发送加密环境。*/
    abcdk_enigma_t *en_send_ctx;

    /**接收加密环境。*/
    abcdk_enigma_t *en_recv_ctx;


    /**句柄。*/
    int fd;
    
};//abcdk_easyssl_t;


void abcdk_easyssl_destroy(abcdk_easyssl_t **ctx)
{
    abcdk_easyssl_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_enigma_free(&ctx_p->en_recv_ctx);
    abcdk_enigma_free(&ctx_p->en_send_ctx);

    abcdk_heap_free(ctx_p);
}

int _abcdk_easyssl_init_enigma(abcdk_easyssl_t *ctx,const uint8_t *key,size_t size,uint32_t scheme)
{
    int gp = sizeof(uint64_t);
    size_t rows;
    abcdk_object_t *tmp;

    /*计算转子数量。*/
    rows = abcdk_align(size,gp)/gp;

    size_t sizes[2] = {rows * gp, rows * 256};
    tmp = abcdk_object_alloc(sizes, 2, 0);
    if(!tmp)
        return -1;

    for (size_t i = 0; i < size; i++)
    {
       ABCDK_PTR2U64PTR(tmp->pptrs[0],0)[i%gp] <<= 8;
       ABCDK_PTR2U64PTR(tmp->pptrs[0],0)[i%gp] |= key[i];
    }


}


abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *key,size_t size,uint32_t scheme)
{
    abcdk_easyssl_t *ctx;

    assert(key != NULL && size > 0);

    ctx = (abcdk_easyssl_t*)abcdk_heap_alloc(sizeof(abcdk_easyssl_t));
    if(!ctx)
        return NULL;

    
}