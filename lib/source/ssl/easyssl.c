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

    /**发送队列。*/
    abcdk_tree_t *send_queue;

    /**发送游标。*/
    size_t send_pos;

    /**重发指针和长度。*/
    const void *send_repeated_p;
    size_t send_repeated_l;

    /**接收缓存。*/
    abcdk_stream_t *recv_buf;

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
    abcdk_tree_free(&ctx_p->send_queue);
    abcdk_stream_destroy(&ctx_p->recv_buf);

    abcdk_heap_free(ctx_p);
}

int _abcdk_easyssl_init_enigma(abcdk_easyssl_t *ctx,const uint8_t *key,size_t size,uint32_t scheme)
{
    uint8_t hashcode[32];
    uint64_t seed[4] = {0};
    int chk;

    /*密钥转换为定长HASHCODE。*/
    chk = abcdk_sha256_once(key,size,hashcode);
    if(chk != 0)
        return -1;
    
    /*分解成4个64位整数。不能直接复制内存，因为存在大小端存储顺序不同的问题。*/
    for (int i = 0; i < 32; i++)
    {
        seed[i % 4] <<= 8;
        seed[i % 4] |= (uint64_t)hashcode[i];
    }

    ctx->en_send_ctx = abcdk_enigma_create3(seed,4,256);
    ctx->en_recv_ctx = abcdk_enigma_create3(seed,4,256);

    if(!ctx->en_send_ctx || !ctx->en_recv_ctx)
        return -2;

    ctx->send_queue = abcdk_tree_alloc3(1);
    if(!ctx->send_queue)
        return -3;

    ctx->recv_buf = abcdk_stream_create();
    if(!ctx->recv_buf)
        return -4;

    ctx->send_pos = 0;
    ctx->send_repeated_p = NULL;
    ctx->send_repeated_l = 0;
    ctx->fd = -1;

    return 0;

}


abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *key,size_t size,uint32_t scheme)
{
    abcdk_easyssl_t *ctx;
    int chk;

    assert(key != NULL && size > 0);

    ctx = (abcdk_easyssl_t*)abcdk_heap_alloc(sizeof(abcdk_easyssl_t));
    if(!ctx)
        return NULL;

    if(scheme == ABCDK_EASYSSL_SCHEME_ENIGMA)
        chk = _abcdk_easyssl_init_enigma(ctx,key,size,scheme);
    else 
        chk = -22;


    if(chk == 0)
        return ctx;

ERR:

    abcdk_easyssl_destroy(&ctx);
    return NULL;
}

int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd)
{
    int old;

    assert(ctx != NULL && fd >= 0);

    old = ctx->fd;
    ctx->fd = fd;

    return old;
}

int abcdk_easyssl_get_fd(abcdk_easyssl_t *ctx)
{
    int old;

    assert(ctx != NULL);

    old = ctx->fd;

    return old;
}

ssize_t abcdk_easyssl_send(abcdk_easyssl_t *ctx,const void *data,size_t size)
{
    abcdk_tree_t *en_data;
    abcdk_tree_t *p;
    ssize_t slen;

    assert(ctx != NULL && data != NULL && size >0);

    /*警告：如果参数的指针和长度未改变，则认为是管道空闲重发。由于前一次调用已经对数据进行加密并加入待发送对列，因此忽略即可。*/
    if(ctx->send_repeated_p != data || ctx->send_repeated_l != size)
    {
        en_data = abcdk_tree_alloc3(size);
        if(!en_data)
            return -1;

        /*记录指针和长度，重发时会检测这两个值。*/
        ctx->send_repeated_p = data;
        ctx->send_repeated_l = size;

        /*加密。*/
        abcdk_enigma_light_batch_u8(ctx->en_send_ctx,en_data->obj->pptrs[0],data,size);

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue,en_data,0);
    }

NEXT_MSG:

    p = abcdk_tree_child(ctx->send_queue,1);

    /*通知应用层，发送队列空闲。*/
    if(!p)
    {
        ctx->send_repeated_p = NULL;
        ctx->send_repeated_l = 0;
        return size;
    }

    /*
     * 发。
     * 
     * 警告：补发数据时参数不能改变(指针和长度)。
    */
    //slen = send(ctx->fd, ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->send_pos), p->obj->sizes[0] - ctx->send_pos,0);
    slen = write(ctx->fd, ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->send_pos), p->obj->sizes[0] - ctx->send_pos);
    if (slen < 0)
        return -1;
    else if (slen == 0)
        return 0;

    /*滚动发送游标。*/
    ctx->send_pos += slen;

    /*当前节点未发送完整，则继续发送。*/
    if (ctx->send_pos < p->obj->sizes[0])
        goto NEXT_MSG;

    /*发送游标归零。*/
    ctx->send_pos = 0;

    /*从队列中删除已经发送完整的节点。*/
    abcdk_tree_unlink(p);
    abcdk_tree_free(&p);

    /*并继续发送剩余节点。*/
    goto NEXT_MSG;  
}