/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ssl/easyssl.h"

/*分块长度。*/
#define ABCDK_EASYSSL_CHUNK_SIZE (256*1024LLU)

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

    /*发送前撒盐。0 未，1 已。*/
    int send_sprinkle_salt;

    /**接收队列。*/
    abcdk_stream_t *recv_queue;

    /**接收缓存。*/
    abcdk_object_t *recv_buf;

    /**已接收的盐长度。*/
    size_t recv_salt_len;

    /*盐长度。*/
    size_t salt_len;

    /**发送句柄。*/
    int send_fd;

    /**接收句柄。*/
    int recv_fd;
    
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
    abcdk_stream_destroy(&ctx_p->recv_queue);
    abcdk_object_unref(&ctx_p->recv_buf);

    abcdk_heap_free(ctx_p);
}

int _abcdk_easyssl_init_enigma(abcdk_easyssl_t *ctx,const uint8_t *key,size_t size,uint32_t scheme)
{
    uint8_t hashcode[32];
    uint64_t send_seed[4] = {0},recv_seed[4] = {0};
    int chk;

    /*密钥转换为定长HASHCODE。*/
    chk = abcdk_sha256_once(key,size,hashcode);
    if(chk != 0)
        return -1;
    
    /*分解成4个64位整数。不能直接复制内存，因为存在大小端存储顺序不同的问题。*/
    for (int i = 0; i < 32; i++)
    {
        send_seed[i % 4] <<= 8;
        send_seed[i % 4] |= (uint64_t)hashcode[i];
        recv_seed[i % 4] <<= 8;
        recv_seed[i % 4] |= (uint64_t)hashcode[i];
    }

    ctx->en_send_ctx = abcdk_enigma_create3(send_seed,4,256);
    ctx->en_recv_ctx = abcdk_enigma_create3(recv_seed,4,256);

    if(!ctx->en_send_ctx || !ctx->en_recv_ctx)
        return -2;

    ctx->send_queue = abcdk_tree_alloc3(1);
    if(!ctx->send_queue)
        return -3;

    ctx->recv_queue = abcdk_stream_create();
    if(!ctx->recv_queue)
        return -4;

    ctx->recv_buf = abcdk_object_alloc2(ABCDK_EASYSSL_CHUNK_SIZE);
    if(!ctx->recv_buf)
        return -5;

    ctx->send_pos = 0;
    ctx->send_repeated_p = NULL;
    ctx->send_repeated_l = 0;
    ctx->send_sprinkle_salt = 0;
    ctx->recv_salt_len = 0;

    return 0;

}


abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *key,size_t size,uint32_t scheme, size_t salt)
{
    abcdk_easyssl_t *ctx;
    int chk;

    assert(key != NULL && size > 0 && salt <= 256);

    ctx = (abcdk_easyssl_t*)abcdk_heap_alloc(sizeof(abcdk_easyssl_t));
    if(!ctx)
        return NULL;

    ctx->salt_len = salt;
    ctx->send_fd = -1;
    ctx->recv_fd = -1;

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

abcdk_easyssl_t *abcdk_easyssl_create_from_file(const char *file,uint32_t scheme,size_t salt)
{
    abcdk_easyssl_t *ctx;
    abcdk_object_t *key;

    assert(file != NULL && salt <= 256);

    key = abcdk_mmap_filename(file,0,0,0,0);
    if(!key)
        return NULL;

    ctx = abcdk_easyssl_create(key->pptrs[0],key->sizes[0],scheme,salt);
    abcdk_object_unref(&key);
    if(!ctx)
        return NULL;

    return ctx;
}

int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd,int writer)
{
    int old;

    assert(ctx != NULL && fd >= 0);

    if(writer)
    {
        old = ctx->send_fd;
        ctx->send_fd = fd;
    }
    else
    {
        old = ctx->recv_fd;
        ctx->recv_fd = fd;
    }

    return old;
}

int abcdk_easyssl_get_fd(abcdk_easyssl_t *ctx,int writer)
{
    int old;

    assert(ctx != NULL);

    if(writer)
        old = ctx->send_fd;
    else 
        old = ctx->recv_fd;

    return old;
}

int abcdk_easyssl_do_handshake(abcdk_easyssl_t *ctx,int server)
{
    assert(ctx != NULL);

    ABCDK_ASSERT(ctx->send_fd >= 0 && ctx->recv_fd >= 0,"未关联句柄，无法完成握手。");

    return 1;
}

ssize_t _abcdk_easyssl_write(abcdk_easyssl_t *ctx,const void *data,size_t size)
{
    char salt[256+1] = {0};
    abcdk_tree_t *en_data = NULL;
    abcdk_tree_t *p = NULL;
    ssize_t slen = 0;

    assert(ctx != NULL && data != NULL && size >0);

    /*发送前先撒盐。*/
    if(ctx->salt_len > 0 && !ctx->send_sprinkle_salt)
    {
        en_data = abcdk_tree_alloc3(ctx->salt_len);
        if(!en_data)
            return 0;//内存不足时，关闭当前句柄。

        /*从所有可见字符中选取。*/
        abcdk_rand_string(salt,ctx->salt_len,0);
        
        /*加密。*/
        abcdk_enigma_light_batch_u8(ctx->en_send_ctx,en_data->obj->pptrs[0],salt,ctx->salt_len);

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue,en_data,0);

        /*撒盐一次即可。*/
        ctx->send_sprinkle_salt = 1;
    }

    /*警告：如果参数的指针和长度未改变，则认为是管道空闲重发。由于前一次调用已经对数据进行加密并加入待发送对列，因此忽略即可。*/
    if(ctx->send_repeated_p != data || ctx->send_repeated_l != size)
    {
        en_data = abcdk_tree_alloc3(size);
        if(!en_data)
            return 0;//内存不足时，关闭当前句柄。

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

    assert(ctx->send_fd >= 0);

    /*
     * 发。
     * 
     * 警告：补发数据时参数不能改变(指针和长度)。
    */
    slen = write(ctx->send_fd, ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->send_pos), p->obj->sizes[0] - ctx->send_pos);
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

ssize_t abcdk_easyssl_write(abcdk_easyssl_t *ctx,const void *data,size_t size)
{
    ssize_t slen = 0,alen = 0;

    assert(ctx != NULL && data != NULL && size >0);

    while(alen < size)
    {
        slen = _abcdk_easyssl_write(ctx,ABCDK_PTR2VPTR(data,alen),ABCDK_MIN(size-alen,(size_t)(ABCDK_EASYSSL_CHUNK_SIZE)));
        if (slen < 0)
            return (alen > 0 ? alen : -1); //优先返回已发送的数据长度。
        else if (slen == 0)
            return (alen > 0 ? alen : 0); //优先返回已发送的数据长度。

        alen += slen;
    }

    return alen;
}

ssize_t abcdk_easyssl_read(abcdk_easyssl_t *ctx,void *data,size_t size)
{
    char salt[256+1] = {0};
    abcdk_object_t *de_data = NULL;
    ssize_t rlen = 0,alen = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size >0);

NEXT_LOOP:

    /*如果数据存在盐则先读取盐。*/
    if (ctx->salt_len > 0 && ctx->salt_len > ctx->recv_salt_len)
    {
        rlen = abcdk_stream_read(ctx->recv_queue, ABCDK_PTR2VPTR(salt,ctx->recv_salt_len), ctx->salt_len - ctx->recv_salt_len);
        if (rlen > 0)
            ctx->recv_salt_len += rlen;
    }

    /*盐读取完成后，才是真实数据。*/
    if (ctx->salt_len == ctx->recv_salt_len)
    {
        rlen = abcdk_stream_read(ctx->recv_queue, ABCDK_PTR2VPTR(data,alen),size - alen);
        if (rlen > 0)
            alen += rlen;

        if(alen >= size)
            return alen;
    }

    assert(ctx->recv_fd >= 0);

    /*收。*/
    rlen = read(ctx->recv_fd, ctx->recv_buf->pptrs[0], ctx->recv_buf->sizes[0]);
    if (rlen < 0)
        return (alen > 0 ? alen : -1); //优先返回已接收的数据长度。
    else if (rlen == 0)
        return (alen > 0 ? alen : 0); //优先返回已接收的数据长度。

    de_data = abcdk_object_alloc2(rlen);
    if(!de_data)
        return 0;//内存不足时，关闭当前句柄。
    
    /*解密。*/
    abcdk_enigma_light_batch_u8(ctx->en_recv_ctx,de_data->pptrs[0],ctx->recv_buf->pptrs[0],rlen);

    /*追加到接收队列。*/
    chk = abcdk_stream_write(ctx->recv_queue,de_data);
    if(chk != 0)
        return 0;//内存不足时，关闭当前句柄。

    goto NEXT_LOOP;
}