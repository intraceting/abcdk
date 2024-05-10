/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/rpc/rpc.h"

typedef struct _abcdk_rpc
{
    /*配置。*/
    abcdk_rpc_config_t cfg;

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*请求服务员。*/
    abcdk_waiter_t *req_waiter;

    /*消息编号*/
    uint64_t mid_next;

} ;//abcdk_rpc_node_t;

void abcdk_rpc_destroy(abcdk_rpc_t **ctx)
{
    abcdk_rpc_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_receiver_unref(&ctx_p->req_data);
    abcdk_waiter_free(&ctx_p->req_waiter);
    abcdk_heap_free(ctx_p);
}

static void _abcdk_rpc_waiter_destroy_cb(void *msg)
{
    abcdk_object_unref((abcdk_object_t**)&msg);
}

abcdk_rpc_t *abcdk_rpc_create(abcdk_rpc_config_t *cfg)
{
    abcdk_rpc_t *ctx;

    assert(cfg != NULL);
    assert(cfg->request_cb != NULL && cfg->output_cb != NULL);

    ctx = abcdk_heap_alloc(sizeof(abcdk_rpc_t));
    if(!ctx)
        return NULL;

    ctx->cfg = *cfg;
    ctx->req_data = NULL;
    ctx->req_waiter = abcdk_waiter_alloc(_abcdk_rpc_waiter_destroy_cb);
    ctx->mid_next = 1;

    return ctx;
ERR:

    abcdk_rpc_destroy(&ctx);

    return NULL;
}

static void _abcdk_rpc_input_process(abcdk_rpc_t *ctx)
{
    const void *req_data;
    size_t req_size;
    uint32_t len;
    uint8_t cmd;
    uint64_t mid;
    abcdk_object_t *cargo;
    int chk;

    req_data = abcdk_receiver_data(ctx->req_data, 0);
    req_size = abcdk_receiver_length(ctx->req_data);

    len = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 0, 32);
    cmd = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 32, 8);
    mid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);

    if (cmd == 1) //RSP
    {
        cargo = abcdk_object_copyfrom(ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
        if(!cargo)
            return;

        chk = abcdk_waiter_response(ctx->req_waiter, mid, cargo);
        if(chk != 0)
            abcdk_object_unref(&cargo);
    }
    else if (cmd == 2) //REQ
    {
        ctx->cfg.request_cb(ctx->cfg.opaque,mid,ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
    }

}

static void _abcdk_rpc_input_interrupt(abcdk_rpc_t *ctx)
{
    abcdk_waiter_cancel(ctx->req_waiter);
}

int abcdk_rpc_input(abcdk_rpc_t *ctx, const void *data, size_t size)
{
    size_t remain = 0;
    int chk;

    assert(ctx != NULL);

    if (data != NULL && size > 0)
    {
        size_t remain = 0;
        int chk;

        if (!ctx->req_data)
        {
            ctx->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, 16 * 1024 * 1024, NULL);
        }

        if (!ctx->req_data)
            return 0;

        chk = abcdk_receiver_append(ctx->req_data, data, size, &remain);
        if (chk < 0)
            return -1;
        else if (chk == 0) /*数据包不完整，继续接收。*/
            return 0;

        _abcdk_rpc_input_process(ctx);

        /*一定要回收。*/
        abcdk_receiver_unref(&ctx->req_data);

        /*如果有剩余数据则递归处理。*/
        if (remain > 0)
        {
            chk = abcdk_rpc_input(ctx, ABCDK_PTR2VPTR(data, size - remain), remain);
            if (chk != 0)
                return -1;
        }
    }
    else
    {
        _abcdk_rpc_input_interrupt(ctx);
    }

    return 0;
}

static int _abcdk_rpc_post(abcdk_rpc_t *ctx, uint8_t cmd, uint64_t mid, const void *data, size_t size)
{
    abcdk_object_t *msg;
    int chk;

    /*
     * |Length  |CMD    |MID     |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length： 不包含自身。
     * CMD：1 应答，2 请求。
     * MID：消息ID。
     */

    msg = abcdk_object_alloc2(4 + 1 + 8 + size);
    if (!msg)
        return -1;

    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 0, 32, msg->sizes[0] - 4);
    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 32, 8, cmd);
    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 40, 64, mid);
    memcpy(msg->pptrs[0] + 13, data, size);

    chk = ctx->cfg.output_cb(ctx->cfg.opaque, msg->pptrs[0], msg->sizes[0]);
    abcdk_object_unref(&msg);

    if(chk != 0)
        return -2;

    return 0;
}

int abcdk_rpc_request(abcdk_rpc_t *ctx, const void *req, size_t req_size, abcdk_object_t **rsp)
{
    uint64_t mid;
    abcdk_object_t *rsp_p = NULL;
    int chk;

    assert(ctx != NULL && req != NULL && req_size > 0);

    mid = abcdk_atomic_fetch_and_add(&ctx->mid_next, 1);

    if (rsp)
    {
        chk = abcdk_waiter_register(ctx->req_waiter, mid);
        if (chk != 0)
            return -1;
    }

    chk = _abcdk_rpc_post(ctx,2,mid,req,req_size);
    if (chk != 0)
        return -2;

    /*如果需要等待应签，这里就可以返回了。*/
    if(!rsp)
        return 0;

    *rsp = (abcdk_object_t *)abcdk_waiter_wait(ctx->req_waiter, mid, 365 * 24 * 60 * 60);
    if (!*rsp)
        return -3;

    return 0;
}

int abcdk_rpc_response(abcdk_rpc_t *ctx, uint64_t mid,const void *data,size_t size)
{
    int chk;

    assert(ctx != NULL && data != NULL && size >0);

    chk = _abcdk_rpc_post(ctx,1,mid,data,size);
    if(chk != 0)
        return -1;

    return 0;
}