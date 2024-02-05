/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/asio/tipc.h"

/**简单的TIPC服务。*/
struct _abcdk_tipc
{
    /*配置。*/
    abcdk_tipc_config_t cfg;

    /*通讯IO。*/
    abcdk_asynctcp_t *io_ctx;

    /*节点列表。*/
    abcdk_map_t *slave_list;

    /*节点同步锁。*/
    abcdk_mutex_t *slave_mutex;

    /*下一个MID。*/
    volatile uint64_t mid_next;

}; // abcdk_tipc_t;

typedef struct _abcdk_tipc_node
{
    /*父级。*/
    abcdk_tipc_t *father;

    /*SSL环境。*/
    SSL_CTX *ssl_ctx;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*服务ID。*/
    uint64_t id;

    /*服务标识。*/
    uint64_t mark;

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*请求服务员。*/
    abcdk_waiter_t *req_waiter;

} abcdk_tipc_node_t;

typedef struct _abcdk_tipc_slave
{
    /*服务ID。*/
    uint64_t id;

    /*地址。*/
    char location[NAME_MAX];

    /*连接方式。0：任意，1: 主动，2：被动。*/
    int flag;

    /*被动连接管道。*/
    abcdk_asynctcp_node_t *pipe1;

    /*被动连接服务标识(防止多个相同的ID出现在不同的地方)。*/
    uint64_t mark1;

    /*主动连接管道。*/
    abcdk_asynctcp_node_t *pipe2;

} abcdk_tipc_slave_t;

static int _abcdk_tipc_slave_register(abcdk_tipc_t *ctx, abcdk_asynctcp_node_t *pipe, const char *location)
{
    abcdk_object_t *slave_p;
    abcdk_tipc_slave_t *slave_ctx_p;
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(pipe);

    /*远程的ID不能与自己的ID相同。*/
    if (ctx->cfg.id == node_ctx_p->id)
        return -1;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_p = abcdk_map_find2(ctx->slave_list, &node_ctx_p->id, sizeof(abcdk_tipc_slave_t));
    if (!slave_p)
    {
        chk = -2;
        goto ERR;
    }

    slave_ctx_p = (abcdk_tipc_slave_t *)slave_p->pptrs[ABCDK_MAP_VALUE];

    if (location)
        strncpy(slave_ctx_p->location, location, NAME_MAX);

    if (node_ctx_p->flag == 1)
    {
        if (slave_ctx_p->pipe1 == NULL)
        {
            slave_ctx_p->pipe1 = abcdk_asynctcp_refer(pipe);
            slave_ctx_p->mark1 = node_ctx_p->mark;
        }
        else if (slave_ctx_p->mark1 != node_ctx_p->mark)
        {
            chk = -3;
            goto ERR;
        }
        else if (slave_ctx_p->pipe1 != pipe)
        {
            abcdk_asynctcp_unref(&slave_ctx_p->pipe1);
            slave_ctx_p->pipe1 = abcdk_asynctcp_refer(pipe);
        }
    }
    else
    {
        if (slave_ctx_p->pipe2 != pipe)
        {
            abcdk_asynctcp_unref(&slave_ctx_p->pipe2);
            slave_ctx_p->pipe2 = abcdk_asynctcp_refer(pipe);
        }
    }

    /* 当双向建立连成功时，保留由ID大到小的主动连接。*/
    if (slave_ctx_p->pipe1 && slave_ctx_p->pipe2)
    {
        if (ctx->cfg.id > node_ctx_p->id)
        {
            abcdk_asynctcp_set_timeout(slave_ctx_p->pipe1, 1);
            abcdk_asynctcp_unref(&slave_ctx_p->pipe1);

            slave_ctx_p->flag = 1;
        }
        else
        {
            abcdk_asynctcp_set_timeout(slave_ctx_p->pipe2, 1);
            abcdk_asynctcp_unref(&slave_ctx_p->pipe2);

            slave_ctx_p->flag = 2;
        }
    }

    abcdk_mutex_unlock(ctx->slave_mutex);
    return 0;

ERR:

    abcdk_mutex_unlock(ctx->slave_mutex);
    return chk;
}

static void _abcdk_tipc_slave_unregister(abcdk_tipc_t *ctx, abcdk_asynctcp_node_t *pipe)
{
    abcdk_object_t *slave_p;
    abcdk_tipc_slave_t *slave_ctx_p;
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(pipe);

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_p = abcdk_map_find2(ctx->slave_list, &node_ctx_p->id, 0);
    if (!slave_p)
        goto END;

    slave_ctx_p = (abcdk_tipc_slave_t *)slave_p->pptrs[ABCDK_MAP_VALUE];
    if (slave_ctx_p->pipe1 == pipe)
    {
        abcdk_asynctcp_unref(&slave_ctx_p->pipe1);
        slave_ctx_p->mark1 = 0;
    }
    else if (slave_ctx_p->pipe2 == pipe)
    {
        abcdk_asynctcp_unref(&slave_ctx_p->pipe2);
    }
    

END:

    abcdk_mutex_unlock(ctx->slave_mutex);
    return;
}

static abcdk_asynctcp_node_t *_abcdk_tipc_slave_find_pipe(abcdk_tipc_t *ctx,uint64_t id)
{
    abcdk_object_t *slave_p;
    abcdk_asynctcp_node_t *node_p = NULL;
    abcdk_tipc_slave_t *slave_ctx_p;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_p = abcdk_map_find2(ctx->slave_list, &id, 0);
    if (!slave_p)
        goto END;

    slave_ctx_p = (abcdk_tipc_slave_t *)slave_p->pptrs[ABCDK_MAP_VALUE];

    /*优选择可能被保留的链路。*/
    if(ctx->cfg.id > id)
        node_p = (slave_ctx_p->pipe2?slave_ctx_p->pipe2:slave_ctx_p->pipe1);
    else 
        node_p = (slave_ctx_p->pipe1?slave_ctx_p->pipe1:slave_ctx_p->pipe2);

    /*增加引用计数。*/
    if(node_p)
        node_p = abcdk_asynctcp_refer(node_p);
END:

    abcdk_mutex_unlock(ctx->slave_mutex);

    return node_p;
}

static void _abcdk_tipc_slave_destructor_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_tipc_t *ctx_p = (abcdk_tipc_t *)opaque;
    const char *id_p = (const char *)obj->pptrs[ABCDK_MAP_KEY];
    abcdk_tipc_slave_t *slave_ctx_p = (abcdk_tipc_slave_t *)obj->pptrs[ABCDK_MAP_VALUE];

    abcdk_asynctcp_unref(&slave_ctx_p->pipe1);
    abcdk_asynctcp_unref(&slave_ctx_p->pipe2);
}

static void _abcdk_tipc_slave_construct_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_tipc_t *ctx_p = (abcdk_tipc_t *)opaque;
    uint64_t id = *((uint64_t *)obj->pptrs[ABCDK_MAP_KEY]);
    abcdk_tipc_slave_t *slave_ctx_p = (abcdk_tipc_slave_t *)obj->pptrs[ABCDK_MAP_VALUE];

    slave_ctx_p->id = id;
    slave_ctx_p->flag = 0;
}

void abcdk_tipc_destroy(abcdk_tipc_t **ctx)
{
    abcdk_tipc_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_asynctcp_stop(&ctx_p->io_ctx);
    abcdk_map_destroy(&ctx_p->slave_list);
    abcdk_mutex_destroy(&ctx_p->slave_mutex);

    abcdk_heap_free(ctx_p);
}

abcdk_tipc_t *abcdk_tipc_create(abcdk_tipc_config_t *cfg)
{
    abcdk_tipc_t *ctx;

    assert(cfg != NULL);
    assert(cfg->id != 0 && cfg->request_cb != NULL);

    ctx = (abcdk_tipc_t *)abcdk_heap_alloc(sizeof(abcdk_tipc_t));
    if (!ctx)
        return NULL;

    ctx->cfg = *cfg;
    ctx->io_ctx = abcdk_asynctcp_start(1000, -1);
    ctx->slave_list = abcdk_map_create(16);
    ctx->slave_mutex = abcdk_mutex_create();
    ctx->mid_next = 1;

    ctx->slave_list->construct_cb = _abcdk_tipc_slave_construct_cb;
    ctx->slave_list->destructor_cb = _abcdk_tipc_slave_destructor_cb;
    ctx->slave_list->opaque = ctx;

    return ctx;
}

static void _abcdk_tipc_node_destroy_cb(void *userdata)
{
    abcdk_tipc_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_tipc_node_t *)userdata;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_ctx_free(&ctx->ssl_ctx);
#endif // HEADER_SSL_H

    abcdk_receiver_unref(&ctx->req_data);
    abcdk_waiter_free(&ctx->req_waiter);
}

static void _abcdk_tipc_node_waiter_destroy_cb(void *msg)
{
    abcdk_object_t *obj_p;

    obj_p = (abcdk_object_t *)msg;

    abcdk_object_unref(&obj_p);
}

static abcdk_asynctcp_node_t *_abcdk_tipc_node_create(abcdk_tipc_t *ctx,int flag)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;

    node_p = abcdk_asynctcp_alloc(ctx->io_ctx, sizeof(abcdk_tipc_node_t), _abcdk_tipc_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->flag = flag;
    if(flag != 0)
        node_ctx_p->req_waiter = abcdk_waiter_alloc(_abcdk_tipc_node_waiter_destroy_cb);

    return node_p;
}

static void _abcdk_tipc_prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_tipc_node_t *listen_ctx_p, *node_ctx_p;

    listen_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(listen);

    node_p = _abcdk_tipc_node_create(listen_ctx_p->father,1);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_tipc_event_accept(abcdk_asynctcp_node_t *node, int *result)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    /*默认：允许。*/
    *result = 0;

    if (node_ctx_p->father->cfg.accept_cb)
        node_ctx_p->father->cfg.accept_cb(node_ctx_p->father->cfg.opaque, node_ctx_p->remote_addr, result);

    if (*result != 0)
        abcdk_trace_output(LOG_INFO, "禁止客户端('%s')连接到本机。", node_ctx_p->remote_addr);
}

static int _abcdk_tipc_post_register(abcdk_asynctcp_node_t *node);

static void _abcdk_tipc_event_connect(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    /*设置超时。*/
    abcdk_asynctcp_set_timeout(node, 180 * 1000);

    if (node_ctx_p->flag == 2)
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    ssl_p = abcdk_asynctcp_ssl(node);
    if (!ssl_p)
        goto END;

#ifdef HEADER_SSL_H

    /*检查SSL验证结果。*/
    chk = SSL_get_verify_result(ssl_p);
    if (chk != X509_V_OK)
    {
        abcdk_trace_output(LOG_INFO, "验证('%s')的证书失败，证书已过期或未生效。", node_ctx_p->remote_addr);

        /*修改超时，使用超时检测器关闭。*/
        abcdk_asynctcp_set_timeout(node, 1);
        return;
    }

#endif // HEADER_SSL_H

END:

    abcdk_trace_output(LOG_INFO, "本机与%s('%s')的连接已经建立。", (node_ctx_p->flag == 2 ? "客户端" : "服务端"), node_ctx_p->remote_addr);

    /*发送注册消息。*/
    if (node_ctx_p->flag == 2)
        _abcdk_tipc_post_register(node);


    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _abcdk_tipc_event_output(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);
}

static void _abcdk_tipc_event_close(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    if (!node_ctx_p->remote_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    if (node_ctx_p->flag == 0)
    {
        abcdk_trace_output(LOG_INFO, "监听关闭");
        return;
    }

    abcdk_trace_output(LOG_INFO, "本机与%s('%s')的连接已经断开。", (node_ctx_p->flag == 2 ? "客户端" : "服务端"), node_ctx_p->remote_addr);

    /*取消所有等待的。*/
    abcdk_waiter_cancel(node_ctx_p->req_waiter);

    _abcdk_tipc_slave_unregister(node_ctx_p->father, node);
}

static void _abcdk_tipc_event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    if (event == ABCDK_ASYNCTCP_EVENT_ACCEPT)
    {
        _abcdk_tipc_event_accept(node, result);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CONNECT)
    {
        _abcdk_tipc_event_connect(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
    {
        _abcdk_tipc_event_output(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CLOSE || event == ABCDK_ASYNCTCP_EVENT_INTERRUPT)
    {
        _abcdk_tipc_event_close(node);
    }
}

static void _abcdk_tpic_request_process(abcdk_asynctcp_node_t *node);

static void _abcdk_tipc_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    if (!node_ctx_p->req_data)
        node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_SMB, 16 * 1024 * 1024, NULL);

    if (!node_ctx_p->req_data)
        goto ERR;

    chk = abcdk_receiver_append(node_ctx_p->req_data, data, size, remain);
    if (chk < 0)
        goto ERR;
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_tpic_request_process(node);

    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);

    /*No Error.*/
    return;

ERR:

    abcdk_asynctcp_set_timeout(node, 1);
}

int abcdk_tipc_listen(abcdk_tipc_t *ctx, abcdk_sockaddr_t *addr)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    assert(ctx != NULL && addr != NULL);

    node_p = _abcdk_tipc_node_create(ctx,0);
    if (!node_p)
        return -1;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node_p);

#ifdef HEADER_SSL_H
    if (ctx->cfg.cert_file && ctx->cfg.key_file)
    {
        node_ctx_p->ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(1, ctx->cfg.ca_file, ctx->cfg.ca_path, ctx->cfg.cert_file, ctx->cfg.key_file, NULL);
        if (!node_ctx_p->ssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载证书或私钥失败，无法创建SSL安全环境。");
            return -2;
        }
    }
#endif // HEADER_SSL_H

    cb.prepare_cb = _abcdk_tipc_prepare_cb;
    cb.event_cb = _abcdk_tipc_event_cb;
    cb.request_cb = _abcdk_tipc_request_cb;

    chk = abcdk_asynctcp_listen(node_p, node_ctx_p->ssl_ctx, addr, &cb);
    abcdk_asynctcp_unref(&node_p);
    if (chk == 0)
        return 0;

    return -3;
}

int abcdk_tipc_connect(abcdk_tipc_t *ctx, const char *location, uint64_t id)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_sockaddr_t addr = {0};
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    assert(ctx != NULL && location != NULL && id > 0);
    assert(ctx->cfg.id != id);

    chk = abcdk_sockaddr_from_string(&addr, location, 1);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "连接地址'%s'无法识别。", location);
        return -4;
    }

    node_p = _abcdk_tipc_node_create(ctx,2);
    if (!node_p)
        return -1;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->id = id;
    node_ctx_p->mark = rand()*rand();

#ifdef HEADER_SSL_H
    if (ctx->cfg.cert_file && ctx->cfg.key_file)
    {
        node_ctx_p->ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(0, ctx->cfg.ca_file, ctx->cfg.ca_path, ctx->cfg.cert_file, ctx->cfg.key_file, NULL);
        if (!node_ctx_p->ssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载证书或私钥失败，无法创建SSL安全环境。");
            return -2;
        }
    }
#endif // HEADER_SSL_H

    _abcdk_tipc_slave_register(node_ctx_p->father, node_p, location);

    cb.prepare_cb = _abcdk_tipc_prepare_cb;
    cb.event_cb = _abcdk_tipc_event_cb;
    cb.request_cb = _abcdk_tipc_request_cb;

    chk = abcdk_asynctcp_connect(node_p, node_ctx_p->ssl_ctx, &addr, &cb);
    abcdk_asynctcp_unref(&node_p);
    if (chk == 0)
        return 0;

    return -3; 
}

static int _abcdk_tipc_post_register(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);


    /*
     * |Length  |CMD    |Mine ID  |Mark    |Your ID |
     * |4 Bytes |1 Byte |8 Bytes  |8 Bytes |8 Bytes |
     *
     * Length：长度(不包含自身)。
     * CMD：0 注册。
     * ID：我的ID。
     * Mark：我的标识。
     * ID：你的ID。
     */

    msg_p = abcdk_object_alloc2(4+1+8+8+8);
    if(!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 0, 32, msg_p->sizes[0]-4);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 32, 8, 0);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 40, 64, node_ctx_p->father->cfg.id);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 104, 64, node_ctx_p->mark);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 168, 64, node_ctx_p->id);

    chk = abcdk_asynctcp_post(node, msg_p);
    if(chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_request_process_register(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t myid;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    node_ctx_p->id = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    node_ctx_p->mark = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 104, 64);
    myid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 168, 64);

    if(myid == node_ctx_p->father->cfg.id)
    {
        _abcdk_tipc_slave_register(node_ctx_p->father, node, NULL);
        return;
    }

    abcdk_trace_output(LOG_WARNING,"本地ID标识在远端服务(ID=%llu)中的记录不相同，不允许注册。",node_ctx_p->id);

    abcdk_asynctcp_set_timeout(node,1);

}

static int _abcdk_tipc_post_request(abcdk_asynctcp_node_t *node, int rsp, uint64_t mid, const void *data, size_t size)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    /*
     * |Length  |CMD    |MID     |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length： 不包含自身。
     * CMD：1 请求，2 应答。
     * MID：消息ID。
     * DATA: 变长数据。
     */

    msg_p = abcdk_object_alloc2(4 + 1 + 8 + size);
    if (!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 0, 32, msg_p->sizes[0] - 4);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 32, 8, (rsp ? 2 : 1));
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 40, 64, mid);
    memcpy(msg_p->pptrs[0] + 13, data, size);

    chk = abcdk_asynctcp_post(node, msg_p);
    if (chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_request_process_request(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t mid;
    void *data_p;
    size_t data_l;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    mid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    data_p = ABCDK_PTR2VPTR(req_data, 13);
    data_l = req_size - 13;

    node_ctx_p->father->cfg.request_cb(node_ctx_p->father->cfg.opaque, node_ctx_p->id, mid, data_p, data_l);
}

static void _abcdk_tipc_request_process_response(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t msg_mid;
    abcdk_object_t *rsp_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    msg_mid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);

    rsp_p = abcdk_object_copyfrom(ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
    if (!rsp_p)
        return;

    chk = abcdk_waiter_response(node_ctx_p->req_waiter, msg_mid, rsp_p);
    if (chk != 0)
        abcdk_object_unref(&rsp_p);
}

static void _abcdk_tpic_request_process(abcdk_asynctcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint32_t msg_len;
    uint8_t msg_cmd;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    msg_len = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 0, 32);
    msg_cmd = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 32, 8);

    if (msg_cmd == 0)
    {
        _abcdk_tipc_request_process_register(node);
    }
    else if (msg_cmd == 1)
    {
        _abcdk_tipc_request_process_request(node);
    }
    else if (msg_cmd == 2)
    {
        _abcdk_tipc_request_process_response(node);
    }
    else if (msg_cmd == 3)
    {
    }
}

int abcdk_tipc_request(abcdk_tipc_t *ctx, uint64_t id, const char *data, size_t size, abcdk_object_t **rsp)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *rsp_p = NULL;
    uint64_t mid;
    int chk;

    assert(ctx != NULL && id > 0 && data != NULL && size > 0);

    node_p = _abcdk_tipc_slave_find_pipe(ctx, id);
    if (!node_p)
    {
        chk = -1;
        goto END;
    }

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    mid = abcdk_atomic_fetch_and_add(&ctx->mid_next, 1);

    if (rsp)
    {
        chk = abcdk_waiter_register(node_ctx_p->req_waiter, mid);
        if (chk != 0)
        {
            chk = -2;
            goto END;
        }
    }

    chk = _abcdk_tipc_post_request(node_p, 0, mid, data, size);
    if (chk != 0)
    {
        chk = -3;
        goto END;
    }

    if (rsp)
    {
        rsp_p = (abcdk_object_t *)abcdk_waiter_wait(node_ctx_p->req_waiter, mid, -1);
        if (!rsp_p)
        {
            chk = -4;
            goto END;
        }

        *rsp = rsp_p;
    }

    chk = 0;

END:

    abcdk_asynctcp_unref(&node_p);
    return chk;
}
