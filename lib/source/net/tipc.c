/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/net/tipc.h"

#define ABCDK_TIPC_SLAVE_MAX  9999
#define ABCDK_TIPC_TOPIC_MAX  9999

/**简单的TIPC服务。*/
struct _abcdk_tipc
{
    /*配置。*/
    abcdk_tipc_config_t cfg;

    /*通讯IO。*/
    abcdk_stcp_t *io_ctx;

    /*重连定时器。*/
    abcdk_timer_t *reconnect_timer;

    /*节点列表。*/
    struct _abcdk_tipc_slave *slave_list[ABCDK_TIPC_SLAVE_MAX];

    /*节点同步锁。*/
    abcdk_mutex_t *slave_mutex;

    /*主题列表。*/
    uint8_t topic_list[ABCDK_TIPC_TOPIC_MAX/8];

    /*主题同步锁。*/
    abcdk_mutex_t *topic_mutex;

    /*下一个MID。*/
    volatile uint64_t mid_next;

}; // abcdk_tipc_t;

typedef struct _abcdk_tipc_node
{
    /*父级。*/
    abcdk_tipc_t *father;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*本机地址。*/
    char local_addr[NAME_MAX];

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*远端ID。*/
    uint64_t id;

    /*远端位置。*/
    char location[NAME_MAX];

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*请求服务员。*/
    abcdk_waiter_t *req_waiter;

} abcdk_tipc_node_t;

typedef struct _abcdk_tipc_slave
{
    /*远端ID。*/
    uint64_t id;

    /*远端位置。*/
    char location[NAME_MAX];

    /*标志。0 任意，1 被动，2 主动。*/
    int flag;

    /*状态。1 已连接，2 已断开，3 正在重建连接。*/
    int state;

    /*被动连接的管道。*/
    abcdk_stcp_node_t *pipe1;

    /*主动连接的管道。*/
    abcdk_stcp_node_t *pipe2;

    /*订阅主题列表。*/
    uint8_t topic_list[ABCDK_TIPC_TOPIC_MAX/8];

} abcdk_tipc_slave_t;

static void _abcdk_tipc_slave_free(abcdk_tipc_slave_t **ctx)
{
    abcdk_tipc_slave_t *ctx_p;

    if(!ctx||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_stcp_unref(&ctx_p->pipe1);
    abcdk_stcp_unref(&ctx_p->pipe2);
    abcdk_heap_free(ctx_p);
}

static abcdk_tipc_slave_t *_abcdk_tipc_slave_alloc(uint64_t id)
{
    abcdk_tipc_slave_t *ctx;

    ctx = (abcdk_tipc_slave_t*)abcdk_heap_alloc(sizeof(abcdk_tipc_slave_t));
    if(!ctx)
        return NULL;

    ctx->id = id;

    /*远端默认订阅所有主题。*/
    for(int i = 1;i < ABCDK_TIPC_TOPIC_MAX;i++)
        abcdk_bloom_mark(ctx->topic_list,ABCDK_ARRAY_SIZE(ctx->topic_list),i);
    
    return ctx;
}

static int _abcdk_tipc_slave_register(abcdk_tipc_t *ctx, abcdk_stcp_node_t *pipe)
{
    abcdk_tipc_slave_t *slave_ctx_p;
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(pipe);

    /*远程的ID不能与自己的ID相同。*/
    if (ctx->cfg.id == node_ctx_p->id)
        return -1;
    
    /*可能还未初始化。*/
    if(node_ctx_p->id <= 0 || node_ctx_p->id >=ABCDK_TIPC_SLAVE_MAX)
        chk = -2;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    /*没有则先创建。*/
    if(!ctx->slave_list[node_ctx_p->id])
        ctx->slave_list[node_ctx_p->id] = _abcdk_tipc_slave_alloc(node_ctx_p->id);

    slave_ctx_p = ctx->slave_list[node_ctx_p->id];
    if (!slave_ctx_p)
    {
        chk = -2;
        goto END;
    }

    /*标记为已连接。*/
    slave_ctx_p->state = 1;

    if(slave_ctx_p->location[0] == '\0')
        strncpy(slave_ctx_p->location,node_ctx_p->location,NAME_MAX);

    if (node_ctx_p->flag == 1)
    {
        if (slave_ctx_p->pipe1 == NULL)
        {
            slave_ctx_p->pipe1 = abcdk_stcp_refer(pipe);
        }
        else if (slave_ctx_p->pipe1 != pipe)
        {
            chk = -3;
            goto END;
        }
    }
    else
    {
        if (slave_ctx_p->pipe2 == NULL)
        {
            slave_ctx_p->pipe2 = abcdk_stcp_refer(pipe);
        }
        else if (slave_ctx_p->pipe2 != pipe)
        {
            chk = -3;
            goto END;
        }
    }

    /* 当双向建立连接成功时，保留由ID大到小的主动连接。*/
    if (slave_ctx_p->pipe1 && slave_ctx_p->pipe2)
    {
        if (ctx->cfg.id > node_ctx_p->id)
        {
            abcdk_stcp_set_timeout(slave_ctx_p->pipe1, -1);
            abcdk_stcp_unref(&slave_ctx_p->pipe1);

            slave_ctx_p->flag = 2;
        }
        else
        {
            abcdk_stcp_set_timeout(slave_ctx_p->pipe2, -1);
            abcdk_stcp_unref(&slave_ctx_p->pipe2);

            slave_ctx_p->flag = 1;
        }
    }

    chk = 0;

END:

    abcdk_mutex_unlock(ctx->slave_mutex);
    return chk;
}

/*
 * 如果节点存在返回可用的管道数量，否则返回一个负值。
*/
static int _abcdk_tipc_slave_unregister(abcdk_tipc_t *ctx, abcdk_stcp_node_t *pipe)
{
    abcdk_object_t *slave_p;
    abcdk_tipc_slave_t *slave_ctx_p;
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(pipe);

    /*可能还未初始化。*/
    if(node_ctx_p->id <= 0 || node_ctx_p->id >=ABCDK_TIPC_SLAVE_MAX)
        return -2;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    /*没有则先创建。*/
    if(!ctx->slave_list[node_ctx_p->id])
        ctx->slave_list[node_ctx_p->id] = _abcdk_tipc_slave_alloc(node_ctx_p->id);

    slave_ctx_p = ctx->slave_list[node_ctx_p->id];
    if (!slave_ctx_p)
    {
        chk = -1;
        goto END;
    }

    if(slave_ctx_p->location[0] == '\0')
        strncpy(slave_ctx_p->location,node_ctx_p->location,NAME_MAX);

    if (slave_ctx_p->pipe1 == pipe)
    {
        abcdk_stcp_unref(&slave_ctx_p->pipe1);
    }
    else if (slave_ctx_p->pipe2 == pipe)
    {
        abcdk_stcp_unref(&slave_ctx_p->pipe2);
    }

    /*还剩几个。*/
    chk = (slave_ctx_p->pipe1 ? 1 : 0) + (slave_ctx_p->pipe2 ? 1 : 0);

    /*当所有连接都已经断开后，标记为已断开。*/
    if(chk == 0)
        slave_ctx_p->state = 2;
    
END:

    abcdk_mutex_unlock(ctx->slave_mutex);
    return chk;
}

static abcdk_stcp_node_t *_abcdk_tipc_slave_find_pipe(abcdk_tipc_t *ctx,uint64_t id)
{
    abcdk_stcp_node_t *node_p = NULL;
    abcdk_tipc_slave_t *slave_ctx_p = NULL;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_ctx_p = ctx->slave_list[id];
    if (!slave_ctx_p)
        goto END;

    /*优先选择可能被保留的链路。*/
    if(ctx->cfg.id > id)
        node_p = (slave_ctx_p->pipe2?slave_ctx_p->pipe2:slave_ctx_p->pipe1);
    else 
        node_p = (slave_ctx_p->pipe1?slave_ctx_p->pipe1:slave_ctx_p->pipe2);

    /*增加引用计数。*/
    if(node_p)
        node_p = abcdk_stcp_refer(node_p);

END:

    abcdk_mutex_unlock(ctx->slave_mutex);

    return node_p;
}

static uint64_t _abcdk_tipc_slave_reconnect_cb(void *opaque)
{
    abcdk_tipc_t *ctx = (abcdk_tipc_t *)opaque;
    abcdk_tipc_slave_t *slave_ctx_p = NULL;
    uint64_t id = 1;
    char location[NAME_MAX] = {0};
    int flag;
    int chk;

NEXT:
    
    abcdk_mutex_lock(ctx->slave_mutex, 1);

    for (; id < ABCDK_TIPC_SLAVE_MAX; id++)
    {
        slave_ctx_p = ctx->slave_list[id];
        if (!slave_ctx_p)
            continue;

        /*未断开的不需要处理。*/
        if (slave_ctx_p->state != 2)
            continue;

        /*标志为正在重连。*/
        slave_ctx_p->state = 3;

        /*copy*/
        flag = slave_ctx_p->flag;
        strncpy(location,slave_ctx_p->location,NAME_MAX);
        break;
    }

    abcdk_mutex_unlock(ctx->slave_mutex);

    /*遍历完在，返回。*/
    if( id >= ABCDK_TIPC_SLAVE_MAX )
        return 5000;

    /*如果是被动连接，则等待另外一端发起连接。*/
    if(flag == 1)
        goto NEXT;

    chk = abcdk_tipc_connect(ctx,location,id);
    ABCDK_ASSERT((chk == 0 || chk == -4),"不应当在这里出错的。");

    goto NEXT;
}

static int _abcdk_tipc_slave_subscribe(abcdk_tipc_t *ctx,uint64_t id,uint64_t topic,int unset)
{
    abcdk_tipc_slave_t *slave_ctx_p = NULL;
    int chk = -1;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_ctx_p = ctx->slave_list[id];
    if (!slave_ctx_p)
        goto END;

    if(unset)
        chk = abcdk_bloom_unset(slave_ctx_p->topic_list,ABCDK_ARRAY_SIZE(slave_ctx_p->topic_list),topic);
    else 
        chk = abcdk_bloom_mark(slave_ctx_p->topic_list,ABCDK_ARRAY_SIZE(slave_ctx_p->topic_list),topic);

END:

    abcdk_mutex_unlock(ctx->slave_mutex);

    return chk;
}

static abcdk_stcp_node_t *_abcdk_tipc_slave_topic_find_pipe(abcdk_tipc_t *ctx,uint64_t id,uint64_t topic)
{
    abcdk_stcp_node_t *node_p = NULL;
    abcdk_tipc_slave_t *slave_ctx_p = NULL;
    int chk = -1;

    abcdk_mutex_lock(ctx->slave_mutex, 1);

    slave_ctx_p = ctx->slave_list[id];
    if (!slave_ctx_p)
        goto END;

    chk = abcdk_bloom_filter(slave_ctx_p->topic_list,ABCDK_ARRAY_SIZE(slave_ctx_p->topic_list),topic);
    if(chk != 1)
        goto END;

    /*优先选择可能被保留的链路。*/
    if(ctx->cfg.id > id)
        node_p = (slave_ctx_p->pipe2?slave_ctx_p->pipe2:slave_ctx_p->pipe1);
    else 
        node_p = (slave_ctx_p->pipe1?slave_ctx_p->pipe1:slave_ctx_p->pipe2);

    /*增加引用计数。*/
    if(node_p)
        node_p = abcdk_stcp_refer(node_p);

END:

    abcdk_mutex_unlock(ctx->slave_mutex);

    return node_p;
}


static int _abcdk_tipc_subscribe(abcdk_tipc_t *ctx,uint64_t topic,int unset)
{
    int chk;

    abcdk_mutex_lock(ctx->topic_mutex, 1);

    if(unset)
        chk = abcdk_bloom_unset(ctx->topic_list,ABCDK_ARRAY_SIZE(ctx->topic_list),topic);
    else 
        chk = abcdk_bloom_mark(ctx->topic_list,ABCDK_ARRAY_SIZE(ctx->topic_list),topic);

    abcdk_mutex_unlock(ctx->topic_mutex);

    return chk;
}

/**
 * 验证本机节点订阅主题。
 * 
 * @return 0 未订阅，1 已订阅。
*/
static int _abcdk_tipc_subscribe_filter(abcdk_tipc_t *ctx,uint64_t topic)
{
    int chk;

    abcdk_mutex_lock(ctx->topic_mutex, 1);

    chk = abcdk_bloom_filter(ctx->topic_list,ABCDK_ARRAY_SIZE(ctx->topic_list),topic);

    abcdk_mutex_unlock(ctx->topic_mutex);

    return chk;
}


void abcdk_tipc_destroy(abcdk_tipc_t **ctx)
{
    abcdk_tipc_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_timer_destroy(&ctx_p->reconnect_timer);
    abcdk_stcp_stop(ctx_p->io_ctx);
    abcdk_stcp_destroy(&ctx_p->io_ctx);
    abcdk_mutex_destroy(&ctx_p->slave_mutex);
    abcdk_mutex_destroy(&ctx_p->topic_mutex);

    for (int i = 1; i < ABCDK_TIPC_SLAVE_MAX; i++)
        _abcdk_tipc_slave_free(&ctx_p->slave_list[i]);

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

    ctx->io_ctx = abcdk_stcp_create(sysconf(_SC_NPROCESSORS_ONLN));
    memset(ctx->slave_list,0,sizeof(abcdk_tipc_slave_t*)* ABCDK_ARRAY_SIZE(ctx->slave_list));
    ctx->slave_mutex = abcdk_mutex_create();
    ctx->topic_mutex = abcdk_mutex_create();
    ctx->mid_next = 1;

    /*本机默认不订阅任何主题。*/
    for(int i = 1;i < ABCDK_TIPC_TOPIC_MAX;i++)
        abcdk_bloom_unset(ctx->topic_list,ABCDK_ARRAY_SIZE(ctx->topic_list),i);

    /*重连定时器。*/
    ctx->reconnect_timer = abcdk_timer_create(_abcdk_tipc_slave_reconnect_cb,ctx);

    return ctx;
}

static void _abcdk_tipc_node_destroy_cb(void *userdata)
{
    abcdk_tipc_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_tipc_node_t *)userdata;

    abcdk_receiver_unref(&ctx->req_data);
    abcdk_waiter_free(&ctx->req_waiter);
}

static void _abcdk_tipc_node_waiter_destroy_cb(void *msg)
{
    abcdk_object_t *obj_p;

    obj_p = (abcdk_object_t *)msg;

    abcdk_object_unref(&obj_p);
}

static abcdk_stcp_node_t *_abcdk_tipc_node_create(abcdk_tipc_t *ctx,int flag)
{
    abcdk_stcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;

    node_p = abcdk_stcp_alloc(ctx->io_ctx, sizeof(abcdk_tipc_node_t), _abcdk_tipc_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->flag = flag;
    node_ctx_p->id = 0;
    if(flag != 0)
        node_ctx_p->req_waiter = abcdk_waiter_alloc(_abcdk_tipc_node_waiter_destroy_cb);

    return node_p;
}

static void _abcdk_tipc_prepare_cb(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen)
{
    abcdk_stcp_node_t *node_p;
    abcdk_tipc_node_t *listen_ctx_p, *node_ctx_p;
    int chk;

    listen_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(listen);

    node_p = _abcdk_tipc_node_create(listen_ctx_p->father,1);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_tipc_event_accept(abcdk_stcp_node_t *node, int *result)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    abcdk_stcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    /*默认：允许。*/
    *result = 0;

    if (node_ctx_p->father->cfg.accept_cb)
        node_ctx_p->father->cfg.accept_cb(node_ctx_p->father->cfg.opaque, node_ctx_p->remote_addr, result);

    if (*result != 0)
        abcdk_trace_output(LOG_INFO, "禁止客户端('%s')连接到本机。", node_ctx_p->remote_addr);
}

static int _abcdk_tipc_post_register(abcdk_stcp_node_t *node,int rsp);


static void _abcdk_tipc_event_connect(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    abcdk_trace_output(LOG_INFO, "本机(%s)与远端(%s)的连接已建立。", node_ctx_p->local_addr, node_ctx_p->remote_addr);

    /*设置超时。*/
    abcdk_stcp_set_timeout(node, 24 * 3600);

    /*发送注册消息。*/
    if (node_ctx_p->flag == 2)
        _abcdk_tipc_post_register(node,0);
    
    /*已连接到远端，注册读写事件。*/
    abcdk_stcp_recv_watch(node);
    abcdk_stcp_send_watch(node);
}

static void _abcdk_tipc_event_output(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);
}

static void _abcdk_tipc_event_close(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    if (node_ctx_p->flag == 0)
    {
        abcdk_trace_output(LOG_INFO, "监听关闭，忽略。");
        return;
    }

    abcdk_trace_output(LOG_INFO, "本机(%s)与远端(%s)的连接已断开。",node_ctx_p->local_addr, node_ctx_p->remote_addr);

    /*取消此链路上的所有等待的。*/
    abcdk_waiter_cancel(node_ctx_p->req_waiter);

    /*反注册并返回可用的管道数量。*/
    chk = _abcdk_tipc_slave_unregister(node_ctx_p->father, node);
    if (chk < 0)
        return;
    if (chk > 0)
        return;

    /*当节点的通讯链路全部断掉后，发出通知。*/
    if(node_ctx_p->father->cfg.offline_cb)
        node_ctx_p->father->cfg.offline_cb(node_ctx_p->father->cfg.opaque,node_ctx_p->id);

}

static void _abcdk_tipc_event_cb(abcdk_stcp_node_t *node, uint32_t event, int *result)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    if (!node_ctx_p->remote_addr[0])
        abcdk_stcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    if (!node_ctx_p->local_addr[0])
        abcdk_stcp_get_sockaddr_str(node, node_ctx_p->local_addr, NULL);

    if (event == ABCDK_STCP_EVENT_ACCEPT)
    {
        _abcdk_tipc_event_accept(node, result);
    }
    else if (event == ABCDK_STCP_EVENT_CONNECT)
    {
        _abcdk_tipc_event_connect(node);
    }
    else if (event == ABCDK_STCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_STCP_EVENT_OUTPUT)
    {
        _abcdk_tipc_event_output(node);
    }
    else if (event == ABCDK_STCP_EVENT_CLOSE || event == ABCDK_STCP_EVENT_INTERRUPT)
    {
        _abcdk_tipc_event_close(node);
    }
}

static void _abcdk_tipc_process(abcdk_stcp_node_t *node);

static void _abcdk_tipc_input_cb(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_tipc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (!node_ctx_p->req_data)
        node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_SMB, 16 * 1024 * 1024, NULL);

    if (!node_ctx_p->req_data)
        goto ERR;

    chk = abcdk_receiver_append(node_ctx_p->req_data, data, size, remain);
    if (chk < 0)
    {
        *remain = 0;
        goto ERR;
    }
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_tipc_process(node);

    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);

    /*No Error.*/
    return;

ERR:

    abcdk_stcp_set_timeout(node, -1);
}


int abcdk_tipc_listen(abcdk_tipc_t *ctx, abcdk_sockaddr_t *addr)
{
    abcdk_stcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_stcp_config_t asio_cfg = {0};
    int chk;

    assert(ctx != NULL && addr != NULL);

    node_p = _abcdk_tipc_node_create(ctx,0);
    if (!node_p)
        return -1;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

    asio_cfg.ssl_scheme = ctx->cfg.ssl_scheme;
    asio_cfg.pki_ca_file = ctx->cfg.pki_ca_file;
    asio_cfg.pki_ca_path = ctx->cfg.pki_ca_path;
    asio_cfg.pki_cert_file = ctx->cfg.pki_cert_file;
    asio_cfg.pki_key_file = ctx->cfg.pki_key_file;
    asio_cfg.pki_key_passwd = ctx->cfg.pki_key_passwd;
    asio_cfg.pki_check_cert = ctx->cfg.pki_check_cert;
    asio_cfg.ske_key_file = ctx->cfg.ske_key_file;

    asio_cfg.bind_addr = *addr;

    asio_cfg.prepare_cb = _abcdk_tipc_prepare_cb;
    asio_cfg.event_cb = _abcdk_tipc_event_cb;
    asio_cfg.input_cb = _abcdk_tipc_input_cb;

    chk = abcdk_stcp_listen(node_p, &asio_cfg);
    abcdk_stcp_unref(&node_p);
    if (chk != 0)
        return -3;

    return 0;
}

int abcdk_tipc_connect(abcdk_tipc_t *ctx, const char *location, uint64_t id)
{
    abcdk_stcp_node_t *node_p;
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_sockaddr_t addr = {0};
    abcdk_stcp_config_t asio_cfg = {0};
    int chk;

    assert(ctx != NULL && location != NULL && id > 0 && id < ABCDK_TIPC_SLAVE_MAX);
    ABCDK_ASSERT(ctx->cfg.id != id,"远端ID不能与本机ID相同。");

    chk = abcdk_sockaddr_from_string(&addr, location, 1);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "远端(ID=%llu,IP='%s')的地址无法识别。",id, location);
        return -4;
    }

    node_p = _abcdk_tipc_node_create(ctx,2);
    if (!node_p)
        return -1;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->id = id;
    strncpy(node_ctx_p->location,location,NAME_MAX);

    asio_cfg.ssl_scheme = ctx->cfg.ssl_scheme;
    asio_cfg.pki_ca_file = ctx->cfg.pki_ca_file;
    asio_cfg.pki_ca_path = ctx->cfg.pki_ca_path;
    asio_cfg.pki_cert_file = ctx->cfg.pki_cert_file;
    asio_cfg.pki_key_file = ctx->cfg.pki_key_file;
    asio_cfg.pki_key_passwd = ctx->cfg.pki_key_passwd;
    asio_cfg.pki_check_cert = ctx->cfg.pki_check_cert;
    asio_cfg.ske_key_file = ctx->cfg.ske_key_file;

    asio_cfg.prepare_cb = _abcdk_tipc_prepare_cb;
    asio_cfg.event_cb = _abcdk_tipc_event_cb;
    asio_cfg.input_cb = _abcdk_tipc_input_cb;

    chk = abcdk_stcp_connect(node_p, &addr, &asio_cfg);
    abcdk_stcp_unref(&node_p);
    if (chk != 0)
        return -3;

    return 0; 
}

static int _abcdk_tipc_post_register(abcdk_stcp_node_t *node,int rsp)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);


    /*
     * |Length  |CMD    |Mine ID  |Your ID |
     * |4 Bytes |1 Byte |8 Bytes  |8 Bytes |
     *
     * Length：长度(不包含自身)。
     * CMD：1 注册，2 应答。
     * ID：我的ID。
     * ID：你的ID。
     */

    msg_p = abcdk_object_alloc2(4+1+8+8);
    if(!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 0, 32, msg_p->sizes[0]-4);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 32, 8, (rsp?2:1));
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 40, 64, node_ctx_p->father->cfg.id);
    abcdk_bloom_write_number(msg_p->pptrs[0],msg_p->sizes[0], 104, 64, node_ctx_p->id);

    chk = abcdk_stcp_post(node, msg_p,1);
    if(chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_process_register_req(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t myid;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    node_ctx_p->id = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    myid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 104, 64);

    if(myid == node_ctx_p->father->cfg.id)
    {
        chk = _abcdk_tipc_slave_register(node_ctx_p->father, node);
        if(chk == 0)
        {
            _abcdk_tipc_post_register(node,1);
            return;
        }

        if(chk == -1)
            abcdk_trace_output(LOG_WARNING,"本机ID(%llu)与远端ID(%llu)相同，不允许注册。",node_ctx_p->father->cfg.id,node_ctx_p->id);
        else if(chk == -3)
            abcdk_trace_output(LOG_WARNING,"相同的远端ID(%llu)已经注册并且在线，不允许注册。",node_ctx_p->id);
        else 
            abcdk_trace_output(LOG_WARNING,"其它错误。");
    }
    else
    {
        abcdk_trace_output(LOG_WARNING,"本机ID(%llu)在远端(ID=%llu)登记错误(ID=%llu)，不允许注册。",node_ctx_p->father->cfg.id,node_ctx_p->id,myid);
    }

    abcdk_stcp_set_timeout(node,-1);
}


static void _abcdk_tipc_process_register_rsp(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t myid;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    /*有应答说明连接没问题，直接完成注册即可。*/
    _abcdk_tipc_slave_register(node_ctx_p->father, node);
}

static int _abcdk_tipc_post_message(abcdk_stcp_node_t *node, uint64_t rsp, uint64_t mid, const void *data, size_t size,int key)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    /*
     * |Length  |CMD    |MID     |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length： 不包含自身。
     * CMD：3 请求，4 应答。
     * MID：消息ID。
     * DATA: 变长数据。
     */

    assert((4 + 1 + 8 + size) <= 0xffffff);

    msg_p = abcdk_object_alloc2(4 + 1 + 8 + size);
    if (!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 0, 32, msg_p->sizes[0] - 4);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 32, 8, (rsp ? 4 : 3));
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 40, 64, mid);
    memcpy(msg_p->pptrs[0] + 13, data, size);

    chk = abcdk_stcp_post(node, msg_p,key);
    if (chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_process_message_req(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t msg_mid;
    void *data_p;
    size_t data_l;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    msg_mid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    data_p = ABCDK_PTR2VPTR(req_data, 13);
    data_l = req_size - 13;

    node_ctx_p->father->cfg.request_cb(node_ctx_p->father->cfg.opaque, node_ctx_p->id, msg_mid, data_p, data_l);
}

static void _abcdk_tipc_process_message_rsp(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t msg_mid;
    abcdk_object_t *rsp_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

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

static int _abcdk_tipc_post_subscribe(abcdk_stcp_node_t *node, uint64_t topic,int unset)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    /*
     * |Length  |CMD    |TOPIC    |UNSET  |
     * |4 Bytes |1 Byte |8 Bytes  |1 Byte |
     *
     * Length： 不包含自身。
     * CMD：5 订阅。
     * TOPIC: 主题。
     * UNSET：0 订阅，1 取消。
     */

    msg_p = abcdk_object_alloc2(4 + 1 + 8 + 1);
    if (!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 0, 32, msg_p->sizes[0] - 4);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 32, 8, 5);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 40, 64, topic);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 104, 8, unset);

    chk = abcdk_stcp_post(node, msg_p,1);
    if (chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_process_subscribe(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t topic;
    int unset;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    topic = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    unset = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 104, 8);
    
    /*更新远端主题订阅列表。*/
    _abcdk_tipc_slave_subscribe(node_ctx_p->father,node_ctx_p->id,topic,unset);
  
    abcdk_trace_output(LOG_INFO,"远端(ID=%llu)%s主题(%llu)。",node_ctx_p->id,(unset?"取订":"订阅"), topic);
}

static int _abcdk_tipc_post_publish(abcdk_stcp_node_t *node, uint64_t topic, const void *data, size_t size)
{
    abcdk_tipc_node_t *node_ctx_p;
    abcdk_object_t *msg_p;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    /*
     * |Length  |CMD    |TOPIC   |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length： 不包含自身。
     * CMD：6 发布。
     * TOPIC：主题。
     * DATA: 变长数据。
     */

    assert((4 + 1 + 8 + size) <= 0xffffff);

    msg_p = abcdk_object_alloc2(4 + 1 + 8 + size);
    if (!msg_p)
        return -1;

    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 0, 32, msg_p->sizes[0] - 4);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 32, 8, 6);
    abcdk_bloom_write_number(msg_p->pptrs[0], msg_p->sizes[0], 40, 64, topic);
    memcpy(msg_p->pptrs[0] + 13, data, size);

    chk = abcdk_stcp_post(node, msg_p,0);
    if (chk == 0)
        return 0;

    abcdk_object_unref(&msg_p);
    return -2;
}

static void _abcdk_tipc_process_publish(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint64_t topic;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    topic = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);
    
    chk = _abcdk_tipc_subscribe_filter(node_ctx_p->father,topic);
    if(chk == 0)
    {
        /*通知远端主题订阅发生变更，不需要再向当前节点发布信息，以便节省带宽。*/
        _abcdk_tipc_post_subscribe(node,topic,1);
    }
    else
    {
        if(node_ctx_p->father->cfg.subscribe_cb)
            node_ctx_p->father->cfg.subscribe_cb(node_ctx_p->father->cfg.opaque,node_ctx_p->id,topic,ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
    }
}

static void _abcdk_tipc_process(abcdk_stcp_node_t *node)
{
    abcdk_tipc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint32_t len;
    uint8_t cmd;
    int chk;

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

    len = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 0, 32);
    cmd = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 32, 8);

    if (cmd == 1)
    {
        _abcdk_tipc_process_register_req(node);
    }
    else if (cmd == 2)
    {
        _abcdk_tipc_process_register_rsp(node);
    }
    else if (cmd == 3)
    {
        _abcdk_tipc_process_message_req(node);
    }
    else if (cmd == 4)
    {
        _abcdk_tipc_process_message_rsp(node);
    }
    else if (cmd == 5)
    {
        _abcdk_tipc_process_subscribe(node);
    }
    else if (cmd == 6)
    {
        _abcdk_tipc_process_publish(node);
    }
}

int abcdk_tipc_request(abcdk_tipc_t *ctx, uint64_t id, const char *data, size_t size, abcdk_object_t **rsp)
{
    abcdk_stcp_node_t *node_p = NULL;
    abcdk_tipc_node_t *node_ctx_p = NULL;
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

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

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

    chk = _abcdk_tipc_post_message(node_p, 0, mid, data, size,rsp?1:0);
    if (chk != 0)
    {
        chk = -3;
        goto END;
    }

    if (rsp)
    {
        rsp_p = (abcdk_object_t *)abcdk_waiter_wait(node_ctx_p->req_waiter, mid, 24*60*60*1000L);
        if (!rsp_p)
        {
            chk = -4;
            goto END;
        }

        *rsp = rsp_p;
    }

    chk = 0;

END:

    abcdk_stcp_unref(&node_p);
    return chk;
}

int abcdk_tipc_response(abcdk_tipc_t *ctx,uint64_t id,uint64_t mid, const char *data,size_t size)
{
    abcdk_stcp_node_t *node_p = NULL;
    abcdk_tipc_node_t *node_ctx_p = NULL;
    int chk;

    assert(ctx != NULL && id > 0 && data != NULL && size > 0);

    node_p = _abcdk_tipc_slave_find_pipe(ctx, id);
    if (!node_p)
    {
        chk = -1;
        goto END;
    }

    node_ctx_p = (abcdk_tipc_node_t *)abcdk_stcp_get_userdata(node_p);

    chk = _abcdk_tipc_post_message(node_p, 1, mid, data, size,1);
    if (chk != 0)
    {
        chk = -3;
        goto END;
    }

    chk = 0;

END:

    abcdk_stcp_unref(&node_p);
    return chk;
}

int abcdk_tipc_subscribe(abcdk_tipc_t *ctx, uint64_t topic, int unset)
{
    abcdk_stcp_node_t *node_p = NULL;
    int chk;

    assert(ctx != NULL && topic > 0 && topic < ABCDK_TIPC_TOPIC_MAX);

    _abcdk_tipc_subscribe(ctx, topic, unset);

    /*遍历远端，逐个通知。*/
    for (int i = 1; i < ABCDK_TIPC_SLAVE_MAX; i++)
    {
        node_p = _abcdk_tipc_slave_find_pipe(ctx, i);
        if (!node_p)
            continue;

        _abcdk_tipc_post_subscribe(node_p, topic, unset);
        abcdk_stcp_unref(&node_p);
    }

    return 0;
}

int abcdk_tipc_publish(abcdk_tipc_t *ctx, uint64_t topic, const char *data, size_t size)
{
    abcdk_stcp_node_t *node_p = NULL;
    int chk;

    assert(ctx != NULL && topic > 0 && topic < ABCDK_TIPC_TOPIC_MAX && data != NULL && size > 0);

    /*遍历远端，逐个发布。*/
    for (int i = 1; i < ABCDK_TIPC_SLAVE_MAX; i++)
    {
        node_p = _abcdk_tipc_slave_topic_find_pipe(ctx, i, topic);
        if (!node_p)
            continue;

        _abcdk_tipc_post_publish(node_p, topic, data, size);
        abcdk_stcp_unref(&node_p);
    }

    return 0;
}