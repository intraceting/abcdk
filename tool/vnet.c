/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

/** 常量。*/
enum _abcdkvnet_constant
{
    /** 服务端端。*/
    ABCDKVNET_ROLE_SERVER = 1,
#define ABCDKVNET_ROLE_SERVER ABCDKVNET_ROLE_SERVER

    /** 客户端。*/
    ABCDKVNET_ROLE_CLIENT = 2,
#define ABCDKVNET_ROLE_CLIENT ABCDKVNET_ROLE_CLIENT

    /**请求IP地址.*/
    ABCDKVNET_CMD_ASK_IP = 1,
#define ABCDKVNET_CMD_ASK_IP ABCDKVNET_CMD_ASK_IP
};

/*节点。*/
typedef struct _abcdkvnet_node
{
    /*是否为动态地址。*/
    int addr_is_dhcp;

    /*地址。*/
    abcdk_sockaddr_t addr4;
    abcdk_sockaddr_t addr6;

    /*链路。*/
    abcdk_srpc_session_t *pipe;

}abcdkvnet_node_t;

/*简单的虚拟网络。*/
typedef struct _abcdkvnet
{
    int errcode;
    abcdk_option_t *args;

    /*日志。*/
    abcdk_logger_t *logger;
   
    /*IP池。*/
    abcdk_ipool_t *ipv4_pool;
    abcdk_ipool_t *ipv6_pool;

    /*路由表。*/
    abcdkvnet_node_t *route_list;
    abcdk_mutex_t *route_mutex;

    /*RPC.*/
    abcdk_srpc_t *srpc_ctx;

    /*原始监听对象。*/
    abcdk_srpc_session_t *listen_raw_p;

    /*PKI监听对象。*/
    abcdk_srpc_session_t *listen_pki_p;

    /*ENIGMA监听对象。*/
    abcdk_srpc_session_t *listen_enigma_p;

    /*PKIonENIGMA监听对象。*/
    abcdk_srpc_session_t *listen_pki_enigma_p;

    /*上行对象。*/
    abcdk_srpc_session_t *uplink_p;

    /*上行地址。*/
    abcdk_object_t *uplink_addr;

    /*虚拟地址。*/
    abcdk_sockaddr_t virtual_addr4;
    abcdk_sockaddr_t virtual_addr6;
    
    int max_client;
    
    /*角色。*/
    int role;

    /*监听地址。*/
    const char *listen_raw;

    /*PKI监听地址。*/
    const char *listen_pki;

    /*ENIGMA监听地址。*/
    const char *listen_enigma;

    /*PKIonENIGMA监听地址。*/
    const char *listen_pki_enigma;

    /*CA证书。*/
    const char *pki_ca_file;

    /*CA路径。*/
    const char *pki_ca_path;

    /*证书。*/
    const char *pki_cert_file;

    /*私钥。*/
    const char *pki_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int pki_check_cert;

    /*共享密钥。*/
    const char *enigma_key_file;

    /*盐的长度。*/
    int enigma_salt_size;

    /*上级地址。*/
    const char *uplink;


}abcdkvnet_t;



static void _abcdkvnet_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的虚拟网络。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径。默认：/tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒)，1～60之间有效。默认：30\n");
    fprintf(stderr, "\t\t注：此功能不支持supervisor或类似的工具。\n");

    fprintf(stderr, "\n\t--role < TYPE >\n");
    fprintf(stderr, "\t\t角色。默认：%d\n",ABCDKVNET_ROLE_CLIENT);

    fprintf(stderr, "\n\t\t服务端：%d\n",ABCDKVNET_ROLE_SERVER);
    fprintf(stderr, "\t\t客户端：%d\n",ABCDKVNET_ROLE_CLIENT);

    fprintf(stderr, "\n\t--ipv4-pool-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPv4起始地址。默认：ipv4://10.0.0.1\n");

    fprintf(stderr, "\n\t--ipv4-pool-end < ADDR >\n");
    fprintf(stderr, "\t\tIPv4结束地址。默认：ipv4://10.0.0.255\n");
    
    fprintf(stderr, "\n\t--ipv6-pool-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPv6起始地址。默认：ipv6://[fc00::1]\n");

    fprintf(stderr, "\n\t--ipv6-pool-end < ADDR >\n");
    fprintf(stderr, "\t\tIPv6结束地址。默认：ipv6://[fc00::ff]\n");

    fprintf(stderr, "\n\t\t例：ipv4://IP\n");
    fprintf(stderr, "\t\t例：ipv6://[IP]\n");

    fprintf(stderr, "\n\t--listen-raw < ADDR >\n");
    fprintf(stderr, "\t\t监听地址。\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-pki < ADDR >\n");
    fprintf(stderr, "\t\tPKI监听地址。\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--listen-enigma < ADDR >\n");
    fprintf(stderr, "\t\tENIGMA监听地址。\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-pki-enigma < ADDR >\n");
    fprintf(stderr, "\t\tPKIonENIGMA监听地址。\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t\t例：ipv4://IP:PORT\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--pki-ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书。\n");

    fprintf(stderr, "\n\t--pki-ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书，同时验证吊销列表。\n");

    fprintf(stderr, "\n\t--pki-cert-file < FILE >\n");
    fprintf(stderr, "\t\t证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--pki-key-file < FILE >\n");
    fprintf(stderr, "\t\t私钥文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--pki-check-cert < 0|1 >\n");
    fprintf(stderr, "\t\t是否验证对端证书。默认：1。\n");

    fprintf(stderr, "\n\t\t0：否\n");
    fprintf(stderr, "\t\t1：是\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--enigma-key-file < FILE >\n");
    fprintf(stderr, "\t\t共享密钥文件。\n");

    fprintf(stderr, "\n\t--enigma-salt-size < SIZE >\n");
    fprintf(stderr, "\t\t监的长度。默认：123。\n");

    fprintf(stderr, "\n\t--uplink < URL >\n");
    fprintf(stderr, "\t\t上行地址。\n");

    fprintf(stderr, "\n\t\t例：raw://DOMAIN:PORT\n");
    fprintf(stderr, "\t\t例：pki://DOMAIN:PORT\n");
    fprintf(stderr, "\t\t例：enigma://DOMAIN:PORT\n");
    fprintf(stderr, "\t\t例：pki-enigma://DOMAIN:PORT\n");
}

static int _abcdkvnet_server_ip_allocate(abcdkvnet_t *ctx, abcdk_srpc_session_t *session, abcdk_sockaddr_t *addr4, abcdk_sockaddr_t *addr6)
{
    abcdkvnet_node_t *node_p = NULL;
    int chk4,chk6;
    int chk = -1;
    
    /*分配IP地址。*/
    chk4 = abcdk_ipool_dhcp_request(ctx->ipv4_pool,addr4);
    chk6 = abcdk_ipool_dhcp_request(ctx->ipv6_pool,addr6);
    if(chk4 != 0 && chk6 != 0)
        return -11;
    
    abcdk_mutex_lock(ctx->route_mutex, 1);

    /*查找空闲的位置。*/
    for (int i = 0; i < ctx->max_client; i++)
    {
        if(ctx->route_list[i].pipe)
            continue;

        node_p = &ctx->route_list[i];
        break;
    }

    if(node_p)
    {
        /*绑定IP地址。*/
        node_p->pipe = abcdk_srpc_refer(session);
        node_p->addr4 = *addr4;
        node_p->addr6 = *addr6;
        chk = 0;
    }
    else
    {
        /*无空闲位置，回收IP地址。*/
        if(chk4 == 0)
            abcdk_ipool_reclaim(ctx->ipv4_pool,addr4);
        if(chk6 == 0)
            abcdk_ipool_reclaim(ctx->ipv6_pool,addr6);
    }

    abcdk_mutex_unlock(ctx->route_mutex);

    return chk;
}

static void _abcdkvnet_server_ip_reclaim(abcdkvnet_t *ctx, abcdk_srpc_session_t *session)
{
    abcdkvnet_node_t *node_p = NULL;

    abcdk_mutex_lock(ctx->route_mutex, 1);

    for (int i = 0; i < ctx->max_client; i++)
    {
        if(ctx->route_list[i].pipe != session)
            continue;

        node_p = &ctx->route_list[i];
        break;
    }
    
    if(node_p)
    {
        abcdk_srpc_unref(&node_p->pipe);

        if(node_p->addr4.family == AF_INET)
            abcdk_ipool_reclaim(ctx->ipv4_pool,&node_p->addr4);
        if(node_p->addr6.family == AF_INET)
            abcdk_ipool_reclaim(ctx->ipv4_pool,&node_p->addr6);

        memset(node_p,0,sizeof(abcdkvnet_node_t));
    }

    abcdk_mutex_unlock(ctx->route_mutex);
}

static int _abcdkvnet_server_cmd_1(abcdkvnet_t *ctx,abcdk_srpc_session_t *session,abcdk_object_t **rsp)
{
    abcdk_object_t *rsp_p = NULL;
    abcdk_bit_t sbit = {0};
    abcdk_sockaddr_t addr4 = {0},addr6 = {0};
    int chk;

    rsp_p = abcdk_object_alloc2(100);
    if(!rsp_p)
        return -1;

    sbit.data = rsp_p->pptrs[0];
    sbit.size = rsp_p->sizes[0];

    abcdk_bit_write_number(&sbit,16,ABCDKVNET_CMD_ASK_IP);

    if(chk != 0)
        abcdk_bit_write_number(&sbit,16,11);
    else 
        abcdk_bit_write_number(&sbit,16,0);

    abcdk_bit_write_buffer(&sbit,(uint8_t*)&addr4.addr4.sin_addr.s_addr,4);
    abcdk_bit_write_buffer(&sbit,addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);

    return 0;
}

static int _abcdkvnet_server_cmd_process(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, const void *data, size_t size,abcdk_object_t **rsp)
{
    abcdk_bit_t rbit = {0};
    uint16_t cmd;
    int chk;

    rbit.data = (void*)data;
    rbit.size = size;

    cmd = abcdk_bit_read2number(&rbit,16);

    if(cmd == ABCDKVNET_CMD_ASK_IP)
        chk = _abcdkvnet_server_cmd_1(ctx,session,rsp);

    return chk;
}

static int _abcdkvnet_client_cmd_process(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, const void *data, size_t size,abcdk_object_t **rsp)
{
    abcdk_bit_t rbit = {0};
    uint16_t cmd;
    int chk;

    rbit.data = (void*)data;
    rbit.size = size;

    cmd = abcdk_bit_read2number(&rbit,16);

    return chk;
}

static void _abcdkvnet_srpc_prepare_cb(void *opaque,abcdk_srpc_session_t **session,abcdk_srpc_session_t *listen)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
    abcdk_srpc_session_t *session_p = NULL;

    session_p = abcdk_srpc_alloc(ctx->srpc_ctx);
    if(!session_p)
        return;

    *session = session_p;
}

static void _abcdkvnet_srpc_request_cb(void *opaque, abcdk_srpc_session_t *session, uint64_t mid, const void *data, size_t size)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
    abcdk_object_t *rsp = NULL;
    int chk;

    if(ctx->role == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_server_cmd_process(ctx,session,data,size,&rsp);
    else if(ctx->role == ABCDKVNET_ROLE_CLIENT)
        _abcdkvnet_client_cmd_process(ctx,session,data,size,&rsp);
    else 
        abcdk_srpc_set_timeout(session,1);

    if(!rsp)
        return;

    abcdk_srpc_response(session,mid,rsp->pptrs[0],rsp->sizes[0]);
    abcdk_object_unref(&rsp);

}

static void _abcdkvnet_srpc_close_cb(void *opaque,abcdk_srpc_session_t *session)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;

}

static int _abcdkproxy_server_start_listen(abcdkvnet_t *ctx, int ssl_scheme)
{
    const char *listen_p = NULL;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_srpc_session_t *session_p = NULL;
    abcdk_srpc_config_t srpc_cfg = {0};
    int chk;

    if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
        listen_p = ctx->listen_raw;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        listen_p = ctx->listen_pki;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        listen_p = ctx->listen_enigma;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        listen_p = ctx->listen_pki_enigma;

    /*未启用。*/
    if(!listen_p)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen_p, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'无法识别。", listen_p);
        return -1;
    }

    if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
        session_p = ctx->listen_raw_p = abcdk_srpc_alloc(ctx->srpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        session_p = ctx->listen_pki_p = abcdk_srpc_alloc(ctx->srpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        session_p = ctx->listen_enigma_p = abcdk_srpc_alloc(ctx->srpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        session_p = ctx->listen_pki_enigma_p = abcdk_srpc_alloc(ctx->srpc_ctx);

    if (!session_p)
        return -2;

    srpc_cfg.opaque = ctx;
    srpc_cfg.ssl_scheme = ssl_scheme;
    srpc_cfg.pki_ca_file = ctx->pki_ca_file;
    srpc_cfg.pki_ca_path = ctx->pki_ca_path;
    srpc_cfg.pki_cert_file = ctx->pki_cert_file;
    srpc_cfg.pki_key_file = ctx->pki_key_file;
    srpc_cfg.pki_check_cert = ctx->pki_check_cert;
    srpc_cfg.enigma_key_file = ctx->enigma_key_file;
    srpc_cfg.enigma_salt_size = ctx->enigma_salt_size;

    srpc_cfg.prepare_cb = _abcdkvnet_srpc_prepare_cb;
    srpc_cfg.request_cb = _abcdkvnet_srpc_request_cb;
    srpc_cfg.close_cb = _abcdkvnet_srpc_close_cb;

    chk = abcdk_srpc_listen(session_p, &listen_addr, &srpc_cfg);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'失败，无权限或被占用。", listen_p);
        return -3;
    }

    return 0;
}

static void _abcdkvnet_process_server(abcdkvnet_t *ctx)
{
    int chk;

    const char *ipv4_pool_begin_p = abcdk_option_get(ctx->args,"--ipv4-pool-begin",0,"ipv4://10.0.0.1");
    const char *ipv4_pool_end_p = abcdk_option_get(ctx->args,"--ipv4-pool-end",0,"ipv4://10.0.0.255");
    const char *ipv6_pool_begin_p = abcdk_option_get(ctx->args,"--ipv6-pool-begin",0,"ipv6://[fc00::1]");
    const char *ipv6_pool_end_p = abcdk_option_get(ctx->args,"--ipv6-pool-end",0,"ipv6://[fc00::ff]");

    ctx->listen_raw = abcdk_option_get(ctx->args, "--listen-raw", 0, NULL);
    ctx->listen_pki = abcdk_option_get(ctx->args, "--listen-pki", 0, NULL);
    ctx->listen_enigma = abcdk_option_get(ctx->args, "--listen-enigma", 0, NULL);
    ctx->listen_pki_enigma = abcdk_option_get(ctx->args, "--listen-pki-enigma", 0, NULL);

    ctx->pki_ca_file = abcdk_option_get(ctx->args, "--pki-ca-file", 0, NULL);
    ctx->pki_ca_path = abcdk_option_get(ctx->args, "--pki-ca-path", 0, NULL);
    ctx->pki_cert_file = abcdk_option_get(ctx->args, "--pki-cert-file", 0, NULL);
    ctx->pki_key_file = abcdk_option_get(ctx->args, "--pki-key-file", 0, NULL);
    ctx->pki_check_cert = abcdk_option_get_int(ctx->args, "--pki-check-cert", 0, 1);

    ctx->enigma_key_file = abcdk_option_get(ctx->args, "--enigma-key-file", 0, NULL);
    ctx->enigma_salt_size = abcdk_option_get_int(ctx->args, "--enigma-salt-size", 0, 123);

    /*修复不支持的配置。*/
    ctx->enigma_key_file = (ctx->enigma_key_file?ctx->enigma_key_file:"");
    ctx->enigma_salt_size = ABCDK_CLAMP(ctx->enigma_salt_size, 0, 256);
    
    ctx->uplink = abcdk_option_get(ctx->args, "--uplink", 0, NULL);

    ctx->ipv4_pool = abcdk_ipool_create2(ipv4_pool_begin_p,ipv4_pool_end_p);
    ctx->ipv6_pool = abcdk_ipool_create2(ipv6_pool_begin_p,ipv6_pool_end_p);
    if(!ctx->ipv4_pool || !ctx->ipv6_pool)
        goto END;

    ctx->max_client = ABCDK_MAX(abcdk_ipool_count(ctx->ipv4_pool,0),abcdk_ipool_count(ctx->ipv6_pool,0));
    if(!ctx->max_client)
        goto END;

    ctx->srpc_ctx = abcdk_srpc_create(1000, -1);
    if (!ctx->srpc_ctx)
        goto END;

    chk = _abcdkproxy_server_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_RAW);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_server_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_PKI);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_server_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_ENIGMA);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_server_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA);
    if (chk != 0)
        goto END;
    
    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

END:

    abcdk_srpc_destroy(&ctx->srpc_ctx);
    abcdk_srpc_unref(&ctx->listen_raw_p);
    abcdk_srpc_unref(&ctx->listen_pki_p);
    abcdk_srpc_unref(&ctx->listen_enigma_p);
    abcdk_srpc_unref(&ctx->listen_pki_enigma_p);
    abcdk_ipool_destroy(&ctx->ipv4_pool);
    abcdk_ipool_destroy(&ctx->ipv6_pool);
}

static int _abcdkproxy_client_connect_uplink(abcdkvnet_t *ctx)
{
    abcdk_sockaddr_t uplink_addr = {0};
    abcdk_srpc_config_t srpc_cfg = {0};
    int chk;

    ctx->uplink_p = abcdk_srpc_alloc(ctx->srpc_ctx);
    if(!ctx->uplink_p)
        return -1;

    chk = abcdk_srpc_connect(ctx->uplink_p,&uplink_addr,&srpc_cfg);
    if(chk != 0)
        return -2;

    return 0;
}

static int _abcdkproxy_client_start_dhcp(abcdkvnet_t *ctx)
{
    abcdk_object_t *rsp = NULL;
    char sbuf[100] = {0};
    abcdk_bit_t sbit = {0,sbuf,100},rbit = {0};
    uint16_t err;
    int chk;
    
    abcdk_bit_write_number(&sbit,16,ABCDKVNET_CMD_ASK_IP);

    chk = abcdk_srpc_request(ctx->uplink_p,sbuf,2,&rsp);
    if(chk != 0)
        return -1;

    rbit.data = rsp->pptrs[0];
    rbit.size = rsp->sizes[0];

    abcdk_bit_seek(&rbit,16);
    err = abcdk_bit_read2number(&rbit,16);
    if(err != 0)
        return -2;

    abcdk_bit_read2buffer(&rbit,(uint8_t*)&ctx->virtual_addr4.addr4.sin_addr.s_addr,4);
    abcdk_bit_read2buffer(&rbit,ctx->virtual_addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);

    return 0;
}

static void _abcdkvnet_process_client(abcdkvnet_t *ctx)
{
    const char *uplink_p = NULL;
    int chk;
     
    uplink_p = abcdk_option_get(ctx->args, "--uplink", 0, NULL);

    ctx->uplink_addr = abcdk_url_split(uplink_p);
    if(!ctx->uplink_addr)
        goto END;
    
    ctx->srpc_ctx = abcdk_srpc_create(10,-1);
    if(!ctx->srpc_ctx)
        goto END;


END:


    abcdk_srpc_destroy(&ctx->srpc_ctx);
    abcdk_object_unref(&ctx->uplink_addr);
    
}

static void _abcdkvnet_process(abcdkvnet_t *ctx)
{
    const char *log_path;

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "proxy.log", "proxy.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

    ctx->role = abcdk_option_get_int(ctx->args,"--role",0,ABCDKVNET_ROLE_CLIENT);

    if(ctx->role == ABCDKVNET_ROLE_CLIENT)
        _abcdkvnet_process_client(ctx);
    else if(ctx->role == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_process_server(ctx);

 

    abcdk_trace_output(LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}

static int _abcdkvnet_daemon_process_cb(void *opaque)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;

    _abcdkvnet_process(ctx);

    return 0;
}

static void _abcdkvnet_daemon(abcdkvnet_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path, "vnet-daemon.log", "vnet-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, logger);

    abcdk_proc_daemon(interval, _abcdkvnet_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_vnet(abcdk_option_t *args)
{
    abcdkvnet_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkvnet_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdkvnet_daemon(&ctx);
        }
        else
        {
            _abcdkvnet_process(&ctx);
        }
    }

    return 0;
}

