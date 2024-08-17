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

    /*虚拟IP地址类型(静态)。*/
    ABCDKVNET_IPADDR_TYPE_STATIC = 1,
#define ABCDKVNET_IP_TYPE_STATIC ABCDKVNET_IP_TYPE_STATIC

    /*虚拟IP地址类型(动态)。*/
    ABCDKVNET_IPADDR_TYPE_DHCP = 2,
#define ABCDKVNET_IP_TYPE_DHCP ABCDKVNET_IP_TYPE_DHCP

    /**请求IP地址.*/
    ABCDKVNET_CMD_REQUEST_IP = 1,
#define ABCDKVNET_CMD_REQUEST_IP ABCDKVNET_CMD_REQUEST_IP
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

    /*退出标志。*/
    volatile int exit_flag;

    /*日志。*/
    abcdk_logger_t *logger;

   
    /*虚拟IP池。*/
    abcdk_ipool_t *virtual_ipv4_pool;
    abcdk_ipool_t *virtual_ipv6_pool;

    /*虚拟路由表。*/
    abcdk_object_t *virtual_route_list;
    abcdk_mutex_t *virtual_route_mutex;
    
    /*虚拟地址。*/
    abcdk_sockaddr_t virtual_addr4;
    abcdk_sockaddr_t virtual_addr6;

    /*RPC.*/
    abcdk_srpc_t *rpc_ctx;

    /*原始监听对象。*/
    abcdk_srpc_session_t *rpc_listen_raw_session;

    /*PKI监听对象。*/
    abcdk_srpc_session_t *rpc_listen_pki_session;

    /*ENIGMA监听对象。*/
    abcdk_srpc_session_t *rpc_listen_enigma_session;

    /*PKIonENIGMA监听对象。*/
    abcdk_srpc_session_t *rpc_listen_pki_enigma_session;

    /*上行对象。*/
    abcdk_srpc_session_t *rpc_uplink_session;

    /*上行地址。*/
    abcdk_sockaddr_t rpc_uplink_addr;

    /*TUN设备名称。*/
    char tun_name[NAME_MAX];

    /*TUN设备句柄。*/
    int tun_fd;
    
    /*角色。*/
    int role;

    /*虚拟IPV4地址类型。*/
    int virtual_addr4_type;

    /*虚拟IPV4地址类型。*/
    int virtual_addr6_type;

    /*静态虚拟地址。*/
    const char *virtual_static_addr4;
    const char *virtual_static_addr6;

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

    /*上行安全方案。*/
    int uplink_ssl_scheme;

    /*上行地址。*/
    const char *uplink_addr;


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

    fprintf(stderr, "\n\t--virtual-addr4-type < TYPE >\n");
    fprintf(stderr, "\t\t虚拟IPV4地址类型。默认：%d\n", ABCDKVNET_IPADDR_TYPE_DHCP);

    fprintf(stderr, "\n\t--virtual-addr6-type < TYPE >\n");
    fprintf(stderr, "\t\t虚拟IPV6地址类型。默认：%d\n", ABCDKVNET_IPADDR_TYPE_DHCP);

    fprintf(stderr, "\n\t\t%d：静态\n",ABCDKVNET_IPADDR_TYPE_STATIC);
    fprintf(stderr, "\t\t%d：动态\n",ABCDKVNET_IPADDR_TYPE_DHCP);

    fprintf(stderr, "\n\t--virtual-static-addr4 < ADDR >\n");
    fprintf(stderr, "\t\t虚拟IPV4地址(静态)。\n");

    fprintf(stderr, "\n\t--virtual-static-addr6 < ADDR >\n");
    fprintf(stderr, "\t\t虚拟IPV6地址(静态)。\n");

    fprintf(stderr, "\n\t--role < TYPE >\n");
    fprintf(stderr, "\t\t角色。默认：%d\n",ABCDKVNET_ROLE_CLIENT);

    fprintf(stderr, "\n\t\t%d：服务端\n",ABCDKVNET_ROLE_SERVER);
    fprintf(stderr, "\t\t%d：客户端\n",ABCDKVNET_ROLE_CLIENT);

    fprintf(stderr, "\n\t--ipv4-pool-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPV4地址池开始。默认：ipv4://10.0.0.1\n");

    fprintf(stderr, "\n\t--ipv4-pool-end < ADDR >\n");
    fprintf(stderr, "\t\tIPV4地址池结束。默认：ipv4://10.0.0.255\n");
    
    fprintf(stderr, "\n\t--ipv4-pool-dhcp-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPV4地址池DHCP范围开始。默认：ipv4://10.0.0.10\n");

    fprintf(stderr, "\n\t--ipv4-pool-dhcp-end < ADDR >\n");
    fprintf(stderr, "\t\tIPV4地址池DHCP范围结束。默认：ipv4://10.0.0.255\n");

    fprintf(stderr, "\n\t--ipv6-pool-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPV6地址池开始。默认：ipv6://[fc00::1]\n");

    fprintf(stderr, "\n\t--ipv6-pool-end < ADDR >\n");
    fprintf(stderr, "\t\tIPV6地址池结束。默认：ipv6://[fc00::ff]\n");

    fprintf(stderr, "\n\t--ipv6-pool-dhcp-begin < ADDR >\n");
    fprintf(stderr, "\t\tIPV6地址池DHCP范围开始。默认：ipv6://[fc00::a]\n");

    fprintf(stderr, "\n\t--ipv6-pool-dhcp-end < ADDR >\n");
    fprintf(stderr, "\t\tIPV6地址池DHCP范围结束。默认：ipv6://[fc00::ff]\n");

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

    fprintf(stderr, "\n\t--uplink-ssl-scheme < SCHEME >\n");
    fprintf(stderr, "\t\t上行安全方案。默认：%d\n",ABCDK_ASIO_SSL_SCHEME_RAW);

    fprintf(stderr, "\n\t\t%d：RAW\n",ABCDK_ASIO_SSL_SCHEME_RAW);
    fprintf(stderr, "\t\t%d：PKI\n",ABCDK_ASIO_SSL_SCHEME_PKI);
    fprintf(stderr, "\t\t%d：ENIGMA\n",ABCDK_ASIO_SSL_SCHEME_ENIGMA);
    fprintf(stderr, "\t\t%d：PKIonENIGMA\n",ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA);

    fprintf(stderr, "\n\t--uplink-addr < URL >\n");
    fprintf(stderr, "\t\t上行地址。\n");

    fprintf(stderr, "\n\t\tipv4://DOMAIN:PORT\n");
    fprintf(stderr, "\t\tipv6://[DOMAIN]:PORT\n");
}

static int _abcdkvnet_open_tun(const char *name)
{
    struct ifreq ifr;
    int fd;
    int chk;

#define AVBCDKVNET_TUN_DEV "/dev/net/tun"

    fd = open(AVBCDKVNET_TUN_DEV, O_RDWR);
    if (fd < 0)
    {
        abcdk_trace_output(LOG_ERR, "打开'%s'失败，权限不足。", AVBCDKVNET_TUN_DEV);
        return -1;
    }

    memset(&ifr, 0, sizeof(ifr));

    ifr.ifr_flags = IFF_TUN | IFF_NO_PI;
    strncpy(ifr.ifr_name, name, IFNAMSIZ);

    chk = ioctl(fd, TUNSETIFF, (void *)&ifr);
    if (chk >= 0)
        return fd;

    abcdk_trace_output(LOG_ERR, "创建TUN(%s)失败，权限不足或已存在。", name);
    close(fd);

    return -1;
}

static int _abcdkvnet_ifconfig(abcdkvnet_t *ctx)
{
    char ipv4str[100] = {0},ipv6str[100] = {0};
    int chk;

    abcdk_sockaddr_to_string(ipv4str, &ctx->virtual_addr4);
    abcdk_sockaddr_to_string(ipv6str, &ctx->virtual_addr6);

    chk = abcdk_net_address_add(AF_INET,ipv4str,32,NULL,0,ctx->tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN设备(%s)添加IPV4地址(%s)失败，权限不足或系统错。", ctx->tun_name,ipv4str);
        return -1;
    }

    /*服务端仅需要配置IP地址。*/
    if(ctx->role == ABCDKVNET_ROLE_SERVER)
        return 0;

    chk = abcdk_net_address_add(AF_INET6,ipv6str,128,NULL,0,ctx->tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN设备(%s)添加IPV6地址(%s)失败，权限不足或系统错。", ctx->tun_name,ipv6str);
        return -1;
    }

    chk = abcdk_net_route_add(AF_INET, "0.0.0.0", 0, ipv4str, 0, ctx->tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN设备(%s)添加IPV6地址(%s)失败，权限不足或系统错。", ctx->tun_name,ipv6str);
        return -1;
    }

    abcdk_net_route_add(AF_INET6, "0", 0, ipv6str, 0, ctx->tun_name);

    return 0;
}

static int _abcdkvnet_ipool_allocate(abcdkvnet_t *ctx, int type, abcdk_sockaddr_t *addr)
{
    int chk = -1;
    /*分配IP地址。*/
    if (type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        if (addr->family == AF_INET)
            chk = abcdk_ipool_static_request(ctx->virtual_ipv4_pool, addr);
        else if (addr->family == AF_INET6)
            chk = abcdk_ipool_static_request(ctx->virtual_ipv6_pool, addr);
    }
    else if (type == ABCDKVNET_IPADDR_TYPE_DHCP)
    {
        if (addr->family == AF_INET)
            chk = abcdk_ipool_dhcp_request(ctx->virtual_ipv4_pool, addr);
        else if (addr->family == AF_INET6)
            chk = abcdk_ipool_dhcp_request(ctx->virtual_ipv6_pool, addr);
    }

    return chk;
}

static void _abcdkvnet_ipool_reclaim(abcdkvnet_t *ctx, abcdk_sockaddr_t *addr)
{
    if (addr->family == AF_INET)
        abcdk_ipool_reclaim(ctx->virtual_ipv4_pool, addr);
    else if (addr->family == AF_INET6)
        abcdk_ipool_reclaim(ctx->virtual_ipv6_pool, addr);
}

static int _abcdkvnet_server_ip_allocate(abcdkvnet_t *ctx, abcdk_srpc_session_t *session,int type4, abcdk_sockaddr_t *addr4,int type6, abcdk_sockaddr_t *addr6)
{
    abcdkvnet_node_t *node_p = NULL;
    int chk = -1;

    if(!session)
        return -1;
    
    /*分配IPV4地址。*/
    chk = _abcdkvnet_ipool_allocate(ctx,type4,addr4);
    if(chk != 0)
        return -11;

    /*分配IPV6地址。*/
    chk = _abcdkvnet_ipool_allocate(ctx,type6,addr6);
    if(chk != 0)
        return -11;
    
    abcdk_mutex_lock(ctx->virtual_route_mutex, 1);

    for (int i = 0; i < ctx->virtual_route_list->numbers; i++)
    {
        node_p = (abcdkvnet_node_t *)ctx->virtual_route_list->pptrs[i];
        if (!node_p->pipe)
            break;

        node_p = NULL;
    }

    if(node_p)
    {
        /*引用链路。*/
        node_p->pipe = abcdk_srpc_refer(session);

        /*绑定IP地址。*/
        node_p->addr4 = *addr4;
        node_p->addr6 = *addr6;

        chk = 0;
    }
    else
    {
        /*无空闲位置时，回收动态分配的IP地址。*/

        if (type4 == ABCDKVNET_IPADDR_TYPE_DHCP)
            _abcdkvnet_ipool_reclaim(ctx,addr4);

        if (type6 == ABCDKVNET_IPADDR_TYPE_DHCP)
            _abcdkvnet_ipool_reclaim(ctx,addr6);
        
        chk = -1;
    }

    abcdk_mutex_unlock(ctx->virtual_route_mutex);

    return chk;
}

static void _abcdkvnet_server_ip_reclaim(abcdkvnet_t *ctx, abcdk_srpc_session_t *session)
{
    abcdkvnet_node_t *node_p = NULL;

    if(!session)
        return;

    abcdk_mutex_lock(ctx->virtual_route_mutex, 1);

    for (int i = 0; i < ctx->virtual_route_list->numbers; i++)
    {
        node_p = (abcdkvnet_node_t *)ctx->virtual_route_list->pptrs[i];
        if(node_p->pipe == session)
            break;

        node_p = NULL;
    }
    
    if(node_p)
    {
        abcdk_srpc_unref(&node_p->pipe);
        _abcdkvnet_ipool_reclaim(ctx,&node_p->addr4);
        _abcdkvnet_ipool_reclaim(ctx,&node_p->addr6);

        memset(node_p,0,sizeof(abcdkvnet_node_t));
    }

    abcdk_mutex_unlock(ctx->virtual_route_mutex);
}

static int _abcdkvnet_server_cmd_request_ip(abcdkvnet_t *ctx,abcdk_srpc_session_t *session,abcdk_bit_t *req,abcdk_object_t **rsp)
{
    abcdk_object_t *rsp_p = NULL;
    abcdk_bit_t rspbit = {0};
    int type4,type6;
    abcdk_sockaddr_t addr4 = {AF_INET},addr6 = {AF_INET6};
    int chk;

    type4 = abcdk_bit_read2number(req,8);
    abcdk_bit_read2buffer(req,(uint8_t*)&addr4.addr4.sin_addr.s_addr,4);

    type6 = abcdk_bit_read2number(req,8);
    abcdk_bit_read2buffer(req,addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);
    
    /*为了防止客户端重复申请从而造成的IP地址丢失，因此要先回收之前注册的地址。*/
    _abcdkvnet_server_ip_reclaim(ctx,session);
    
    /*申请IP地址。*/
    chk = _abcdkvnet_server_ip_allocate(ctx,session,type4,&addr4,type6,&addr6);

    rsp_p = abcdk_object_alloc2(100);
    if(!rsp_p)
        return -1;

    rspbit.data = rsp_p->pptrs[0];
    rspbit.size = rsp_p->sizes[0];

    abcdk_bit_write_number(&rspbit,16,ABCDKVNET_CMD_REQUEST_IP);
    
    if(chk != 0)
    {
        abcdk_bit_write_number(&rspbit,32,11);
        abcdk_bit_write_number(&rspbit,32,0);
        abcdk_bit_write_number(&rspbit,64,0);
        abcdk_bit_write_number(&rspbit,64,0);
    }
    else 
    {
        abcdk_bit_write_number(&rspbit,32,0);
        abcdk_bit_write_buffer(&rspbit,(uint8_t*)&addr4.addr4.sin_addr.s_addr,4);
        abcdk_bit_write_buffer(&rspbit,addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);
    }

    /*有效长度*/
    rsp_p->sizes[0] = rspbit.pos /8;

    *rsp = rsp_p;

    return 0;
}

static int _abcdkvnet_server_offline_client(abcdkvnet_t *ctx,abcdk_srpc_session_t *session)
{
    _abcdkvnet_server_ip_reclaim(ctx,session);
}

static int _abcdkvnet_server_cmd_process(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, const void *data, size_t size,abcdk_object_t **rsp)
{
    abcdk_bit_t req = {0};
    uint16_t cmd;
    int chk;

    req.data = (void*)data;
    req.size = size;

    cmd = abcdk_bit_read2number(&req,16);

    if(cmd == ABCDKVNET_CMD_REQUEST_IP)
        chk = _abcdkvnet_server_cmd_request_ip(ctx,session,&req,rsp);

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

    session_p = abcdk_srpc_alloc(ctx->rpc_ctx);
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

    if(ctx->role == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_server_offline_client(ctx,session);
}

static int _abcdkproxy_server_start_listen(abcdkvnet_t *ctx, int ssl_scheme)
{
    const char *listen_p = NULL;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_srpc_session_t *session_p = NULL;
    abcdk_srpc_config_t rpc_cfg = {0};
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
        session_p = ctx->rpc_listen_raw_session = abcdk_srpc_alloc(ctx->rpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        session_p = ctx->rpc_listen_pki_session = abcdk_srpc_alloc(ctx->rpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        session_p = ctx->rpc_listen_enigma_session = abcdk_srpc_alloc(ctx->rpc_ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        session_p = ctx->rpc_listen_pki_enigma_session = abcdk_srpc_alloc(ctx->rpc_ctx);

    if (!session_p)
        return -2;

    rpc_cfg.opaque = ctx;
    rpc_cfg.ssl_scheme = ssl_scheme;
    rpc_cfg.pki_ca_file = ctx->pki_ca_file;
    rpc_cfg.pki_ca_path = ctx->pki_ca_path;
    rpc_cfg.pki_cert_file = ctx->pki_cert_file;
    rpc_cfg.pki_key_file = ctx->pki_key_file;
    rpc_cfg.pki_check_cert = ctx->pki_check_cert;
    rpc_cfg.enigma_key_file = ctx->enigma_key_file;
    rpc_cfg.enigma_salt_size = ctx->enigma_salt_size;

    rpc_cfg.prepare_cb = _abcdkvnet_srpc_prepare_cb;
    rpc_cfg.request_cb = _abcdkvnet_srpc_request_cb;
    rpc_cfg.close_cb = _abcdkvnet_srpc_close_cb;

    chk = abcdk_srpc_listen(session_p, &listen_addr, &rpc_cfg);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'失败，无权限或被占用。", listen_p);
        return -3;
    }

    return 0;
}

static void _abcdkvnet_server_dowork(abcdkvnet_t *ctx)
{

LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        return;

    sleep(1);

    goto LOOP;
}

static void _abcdkvnet_process_server(abcdkvnet_t *ctx)
{
    const char *virtual_addr4_p = NULL;
    const char *virtual_addr6_p = NULL;
    int max_client;
    int chk;

    ctx->virtual_addr4_type = abcdk_option_get_int(ctx->args, "--virtual-addr4-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_addr6_type = abcdk_option_get_int(ctx->args, "--virtual-addr6-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_static_addr4 = abcdk_option_get(ctx->args, "--virtual-static-addr4", 0, NULL);
    ctx->virtual_static_addr6 = abcdk_option_get(ctx->args, "--virtual-static-addr6", 0, NULL);

    const char *ipv4_pool_begin_p = abcdk_option_get(ctx->args,"--ipv4-pool-begin",0,"ipv4://10.0.0.1");
    const char *ipv4_pool_end_p = abcdk_option_get(ctx->args,"--ipv4-pool-end",0,"ipv4://10.0.0.255");
    const char *ipv6_pool_begin_p = abcdk_option_get(ctx->args,"--ipv6-pool-begin",0,"ipv6://[fc00::1]");
    const char *ipv6_pool_end_p = abcdk_option_get(ctx->args,"--ipv6-pool-end",0,"ipv6://[fc00::ff]");
    const char *ipv4_pool_dhcp_begin_p = abcdk_option_get(ctx->args,"--ipv4-pool-dhcp-begin",0,"ipv4://10.0.0.10");
    const char *ipv4_pool_dhcp_end_p = abcdk_option_get(ctx->args,"--ipv4-pool-dhcp-end",0,"ipv4://10.0.0.255");
    const char *ipv6_pool_dhcp_begin_p = abcdk_option_get(ctx->args,"--ipv6-pool-dhcp-begin",0,"ipv6://[fc00::A]");
    const char *ipv6_pool_dhcp_end_p = abcdk_option_get(ctx->args,"--ipv6-pool-dhcp-end",0,"ipv6://[fc00::ff]");

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
    
    if (ctx->virtual_addr4_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk = abcdk_sockaddr_from_string(&ctx->virtual_addr4, ctx->virtual_static_addr4, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_WARNING, "虚拟IPV4静态的地址(%s)解析错误。", ctx->virtual_static_addr4);
            goto END;
        }
    }

    if (ctx->virtual_addr6_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk = abcdk_sockaddr_from_string(&ctx->virtual_addr6, ctx->virtual_static_addr6, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_WARNING, "虚拟IPV6静态的地址(%s)解析错误。", ctx->virtual_static_addr6);
            goto END;
        }
    }

    ctx->virtual_ipv4_pool = abcdk_ipool_create2(ipv4_pool_begin_p,ipv4_pool_end_p);
    if(!ctx->virtual_ipv4_pool)
        abcdk_trace_output(LOG_WARNING,"虚拟IPV4地址池(%s,%s)无效。",ipv4_pool_begin_p,ipv4_pool_end_p);

    ctx->virtual_ipv6_pool = abcdk_ipool_create2(ipv6_pool_begin_p,ipv6_pool_end_p);
    if(!ctx->virtual_ipv6_pool)
        abcdk_trace_output(LOG_WARNING,"虚拟IPV6地址池(%s,%s)无效。",ipv6_pool_begin_p,ipv6_pool_end_p);

    if(!ctx->virtual_ipv4_pool || !ctx->virtual_ipv6_pool)
        goto END;
    
    if(ipv4_pool_dhcp_begin_p && ipv4_pool_dhcp_end_p)
    {
        chk = abcdk_ipool_set_dhcp_range2(ctx->virtual_ipv4_pool,ipv4_pool_dhcp_begin_p,ipv4_pool_dhcp_end_p);
        if(chk != 0)
        {
            abcdk_trace_output(LOG_WARNING,"虚拟IPV4地址池(%s,%s)设置动态范围(%s,%s)错误。",ipv4_pool_begin_p,ipv4_pool_end_p,ipv4_pool_dhcp_begin_p,ipv4_pool_dhcp_end_p);
            goto END;
        }
    }
    
    if(ipv6_pool_dhcp_begin_p && ipv6_pool_dhcp_end_p)
    {
        chk = abcdk_ipool_set_dhcp_range2(ctx->virtual_ipv6_pool,ipv6_pool_dhcp_begin_p,ipv6_pool_dhcp_end_p);
        if(chk != 0)
        {
            abcdk_trace_output(LOG_WARNING,"虚拟IPV6地址池(%s,%s)设置动态范围(%s,%s)错误。",ipv6_pool_begin_p,ipv6_pool_end_p,ipv6_pool_dhcp_begin_p,ipv6_pool_dhcp_end_p);
            goto END;
        }
    }

    /*请求IPV4地址。*/
    ctx->virtual_addr4.family = AF_INET;
    chk = _abcdkvnet_ipool_allocate(ctx, ctx->virtual_addr4_type, &ctx->virtual_addr4);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "请求虚拟IPV4地址失败，超出IPV4地址池(%s,%s)有效范围。",ipv4_pool_begin_p,ipv4_pool_end_p);
        goto END;
    }

    /*请求IPV6地址。*/
    ctx->virtual_addr6.family = AF_INET6;
    chk = _abcdkvnet_ipool_allocate(ctx, ctx->virtual_addr6_type, &ctx->virtual_addr6);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "请求虚拟IPV6地址失败，超出IPV4地址池(%s,%s)有效范围。",ipv6_pool_begin_p,ipv6_pool_end_p);
        goto END;
    }

    max_client = ABCDK_MAX(abcdk_ipool_count(ctx->virtual_ipv4_pool,0),abcdk_ipool_count(ctx->virtual_ipv6_pool,0));
    if(max_client <= 0)
        goto END;

    ctx->virtual_route_list = abcdk_object_alloc3(sizeof(abcdkvnet_node_t),max_client);
    if(!ctx->virtual_route_list)
        goto END;

    ctx->virtual_route_mutex = abcdk_mutex_create();
    if(!ctx->virtual_route_list)
        goto END;

    ctx->rpc_ctx = abcdk_srpc_create(max_client+4, -1);
    if (!ctx->rpc_ctx)
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
 
    _abcdkvnet_server_dowork(ctx);

END:

    abcdk_srpc_destroy(&ctx->rpc_ctx);
    abcdk_srpc_unref(&ctx->rpc_listen_raw_session);
    abcdk_srpc_unref(&ctx->rpc_listen_pki_session);
    abcdk_srpc_unref(&ctx->rpc_listen_enigma_session);
    abcdk_srpc_unref(&ctx->rpc_listen_pki_enigma_session);
    abcdk_ipool_destroy(&ctx->virtual_ipv4_pool);
    abcdk_ipool_destroy(&ctx->virtual_ipv6_pool);
    abcdk_object_unref(&ctx->virtual_route_list);
    abcdk_mutex_destroy(&ctx->virtual_route_mutex);
}

static int _abcdkproxy_client_connect_uplink(abcdkvnet_t *ctx)
{
    int ssl_scheme;
    abcdk_sockaddr_t uplink_addr = {0};
    abcdk_srpc_config_t rpc_cfg = {0};
    int chk;

    chk = abcdk_sockaddr_from_string(&uplink_addr, ctx->uplink_addr, 1);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "上行地址'%s'无法识别。", ctx->uplink_addr);
        return -1;
    }

    ctx->rpc_uplink_session = abcdk_srpc_alloc(ctx->rpc_ctx);
    if(!ctx->rpc_uplink_session)
        return -1;

    rpc_cfg.opaque = ctx;
    rpc_cfg.ssl_scheme = ctx->uplink_ssl_scheme;
    rpc_cfg.pki_ca_file = ctx->pki_ca_file;
    rpc_cfg.pki_ca_path = ctx->pki_ca_path;
    rpc_cfg.pki_cert_file = ctx->pki_cert_file;
    rpc_cfg.pki_key_file = ctx->pki_key_file;
    rpc_cfg.pki_check_cert = ctx->pki_check_cert;
    rpc_cfg.enigma_key_file = ctx->enigma_key_file;
    rpc_cfg.enigma_salt_size = ctx->enigma_salt_size;

    rpc_cfg.prepare_cb = _abcdkvnet_srpc_prepare_cb;
    rpc_cfg.request_cb = _abcdkvnet_srpc_request_cb;
    rpc_cfg.close_cb = _abcdkvnet_srpc_close_cb;

    chk = abcdk_srpc_connect(ctx->rpc_uplink_session,&uplink_addr,&rpc_cfg);
    if(chk != 0)
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session,LOG_ERR, "连接上级(%s)失败。", ctx->uplink_addr);
        return -2;
    }

    return 0;
}

static int _abcdkproxy_client_request_ip(abcdkvnet_t *ctx)
{
    char reqbuf[100] = {0};
    abcdk_bit_t reqbit = {0,reqbuf,100},rspbit = {0};
    abcdk_object_t *rsp = NULL;
    char ipv4str[100] = {0},ipv6str[100] = {0};
    int err;
    int chk;
    
    abcdk_bit_write_number(&reqbit,16,ABCDKVNET_CMD_REQUEST_IP);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_addr4_type);
    abcdk_bit_write_buffer(&reqbit, (uint8_t *)&ctx->virtual_addr4.addr4.sin_addr.s_addr, 4);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_addr6_type);
    abcdk_bit_write_buffer(&reqbit, ctx->virtual_addr6.addr6.sin6_addr.__in6_u.__u6_addr8, 16);
         
    chk = abcdk_srpc_request(ctx->rpc_uplink_session,reqbit.data,reqbit.pos/8,&rsp);
    if(chk != 0)
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session,LOG_ERR, "向上级(%s)请求IP地址失败，网络不通或目标不可达。", ctx->uplink_addr);
        return -1;
    }

    rspbit.data = rsp->pptrs[0];
    rspbit.size = rsp->sizes[0];

    abcdk_bit_seek(&rspbit,16);
    err = abcdk_bit_read2number(&rspbit,32);
    if (err == 0)
    {
        ctx->virtual_addr4.family = AF_INET;
        abcdk_bit_read2buffer(&rspbit, (uint8_t *)&ctx->virtual_addr4.addr4.sin_addr.s_addr, 4);

        ctx->virtual_addr6.family = AF_INET6;
        abcdk_bit_read2buffer(&rspbit, ctx->virtual_addr6.addr6.sin6_addr.__in6_u.__u6_addr8, 16);

        abcdk_sockaddr_to_string(ipv4str, &ctx->virtual_addr4);
        abcdk_sockaddr_to_string(ipv6str, &ctx->virtual_addr6);

        abcdk_srpc_trace_output(ctx->rpc_uplink_session, LOG_INFO, "向上级(%s)请求IP成功，注册的虚拟地址是IPV4(%s)和IPV6(%s)",
                                ctx->uplink_addr, ipv4str, ipv6str);

        chk = 0;
    }
    else 
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session, LOG_ERR, "向上级(%s)请求IP地址失败(ERRNO=%d)。",ctx->uplink_addr, err);

        chk = -1;
    }

    abcdk_object_unref(&rsp);

    return chk;
}

static void _abcdkvnet_client_dowork(abcdkvnet_t *ctx)
{
    int chk;

LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        return;

    chk = _abcdkproxy_client_connect_uplink(ctx);
    if(chk != 0)
        goto ERR;

    chk = _abcdkproxy_client_request_ip(ctx);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR,"向上级(%s)请求IP址地失败，准备重试。",ctx->uplink_addr);
        goto ERR;
    }

ERR:

    /*快速关闭。*/
    if(ctx->rpc_uplink_session)
        abcdk_srpc_set_timeout(ctx->rpc_uplink_session,1);
        
    abcdk_srpc_unref(&ctx->rpc_uplink_session);

    sleep(3);
    goto LOOP;
}


static void _abcdkvnet_process_client(abcdkvnet_t *ctx)
{
    int chk4,chk6;
    int chk;

    ctx->virtual_addr4_type = abcdk_option_get_int(ctx->args, "--virtual-addr4-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_addr6_type = abcdk_option_get_int(ctx->args, "--virtual-addr6-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_static_addr4 = abcdk_option_get(ctx->args, "--virtual-static-addr4", 0, "");
    ctx->virtual_static_addr6 = abcdk_option_get(ctx->args, "--virtual-static-addr6", 0, "");

    ctx->pki_ca_file = abcdk_option_get(ctx->args, "--pki-ca-file", 0, NULL);
    ctx->pki_ca_path = abcdk_option_get(ctx->args, "--pki-ca-path", 0, NULL);
    ctx->pki_cert_file = abcdk_option_get(ctx->args, "--pki-cert-file", 0, NULL);
    ctx->pki_key_file = abcdk_option_get(ctx->args, "--pki-key-file", 0, NULL);
    ctx->pki_check_cert = abcdk_option_get_int(ctx->args, "--pki-check-cert", 0, 1);

    ctx->enigma_key_file = abcdk_option_get(ctx->args, "--enigma-key-file", 0, NULL);
    ctx->enigma_salt_size = abcdk_option_get_int(ctx->args, "--enigma-salt-size", 0, 123);
     
    ctx->uplink_ssl_scheme = abcdk_option_get_int(ctx->args, "--uplink-ssl-scheme", 0, ABCDK_ASIO_SSL_SCHEME_RAW);
    ctx->uplink_addr = abcdk_option_get(ctx->args, "--uplink-addr", 0, "");

    if (ctx->virtual_addr4_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk4 = abcdk_sockaddr_from_string(&ctx->virtual_addr4, ctx->virtual_static_addr4, 0);
        if (chk4 != 0)
        {
            abcdk_trace_output(LOG_WARNING, "静态IPV4地址(%s)解析错误。", ctx->virtual_static_addr4);
            goto END;
        }
    }

    if (ctx->virtual_addr6_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk6 = abcdk_sockaddr_from_string(&ctx->virtual_addr6, ctx->virtual_static_addr6, 0);
        if (chk6 != 0)
        {
            abcdk_trace_output(LOG_WARNING, "静态IPV6地址(%s)解析错误。", ctx->virtual_static_addr6);
            goto END;
        }
    }

    ctx->rpc_ctx = abcdk_srpc_create(10,-1);
    if(!ctx->rpc_ctx)
        goto END;

    _abcdkvnet_client_dowork(ctx);

END:


    abcdk_srpc_destroy(&ctx->rpc_ctx);
}

static void *_abcdkvnet_signal_process(void *opaque)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
   
    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

    abcdk_atomic_store(&ctx->exit_flag,1);
}

static void _abcdkvnet_process(abcdkvnet_t *ctx)
{
    const char *log_path = NULL;
    abcdk_thread_t signal_thread = {0};

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "vnet.log", "vnet.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

    signal_thread.opaque = ctx;
    signal_thread.routine = _abcdkvnet_signal_process;

    /*创建信号线程。*/
    abcdk_thread_create(&signal_thread,1);

    ctx->role = abcdk_option_get_int(ctx->args,"--role",0,ABCDKVNET_ROLE_CLIENT);

    if(ctx->role == ABCDKVNET_ROLE_CLIENT)
        _abcdkvnet_process_client(ctx);
    else if(ctx->role == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_process_server(ctx);

    /*等待信号线程退出。*/
    abcdk_thread_join(&signal_thread);

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

