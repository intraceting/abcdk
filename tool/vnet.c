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
    /**监听者。*/
    ABCDKVNET_ROLE_LISTENER = 0,
#define ABCDKVNET_ROLE_LISTENER ABCDKVNET_ROLE_LISTENER

    /**服务端。*/
    ABCDKVNET_ROLE_SERVER = 1,
#define ABCDKVNET_ROLE_SERVER ABCDKVNET_ROLE_SERVER

    /**客户端。*/
    ABCDKVNET_ROLE_CLIENT = 2,
#define ABCDKVNET_ROLE_CLIENT ABCDKVNET_ROLE_CLIENT

    /**虚拟IP地址类型(静态)。*/
    ABCDKVNET_IPADDR_TYPE_STATIC = 1,
#define ABCDKVNET_IPADDR_TYPE_STATIC ABCDKVNET_IPADDR_TYPE_STATIC

    /**虚拟IP地址类型(动态)。*/
    ABCDKVNET_IPADDR_TYPE_DHCP = 2,
#define ABCDKVNET_IPADDR_TYPE_DHCP ABCDKVNET_IPADDR_TYPE_DHCP

    /**登录注册。*/
    ABCDKVNET_CMD_LOGON = 1,
#define ABCDKVNET_CMD_LOGON ABCDKVNET_CMD_LOGON

    /**投递消息。*/
    ABCDKVNET_CMD_POSTING = 2,
#define ABCDKVNET_CMD_POSTING ABCDKVNET_CMD_POSTING
};

/*节点。*/
typedef struct _abcdkvnet_node
{
    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*地址。*/
    abcdk_sockaddr_t virtual_addr4;
    abcdk_sockaddr_t virtual_addr6;

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
   
    /*虚拟地址池。*/
    abcdk_ipool_t *virtual_ipv4_pool;
    abcdk_ipool_t *virtual_ipv6_pool;
    uint8_t virtual_ipv4_pool_prefix;
    uint8_t virtual_ipv6_pool_prefix;

    /*虚拟路由表。*/
    abcdk_iplan_t *virtual_route_list;
    abcdk_mutex_t *virtual_route_locker;

    /*虚拟地址。*/
    uint8_t virtual_mask_prefix4;
    uint8_t virtual_mask_prefix6;
    abcdk_sockaddr_t virtual_gateway_addr4;
    abcdk_sockaddr_t virtual_gateway_addr6;
    abcdk_sockaddr_t virtual_local_addr4;
    abcdk_sockaddr_t virtual_local_addr6;

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

    /*上行网关。*/
    abcdk_sockaddr_t rpc_uplink_gateway;

    /*虚拟TUN设备名称。*/
    char virtual_tun_name[NAME_MAX];

    /*虚拟TUN设备句柄。*/
    int virtual_tun_fd;

    
    /*虚拟IPV4地址类型。*/
    int virtual_addr4_type;

    /*虚拟IPV4地址类型。*/
    int virtual_addr6_type;

    /*静态虚拟地址。*/
    uint8_t virtual_static_prefix4;
    uint8_t virtual_static_prefix6;
    const char *virtual_static_addr4;
    const char *virtual_static_addr6;

    /*虚拟TUN设备前缀。*/
    const char *virtual_tun_prefix;

    /*虚拟TUN设备最大传输单元。*/
    int virtual_tun_mtu;

    /*设置默认的全局虚拟路由(客户端有效)。0 否，!0 是。*/
    int virtual_default_route;

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

    /*上行安全方案。*/
    int uplink_ssl_scheme;

    /*上行地址(服务端真实地址)。*/
    const char *uplink_addr;

    /*上行网关(网关真实地址)。*/
    const char *uplink_gateway;


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

    fprintf(stderr, "\n\t--virtual-static-prefix4 < BITS >\n");
    fprintf(stderr, "\t\t虚拟IPV4前缀(静态)。默认：24\n");

    fprintf(stderr, "\n\t--virtual-static-prefix6 < BITS >\n");
    fprintf(stderr, "\t\t虚拟IPV6前缀(静态)。默认：120\n");

    fprintf(stderr, "\n\t--virtual-static-addr4 < ADDR >\n");
    fprintf(stderr, "\t\t虚拟IPV4地址(静态)。\n");

    fprintf(stderr, "\n\t--virtual-static-addr6 < ADDR >\n");
    fprintf(stderr, "\t\t虚拟IPV6地址(静态)。\n");

    fprintf(stderr, "\n\t--virtual-tun-prefix < NAME >\n");
    fprintf(stderr, "\t\t虚拟TUN设备前缀。默认: vnet\n");

    fprintf(stderr, "\n\t--virtual-tun-mtu < SIZE >\n");
    fprintf(stderr, "\t\t虚拟TUN设备最大传输单元(1500~65535)。默认: 1500\n");

    fprintf(stderr, "\n\t--virtual-default-route < BOOL >\n");
    fprintf(stderr, "\t\t虚拟全局路由配置状态。默认: 0\n");

    fprintf(stderr, "\n\t\t0：禁用\n");
    fprintf(stderr, "\t\t1：启用\n");

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
    fprintf(stderr, "\t\t监的长度(0~255)。默认：123。\n");

    fprintf(stderr, "\n\t--uplink-ssl-scheme < SCHEME >\n");
    fprintf(stderr, "\t\t上行安全方案。默认：%d\n",ABCDK_ASIO_SSL_SCHEME_RAW);

    fprintf(stderr, "\n\t\t%d：RAW\n",ABCDK_ASIO_SSL_SCHEME_RAW);
    fprintf(stderr, "\t\t%d：PKI\n",ABCDK_ASIO_SSL_SCHEME_PKI);
    fprintf(stderr, "\t\t%d：ENIGMA\n",ABCDK_ASIO_SSL_SCHEME_ENIGMA);
    fprintf(stderr, "\t\t%d：PKIonENIGMA\n",ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA);

    fprintf(stderr, "\n\t--uplink-addr < ADDR >\n");
    fprintf(stderr, "\t\t上行地址。\n");

    fprintf(stderr, "\n\t\tipv4://IP:PORT\n");
    fprintf(stderr, "\t\tipv6://[IP]:PORT\n");

    fprintf(stderr, "\n\t--uplink-gateway < ADDR >\n");
    fprintf(stderr, "\t\t上行网关。\n");

    fprintf(stderr, "\n\t\tipv4://IP\n");
    fprintf(stderr, "\t\tipv6://[IP]\n");
}

static int _abcdkvnet_tun_open(const char *name)
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

static int _abcdkvnet_tun_poll(int fd,int event, time_t timeout)
{
    if(fd < 0)
        return -1;

    return abcdk_poll(fd,event,timeout);
}

static ssize_t _abcdkvnet_tun_read(int fd,void *buf,size_t size)
{
    if(fd < 0)
        return 0;

    return read(fd, buf, size);
}

static ssize_t _abcdkvnet_tun_write(int fd,const void *buf,size_t size)
{
    if(fd < 0)
        return 0;

    return write(fd, buf, size);
}

static void _abcdkvnet_node_free(abcdk_srpc_session_t **session)
{
    abcdk_srpc_unref(session);
}

static abcdk_srpc_session_t *_abcdkvnet_node_alloc(abcdkvnet_t *ctx,int flag)
{
    abcdk_srpc_session_t *session_p;
    abcdkvnet_node_t *node_ctx_p;

    session_p = abcdk_srpc_alloc(ctx->rpc_ctx,sizeof(abcdkvnet_node_t),NULL);
    if(!session_p)
        return NULL;

    node_ctx_p = (abcdkvnet_node_t *) abcdk_srpc_get_userdata(session_p);

    node_ctx_p->flag = flag;

    return session_p;

ERR:

    _abcdkvnet_node_free(&session_p);
    return NULL;
}

static void _abcdkvnet_uplink_route_del(abcdkvnet_t *ctx)
{
    char upaddr[100] = {0},upgw[100] = {0};

    if(ctx->rpc_uplink_addr.family != ctx->rpc_uplink_gateway.family)
        return;
    
    abcdk_sockaddr_to_string(upaddr,&ctx->rpc_uplink_addr);
    abcdk_sockaddr_to_string(upgw, &ctx->rpc_uplink_gateway);

    abcdk_proc_shell(NULL, NULL,"ip route del %s via %s",upaddr,upgw);
}

static int _abcdkvnet_uplink_route_add(abcdkvnet_t *ctx)
{
    char upaddr[100] = {0},upgw[100] = {0};
    int exitcode = 0;
    int chk;

    if(ctx->rpc_uplink_addr.family != ctx->rpc_uplink_gateway.family)
    {
        abcdk_trace_output(LOG_ERR, "上行地址(%s)和网关(%s)必须使用相同的协议。", upaddr,upgw);
        return -1;
    }

    abcdk_sockaddr_to_string(upaddr,&ctx->rpc_uplink_addr);
    abcdk_sockaddr_to_string(upgw, &ctx->rpc_uplink_gateway);

    chk = abcdk_proc_shell(&exitcode, NULL,"ip route add %s via %s",upaddr,upgw);
    if(chk != 0 || (exitcode != 0 && exitcode != 2))
    {
        abcdk_trace_output(LOG_ERR, "上行地址(%s)的网关(%s)配置路由失败，权限不足或系统错误。", upaddr,upgw);
        return -1;
    }
        
    return 0;
}

static void _abcdkvnet_ifconfig_flush(abcdkvnet_t *ctx)
{
    if(ctx->virtual_tun_fd < 0)
        return ;

    abcdk_net_address_flush(ctx->virtual_tun_name);
    abcdk_net_route_flush( ctx->virtual_tun_name);
}

static int _abcdkvnet_ifconfig_setup(abcdkvnet_t *ctx)
{
    char local4str[100] = {0},local6str[100] = {0};
    char gw4str[100] = {0},gw6str[100] = {0};
    int tun_idx;
    int chk;

    if (ctx->virtual_tun_fd < 0)
    {
        for (tun_idx = 0; tun_idx < 100; tun_idx++)
        {
            memset(ctx->virtual_tun_name, 0, NAME_MAX);
            snprintf(ctx->virtual_tun_name, NAME_MAX, "%s%d", ctx->virtual_tun_prefix, tun_idx);

            ctx->virtual_tun_fd = _abcdkvnet_tun_open(ctx->virtual_tun_name);
            if (ctx->virtual_tun_fd >= 0)
                break;
        }
        
        if(ctx->virtual_tun_fd < 0)
            return -1;

        /*添加异步标志。*/
        chk = abcdk_fflag_set(ctx->virtual_tun_fd,O_NONBLOCK);
    }

    abcdk_sockaddr_to_string(local4str, &ctx->virtual_local_addr4);
    abcdk_sockaddr_to_string(local6str, &ctx->virtual_local_addr6);
    abcdk_sockaddr_to_string(gw4str, &ctx->virtual_gateway_addr4);
    abcdk_sockaddr_to_string(gw6str, &ctx->virtual_gateway_addr6);

    chk = abcdk_net_address_add(4,local4str,ctx->virtual_mask_prefix4,gw4str,0,ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN(%s)添加地址(%s)失败，权限不足或系统错误。", ctx->virtual_tun_name,local4str);
        return -1;
    }

    chk = abcdk_net_address_add(6,local6str,ctx->virtual_mask_prefix6,gw6str,0,ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN(%s)添加地址(%s)失败，权限不足或系统错误。", ctx->virtual_tun_name,local6str);
        return -1;
    }

    chk = abcdk_net_set_mtu(ctx->virtual_tun_mtu,ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "更新TUN(%s)最大传输单元(%s)失败，权限不足或系统错误。", ctx->virtual_tun_name,ctx->virtual_tun_mtu);
        return -1;
    }


    chk = abcdk_net_up(ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "TUN(%s)启动失败，权限不足或系统错误。", ctx->virtual_tun_name);
        return -1;
    }

    /*服务端不需要配置默认路由。*/
    if(ctx->role == ABCDKVNET_ROLE_SERVER)
        return 0;

    /*客户端可能不需要设置全局虚拟地址路由。*/
    if(!ctx->virtual_default_route)
        return 0;

    chk = abcdk_net_route_add(4, "0.0.0.0", 0, gw4str, 0, ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN设备(%s)添加默认的网关(%s)失败，权限不足或系统错误。", ctx->virtual_tun_name,gw4str);
        return -1;
    }

    abcdk_net_route_add(6, "::0", 0, gw6str, 0, ctx->virtual_tun_name);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "向TUN设备(%s)添加默认的路由(%s)失败，权限不足或系统错误。", ctx->virtual_tun_name,gw6str);
        return -1;
    }

    return 0;
}

static int _abcdkvnet_ipool_allocate(abcdkvnet_t *ctx, int type, uint8_t *mask ,abcdk_sockaddr_t *addr)
{
    int chk = -1;

    /*分配IP地址。*/
    if (type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        if (addr->family == AF_INET)
        {
            if (*mask != ctx->virtual_ipv4_pool_prefix)
                return -1;

            chk = abcdk_ipool_static_request(ctx->virtual_ipv4_pool, addr);
        }
        else if (addr->family == AF_INET6)
        {
            if (*mask != ctx->virtual_ipv6_pool_prefix)
                return -1;

            chk = abcdk_ipool_static_request(ctx->virtual_ipv6_pool, addr);
        }
    }
    else if (type == ABCDKVNET_IPADDR_TYPE_DHCP)
    {
        if (addr->family == AF_INET)
        {
            *mask = ctx->virtual_ipv4_pool_prefix;
            chk = abcdk_ipool_dhcp_request(ctx->virtual_ipv4_pool, addr);
        }
        else if (addr->family == AF_INET6)
        {
            *mask = ctx->virtual_ipv6_pool_prefix;
            chk = abcdk_ipool_dhcp_request(ctx->virtual_ipv6_pool, addr);
        }
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

static int _abcdkvnet_iplan_register(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, abcdk_sockaddr_t *addr4,abcdk_sockaddr_t *addr6)
{
    abcdk_srpc_session_t *session_p;
    int chk;

    abcdk_mutex_lock(ctx->virtual_route_locker,1);

    if (addr4->family == AF_INET)
    {
        /*注册到路由表中。*/
        session_p = abcdk_srpc_refer(session);
        chk = abcdk_iplan_insert(ctx->virtual_route_list, addr4, session_p);
        if (chk != 0)
        {
            abcdk_srpc_unref(&session_p);
            goto END;
        }
    }

    if (addr6->family == AF_INET6)
    {
        /*注册到路由表中。*/
        session_p = abcdk_srpc_refer(session);
        chk = abcdk_iplan_insert(ctx->virtual_route_list, addr6, session_p);
        if (chk != 0)
        {
            abcdk_srpc_unref(&session_p);
            goto END;
        }
    }

    /*OK.*/
    chk = 0;

END:

    abcdk_mutex_unlock(ctx->virtual_route_locker);
    return chk;
    
}

static void _abcdkvnet_iplan_unregister(abcdkvnet_t *ctx, abcdk_sockaddr_t *addr4,abcdk_sockaddr_t *addr6)
{
    abcdk_srpc_session_t *session_p;

    abcdk_mutex_lock(ctx->virtual_route_locker,1);

    if(addr4->family == AF_INET)
    {
        session_p = (abcdk_srpc_session_t *)abcdk_iplan_remove(ctx->virtual_route_list,addr4);
        abcdk_srpc_unref(&session_p);
    }

    if(addr6->family == AF_INET6)
    {
        session_p = (abcdk_srpc_session_t *)abcdk_iplan_remove(ctx->virtual_route_list,addr6);
        abcdk_srpc_unref(&session_p);
    }

    abcdk_mutex_unlock(ctx->virtual_route_locker);
}

static abcdk_srpc_session_t *_abcdkvnet_iplan_lookup(abcdkvnet_t *ctx, abcdk_sockaddr_t *addr)
{
    abcdk_srpc_session_t *session_p;

    if(addr->family != AF_INET && addr->family != AF_INET6)
        return NULL;

    abcdk_mutex_lock(ctx->virtual_route_locker,1);

    session_p = (abcdk_srpc_session_t *)abcdk_iplan_lookup(ctx->virtual_route_list,addr);
    if(session_p)
        session_p = abcdk_srpc_refer(session_p);

    abcdk_mutex_unlock(ctx->virtual_route_locker);

    return session_p;
}

static abcdk_srpc_session_t *_abcdkvnet_iplan_lookup_iphdr_dst(abcdkvnet_t *ctx, const void *data)
{
    uint8_t ipver;
    abcdk_sockaddr_t dst;
    struct iphdr *ipv4hdr_p;
    struct ip6_hdr *ipv6hdr_p;

    ipver = ABCDK_PTR2U8(data,0) >> 4;

    if(ipver == 4)
    {
        ipv4hdr_p = (struct iphdr *)ABCDK_PTR2VPTR(data,0);

        dst.family = AF_INET;
        dst.addr4.sin_addr.s_addr = ipv4hdr_p->daddr;
    }
    else if(ipver == 6)
    {
        ipv6hdr_p = (struct ip6_hdr *)ABCDK_PTR2VPTR(data,0);

        dst.family = AF_INET6;
        dst.addr6.sin6_addr = ipv6hdr_p->ip6_dst;
    }

    return _abcdkvnet_iplan_lookup(ctx,&dst);
}

static int _abcdkvnet_server_ip_allocate(abcdkvnet_t *ctx, abcdk_srpc_session_t *session,
                                         int type4, uint8_t *mask4, abcdk_sockaddr_t *addr4,
                                         int type6, uint8_t *mask6, abcdk_sockaddr_t *addr6)
{
    abcdkvnet_node_t *node_ctx_p = NULL;
    char addr4str[100] = {0},addr6str[100] = {0};
    int chk = -1;

    if(!session)
        return -1;

    node_ctx_p = (abcdkvnet_node_t *) abcdk_srpc_get_userdata(session);
    
    /*分配IPV4地址。*/
    chk = _abcdkvnet_ipool_allocate(ctx,type4,mask4,addr4);
    if(chk != 0)
        return -11;

    /*绑定IP地址。*/
    node_ctx_p->virtual_addr4 = *addr4;

    /*分配IPV6地址。*/
    chk = _abcdkvnet_ipool_allocate(ctx,type6,mask6,addr6);
    if(chk != 0)
        return -11;

    /*绑定IP地址。*/
    node_ctx_p->virtual_addr6 = *addr6;
    
    /*注册到路由表中。*/
    chk = _abcdkvnet_iplan_register(ctx,session,addr4,addr6);
    if(chk != 0)
        return -1;

    abcdk_sockaddr_to_string(addr4str, addr4);
    abcdk_sockaddr_to_string(addr6str, addr6);

    abcdk_srpc_trace_output(session,LOG_INFO,"分配给客户端的虚拟址地是IPV4(%s)和IPV6(%s)",addr4str,addr6str);

    return 0;
}

static void _abcdkvnet_server_ip_reclaim(abcdkvnet_t *ctx, abcdk_srpc_session_t *session)
{
    abcdkvnet_node_t *node_ctx_p = NULL;
    char addr4str[100] = {0},addr6str[100] = {0};

    if(!session)
        return;

    node_ctx_p = (abcdkvnet_node_t *) abcdk_srpc_get_userdata(session);

    if(node_ctx_p->virtual_addr4.family != AF_INET && node_ctx_p->virtual_addr6.family != AF_INET6)
        return;

    /*从路由表中删除。*/
    _abcdkvnet_iplan_unregister(ctx,&node_ctx_p->virtual_addr4,&node_ctx_p->virtual_addr6);

    /*回收IP地址。*/
    _abcdkvnet_ipool_reclaim(ctx,&node_ctx_p->virtual_addr4);
    _abcdkvnet_ipool_reclaim(ctx,&node_ctx_p->virtual_addr6);

    if(node_ctx_p->virtual_addr4.family == AF_INET)
        abcdk_sockaddr_to_string(addr4str, &node_ctx_p->virtual_addr4);
    if(node_ctx_p->virtual_addr6.family == AF_INET6)
        abcdk_sockaddr_to_string(addr6str, &node_ctx_p->virtual_addr6);

    abcdk_srpc_trace_output(session,LOG_INFO,"从客户端回收的虚拟址地是IPV4(%s)和IPV6(%s)",addr4str,addr6str);
}

static int _abcdkvnet_server_cmd_logon(abcdkvnet_t *ctx,abcdk_srpc_session_t *session,abcdk_bit_t *req,abcdk_object_t **rsp)
{
    abcdk_object_t *rsp_p = NULL;
    abcdk_bit_t rspbit = {0};
    int type4,type6;
    uint8_t mask4,mask6;
    abcdk_sockaddr_t addr4 = {AF_INET},addr6 = {AF_INET6};
    int chk,chk2;

    if (req->size < 26)
        return -1;

    type4 = abcdk_bit_read2number(req,8);
    mask4 = abcdk_bit_read2number(req,8);
    abcdk_bit_read2buffer(req,(uint8_t*)&addr4.addr4.sin_addr.s_addr,4);

    type6 = abcdk_bit_read2number(req,8);
    mask6 = abcdk_bit_read2number(req,8);
    abcdk_bit_read2buffer(req,addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);
    
    /*为了防止客户端重复申请从而造成的IP地址丢失，因此要先回收之前注册的地址。*/
    _abcdkvnet_server_ip_reclaim(ctx,session);
    
    /*申请IP地址。*/
    chk = _abcdkvnet_server_ip_allocate(ctx,session,type4,&mask4,&addr4,type6,&mask6,&addr6);

    rsp_p = abcdk_object_alloc2(100);
    if(!rsp_p)
        return -1;

    rspbit.data = rsp_p->pptrs[0];
    rspbit.size = rsp_p->sizes[0];

    abcdk_bit_write_number(&rspbit,16,ABCDKVNET_CMD_LOGON);
    
    if(chk != 0)
    {
        abcdk_bit_write_number(&rspbit,32,11);
        abcdk_bit_write_number(&rspbit,8,0);
        abcdk_bit_write_number(&rspbit,32,0);//padding
        abcdk_bit_write_number(&rspbit,8,0);
        abcdk_bit_write_number(&rspbit,64,0);//padding
        abcdk_bit_write_number(&rspbit,64,0);//padding
        abcdk_bit_write_number(&rspbit,32,0);//padding
        abcdk_bit_write_number(&rspbit,64,0);//padding
        abcdk_bit_write_number(&rspbit,64,0);//padding
    }
    else 
    {
        abcdk_bit_write_number(&rspbit,32,0);
        abcdk_bit_write_number(&rspbit,8,mask4);
        abcdk_bit_write_buffer(&rspbit,(uint8_t*)&addr4.addr4.sin_addr.s_addr,4);
        abcdk_bit_write_number(&rspbit,8,mask6);
        abcdk_bit_write_buffer(&rspbit,addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);
        abcdk_bit_write_buffer(&rspbit,(uint8_t*)&ctx->virtual_local_addr4.addr4.sin_addr.s_addr,4);
        abcdk_bit_write_buffer(&rspbit,ctx->virtual_local_addr6.addr6.sin6_addr.__in6_u.__u6_addr8,16);

    }

    /*有效长度*/
    rsp_p->sizes[0] = rspbit.pos /8;

    *rsp = rsp_p;

    return 0;
}

static int _abcdkvnet_server_cmd_posting(abcdkvnet_t *ctx,abcdk_srpc_session_t *session,abcdk_bit_t *req,abcdk_object_t **rsp)
{
    abcdk_srpc_session_t *rpc_subnet_p;
    int16_t data_l;
    void *data_p;
    ssize_t wlen;

    if (req->size < 4)
        return -1;

    data_l = (int16_t)abcdk_bit_read2number(req,16);
    if(data_l <= 0)
        return 0; //仅用于更新活动时间。

#if 0
    /*检查数据包长度是否超过最大传输单元。*/
    if(data_l > ctx->virtual_tun_mtu)
        return -1;
#endif 
    
    data_p = ABCDK_PTR2VPTR(req->data,4);

    /*优先在子网中查找。*/
    rpc_subnet_p = _abcdkvnet_iplan_lookup_iphdr_dst(ctx,data_p);
    if(rpc_subnet_p)
    {
        abcdk_srpc_request(rpc_subnet_p, req->data, 4 + data_l, NULL);
        abcdk_srpc_unref(&rpc_subnet_p);
    }
    else
    {
        wlen = _abcdkvnet_tun_write(ctx->virtual_tun_fd,data_p,data_l);
        if(wlen == 0)
            return -1;
    }

    return 0;
}

static int _abcdkvnet_server_cmd_process(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, const void *data, size_t size,abcdk_object_t **rsp)
{
    abcdk_bit_t req = {0};
    uint16_t cmd;
    int chk;

    req.data = (void*)data;
    req.size = size;
    
    if (req.size < 2)
        return -1;

    cmd = abcdk_bit_read2number(&req,16);

    if(cmd == ABCDKVNET_CMD_LOGON)
        chk = _abcdkvnet_server_cmd_logon(ctx,session,&req,rsp);
    else if(cmd == ABCDKVNET_CMD_POSTING)
        chk = _abcdkvnet_server_cmd_posting(ctx,session,&req,rsp);
    else 
        chk = -1;

    return chk;
}

static int _abcdkvnet_server_offline_client(abcdkvnet_t *ctx,abcdk_srpc_session_t *session)
{
    _abcdkvnet_server_ip_reclaim(ctx,session);
}

static int _abcdkvnet_client_cmd_posting(abcdkvnet_t *ctx,abcdk_srpc_session_t *session,abcdk_bit_t *req,abcdk_object_t **rsp)
{
    int16_t data_l;
    void *data_p;
    ssize_t wlen;

    if (req->size < 4)
        return -1;

    data_l = (int16_t)abcdk_bit_read2number(req,16);
    if(data_l <= 0)
        return 0; //仅用于更新活动时间。

#if 0
    /*检查数据包长度是否超过最大传输单元。*/
    if(data_l > ctx->virtual_tun_mtu)
        return -1;
#endif

    data_p = ABCDK_PTR2VPTR(req->data,4);

    wlen = _abcdkvnet_tun_write(ctx->virtual_tun_fd,data_p,data_l);
    if(wlen == 0)
        return -1;

    return 0;
}

static int _abcdkvnet_client_cmd_process(abcdkvnet_t *ctx,abcdk_srpc_session_t *session, const void *data, size_t size,abcdk_object_t **rsp)
{
    abcdk_bit_t req = {0};
    uint16_t cmd;
    int chk = -1;

    req.data = (void*)data;
    req.size = size;

    if (req.size < 2)
        return -1;

    cmd = abcdk_bit_read2number(&req,16);

    if(cmd == ABCDKVNET_CMD_POSTING)
        chk = _abcdkvnet_client_cmd_posting(ctx,session,&req,rsp);
    else 
        chk = -1;

    return chk;
}

static int _abcdkvnet_client_offline_server(abcdkvnet_t *ctx,abcdk_srpc_session_t *session)
{
    return 0;
}

static void _abcdkvnet_srpc_prepare_cb(void *opaque,abcdk_srpc_session_t **session,abcdk_srpc_session_t *listen)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
    abcdk_srpc_session_t *session_p = NULL;

    session_p = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_SERVER);
    if(!session_p)
        return;

    *session = session_p;
}

static void _abcdkvnet_srpc_request_cb(void *opaque, abcdk_srpc_session_t *session, uint64_t mid, const void *data, size_t size)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
    abcdkvnet_node_t *node_ctx_p;
    abcdk_object_t *rsp = NULL;
    int chk;

    node_ctx_p = (abcdkvnet_node_t *) abcdk_srpc_get_userdata(session);

    if(node_ctx_p->flag == ABCDKVNET_ROLE_SERVER)
        chk = _abcdkvnet_server_cmd_process(ctx,session,data,size,&rsp);
    else if(node_ctx_p->flag == ABCDKVNET_ROLE_CLIENT)
        chk = _abcdkvnet_client_cmd_process(ctx,session,data,size,&rsp);
    else 
        chk = -1;   

    if(chk != 0)
    {
        abcdk_srpc_set_timeout(session,1);
        return;
    }

    if(rsp)
    {
        abcdk_srpc_response(session,mid,rsp->pptrs[0],rsp->sizes[0]);
        abcdk_object_unref(&rsp);
    }
}

static void _abcdkvnet_srpc_close_cb(void *opaque,abcdk_srpc_session_t *session)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;
    abcdkvnet_node_t *node_ctx_p;

    node_ctx_p = (abcdkvnet_node_t *) abcdk_srpc_get_userdata(session);

    if(node_ctx_p->flag == ABCDKVNET_ROLE_LISTENER)
        return;

    if(node_ctx_p->flag == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_server_offline_client(ctx,session);
    else if(node_ctx_p->flag == ABCDKVNET_ROLE_CLIENT)
        _abcdkvnet_client_offline_server(ctx,session);

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
        abcdk_trace_output(LOG_ERR, "监听地址(%s)无法识别。", listen_p);
        return -1;
    }

    if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
        session_p = ctx->rpc_listen_raw_session = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_LISTENER);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        session_p = ctx->rpc_listen_pki_session = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_LISTENER);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        session_p = ctx->rpc_listen_enigma_session = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_LISTENER);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        session_p = ctx->rpc_listen_pki_enigma_session = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_LISTENER);

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
        abcdk_trace_output(LOG_ERR, "启动监听(%s')失败，无权限或地址错误或端口被占用。", listen_p);
        return -3;
    }

    return 0;
}

static void _abcdkvnet_server_tun_transfer(abcdkvnet_t *ctx)
{
    abcdk_object_t *reqbuf;
    abcdk_bit_t reqbit = {0};
    abcdk_srpc_session_t *rpc_dst_p;
    int chk;

    reqbuf = abcdk_object_alloc2(ctx->virtual_tun_mtu+4);
    if(!reqbuf)
        return;

    reqbit.data = reqbuf->pptrs[0];
    reqbit.size = reqbuf->sizes[0];

LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        goto END;

    reqbit.pos = 0;
    memset(reqbit.data,0,4);//至少把长度值覆盖掉。

    abcdk_bit_write_number(&reqbit,16,ABCDKVNET_CMD_POSTING);

    chk = _abcdkvnet_tun_poll(ctx->virtual_tun_fd,0x01,3*1000);
    if(chk < 0)
    {
        goto END;
    } 
    else if(chk == 0)
    {
        goto LOOP;
    }
    else 
    {
        chk = _abcdkvnet_tun_read(ctx->virtual_tun_fd,ABCDK_PTR2VPTR(reqbit.data,4),ctx->virtual_tun_mtu);
        if(chk == 0)
            goto END;
        else if(chk < 0)
        {
            if(errno == EAGAIN)
                goto LOOP;
            else 
                goto END;
        }
        else  
        {
            abcdk_bit_write_number(&reqbit,16,chk);
            abcdk_bit_seek(&reqbit,chk*8);//数据已经填充完毕，这里仅移动游标即可。
        }
    }

    rpc_dst_p = _abcdkvnet_iplan_lookup_iphdr_dst(ctx,ABCDK_PTR2VPTR(reqbit.data,4));
    if(!rpc_dst_p)
        goto LOOP;

    chk = abcdk_srpc_request(rpc_dst_p,reqbit.data,reqbit.pos/8,NULL);
    abcdk_srpc_unref(&rpc_dst_p);

    /*因为链路保活由客户端负责，所以不需要关心转发状态。*/
    goto LOOP;

END:

    abcdk_object_unref(&reqbuf);
}

static void _abcdkvnet_server_dowork(abcdkvnet_t *ctx)
{
    int chk;

LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        return;

    _abcdkvnet_ifconfig_flush(ctx);

    chk = _abcdkvnet_ifconfig_setup(ctx);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR,"配置虚拟地址失败。");
        abcdk_closep(&ctx->virtual_tun_fd);
        goto ERR;
    }

    _abcdkvnet_server_tun_transfer(ctx);

ERR:

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
    ctx->virtual_static_prefix4 = abcdk_option_get_int(ctx->args, "--virtual-static-prefix4", 0, 24);
    ctx->virtual_static_prefix6 = abcdk_option_get_int(ctx->args, "--virtual-static-prefix6", 0, 120);
    ctx->virtual_static_addr4 = abcdk_option_get(ctx->args, "--virtual-static-addr4", 0, NULL);
    ctx->virtual_static_addr6 = abcdk_option_get(ctx->args, "--virtual-static-addr6", 0, NULL);
    ctx->virtual_tun_prefix = abcdk_option_get(ctx->args, "--virtual-tun-prefix", 0, "vnet");
    ctx->virtual_tun_mtu = abcdk_option_get_int(ctx->args, "--virtual-tun-mtu", 0, 1500);

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

    ABCDK_CLAMP(ctx->virtual_tun_mtu,1500,65535);
    
    if (ctx->virtual_addr4_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk = abcdk_sockaddr_from_string(&ctx->virtual_local_addr4, ctx->virtual_static_addr4, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_WARNING, "虚拟静态的地址(%s)解析错误。", ctx->virtual_static_addr4);
            goto END;
        }

        ctx->virtual_mask_prefix4 = ctx->virtual_static_prefix4;
    }
    

    if (ctx->virtual_addr6_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk = abcdk_sockaddr_from_string(&ctx->virtual_local_addr6, ctx->virtual_static_addr6, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_WARNING, "虚拟静态的地址(%s)解析错误。", ctx->virtual_static_addr6);
            goto END;
        }

        ctx->virtual_mask_prefix6 = ctx->virtual_static_prefix6;
    }

    ctx->virtual_ipv4_pool = abcdk_ipool_create2(ipv4_pool_begin_p,ipv4_pool_end_p);
    if(!ctx->virtual_ipv4_pool)
        abcdk_trace_output(LOG_WARNING,"虚拟地址池(%s,%s)的范围无效。",ipv4_pool_begin_p,ipv4_pool_end_p);

    ctx->virtual_ipv6_pool = abcdk_ipool_create2(ipv6_pool_begin_p,ipv6_pool_end_p);
    if(!ctx->virtual_ipv6_pool)
        abcdk_trace_output(LOG_WARNING,"虚拟地址池(%s,%s)的范围无效。",ipv6_pool_begin_p,ipv6_pool_end_p);

    if(!ctx->virtual_ipv4_pool || !ctx->virtual_ipv6_pool)
        goto END;

    ctx->virtual_ipv4_pool_prefix = abcdk_ipool_prefix(ctx->virtual_ipv4_pool);
    ctx->virtual_ipv6_pool_prefix = abcdk_ipool_prefix(ctx->virtual_ipv6_pool);
    
    if(ipv4_pool_dhcp_begin_p && ipv4_pool_dhcp_end_p)
    {
        chk = abcdk_ipool_set_dhcp_range2(ctx->virtual_ipv4_pool,ipv4_pool_dhcp_begin_p,ipv4_pool_dhcp_end_p);
        if(chk != 0)
        {
            abcdk_trace_output(LOG_WARNING,"虚拟地址池(%s,%s)设置动态范围(%s,%s)错误。",ipv4_pool_begin_p,ipv4_pool_end_p,ipv4_pool_dhcp_begin_p,ipv4_pool_dhcp_end_p);
            goto END;
        }
    }
    
    if(ipv6_pool_dhcp_begin_p && ipv6_pool_dhcp_end_p)
    {
        chk = abcdk_ipool_set_dhcp_range2(ctx->virtual_ipv6_pool,ipv6_pool_dhcp_begin_p,ipv6_pool_dhcp_end_p);
        if(chk != 0)
        {
            abcdk_trace_output(LOG_WARNING,"虚拟地址池(%s,%s)设置动态范围(%s,%s)错误。",ipv6_pool_begin_p,ipv6_pool_end_p,ipv6_pool_dhcp_begin_p,ipv6_pool_dhcp_end_p);
            goto END;
        }
    }

    /*请求IPV4地址。*/
    ctx->virtual_local_addr4.family = AF_INET;
    chk = _abcdkvnet_ipool_allocate(ctx, ctx->virtual_addr4_type,&ctx->virtual_mask_prefix4, &ctx->virtual_local_addr4);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "请求虚拟地址失败，超出地址池(%s,%s)有效范围。",ipv4_pool_begin_p,ipv4_pool_end_p);
        goto END;
    }

    /*请求IPV6地址。*/
    ctx->virtual_local_addr6.family = AF_INET6;
    chk = _abcdkvnet_ipool_allocate(ctx, ctx->virtual_addr6_type,&ctx->virtual_mask_prefix6, &ctx->virtual_local_addr6);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "请求虚拟地址失败，超出地址池(%s,%s)有效范围。",ipv6_pool_begin_p,ipv6_pool_end_p);
        goto END;
    }

    /*自己就是网关。*/
    ctx->virtual_gateway_addr4 = ctx->virtual_local_addr4;
    ctx->virtual_gateway_addr6 = ctx->virtual_local_addr6;
    
    ctx->virtual_route_list = abcdk_iplan_create();
    if(!ctx->virtual_route_list)
        goto END;

    ctx->virtual_route_locker = abcdk_mutex_create();
    if(!ctx->virtual_route_locker)
        goto END;

    max_client = ABCDK_MAX(abcdk_ipool_count(ctx->virtual_ipv4_pool,0),abcdk_ipool_count(ctx->virtual_ipv6_pool,0));
    if(max_client <= 0)
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
    _abcdkvnet_node_free(&ctx->rpc_listen_raw_session);
    _abcdkvnet_node_free(&ctx->rpc_listen_pki_session);
    _abcdkvnet_node_free(&ctx->rpc_listen_enigma_session);
    _abcdkvnet_node_free(&ctx->rpc_listen_pki_enigma_session);
    abcdk_ipool_destroy(&ctx->virtual_ipv4_pool);
    abcdk_ipool_destroy(&ctx->virtual_ipv6_pool);
    abcdk_iplan_destroy(&ctx->virtual_route_list);
    abcdk_mutex_destroy(&ctx->virtual_route_locker);
}

static int _abcdkvnet_client_connect_uplink(abcdkvnet_t *ctx)
{
    int ssl_scheme;
    abcdk_srpc_config_t rpc_cfg = {0};
    int chk;

    ctx->rpc_uplink_session = _abcdkvnet_node_alloc(ctx,ABCDKVNET_ROLE_CLIENT);
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

    chk = abcdk_srpc_connect(ctx->rpc_uplink_session,&ctx->rpc_uplink_addr,&rpc_cfg);
    if(chk != 0)
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session,LOG_ERR, "连接上行地址(%s)失败，网络不通或目标不可达。", ctx->uplink_addr);
        return -2;
    }

    return 0;
}

static int _abcdkvnet_client_logon(abcdkvnet_t *ctx)
{
    char reqbuf[100] = {0};
    abcdk_bit_t reqbit = {0,reqbuf,100},rspbit = {0};
    abcdk_object_t *rsp = NULL;
    char local4str[100] = {0},local6str[100] = {0};
    char gw4str[100] = {0},gw6str[100] = {0};
    int err;
    int chk;
    
    abcdk_bit_write_number(&reqbit,16,ABCDKVNET_CMD_LOGON);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_addr4_type);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_mask_prefix4);
    abcdk_bit_write_buffer(&reqbit, (uint8_t *)&ctx->virtual_local_addr4.addr4.sin_addr.s_addr, 4);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_addr6_type);
    abcdk_bit_write_number(&reqbit, 8, ctx->virtual_mask_prefix6);
    abcdk_bit_write_buffer(&reqbit, ctx->virtual_local_addr6.addr6.sin6_addr.__in6_u.__u6_addr8, 16);
         
    chk = abcdk_srpc_request(ctx->rpc_uplink_session,reqbit.data,reqbit.pos/8,&rsp);
    if(chk != 0)
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session,LOG_ERR, "登录注册失败，上行地址(%s)网络不通或目标不可达。", ctx->uplink_addr);
        return -1;
    }

    rspbit.data = rsp->pptrs[0];
    rspbit.size = rsp->sizes[0];

    abcdk_bit_seek(&rspbit,16);
    err = abcdk_bit_read2number(&rspbit,32);
    if (err == 0)
    {
        ctx->virtual_mask_prefix4 = abcdk_bit_read2number(&rspbit,8);

        ctx->virtual_local_addr4.family = AF_INET;
        abcdk_bit_read2buffer(&rspbit, (uint8_t *)&ctx->virtual_local_addr4.addr4.sin_addr.s_addr, 4);

        ctx->virtual_mask_prefix6 = abcdk_bit_read2number(&rspbit,8);

        ctx->virtual_local_addr6.family = AF_INET6;
        abcdk_bit_read2buffer(&rspbit, ctx->virtual_local_addr6.addr6.sin6_addr.__in6_u.__u6_addr8, 16);

        ctx->virtual_gateway_addr4.family = AF_INET;
        abcdk_bit_read2buffer(&rspbit, (uint8_t *)&ctx->virtual_gateway_addr4.addr4.sin_addr.s_addr, 4);

        ctx->virtual_gateway_addr6.family = AF_INET6;
        abcdk_bit_read2buffer(&rspbit, ctx->virtual_gateway_addr6.addr6.sin6_addr.__in6_u.__u6_addr8, 16);

        abcdk_sockaddr_to_string(local4str, &ctx->virtual_local_addr4);
        abcdk_sockaddr_to_string(local6str, &ctx->virtual_local_addr6);
        abcdk_sockaddr_to_string(gw4str, &ctx->virtual_gateway_addr4);
        abcdk_sockaddr_to_string(gw6str, &ctx->virtual_gateway_addr6);

        abcdk_srpc_trace_output(ctx->rpc_uplink_session, LOG_INFO, "登录注册完成，分配的虚拟地址是IPV4(%s)和IPV6(%s)，网关的虚拟地址是IPV4(%s)和IPV6(%s)。",
                                local4str, local6str,gw4str,gw6str);

        chk = 0;
    }
    else 
    {
        abcdk_srpc_trace_output(ctx->rpc_uplink_session, LOG_ERR, "登录注册失败，服务不可用或资源不足(ERRNO=%d)。",err);

        chk = -1;
    }

    abcdk_object_unref(&rsp);

    return chk;
}

static void _abcdkvnet_client_tun_transfer(abcdkvnet_t *ctx)
{
    abcdk_object_t *reqbuf;
    abcdk_bit_t reqbit = {0};
    int chk;

    reqbuf = abcdk_object_alloc2(ctx->virtual_tun_mtu+4);
    if(!reqbuf)
        return;

    reqbit.data = reqbuf->pptrs[0];
    reqbit.size = reqbuf->sizes[0];

LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        goto END;

    reqbit.pos = 0;
    memset(reqbit.data,0,4);//至少把长度值覆盖掉。

    abcdk_bit_write_number(&reqbit,16,ABCDKVNET_CMD_POSTING);

    chk = _abcdkvnet_tun_poll(ctx->virtual_tun_fd,0x01,3*1000);
    if(chk < 0)
    {
        goto END;
    }
    else if(chk == 0)
    {
        /*没有数据时仅发送保活消息。*/
        abcdk_bit_write_number(&reqbit,16,0);
    }
    else 
    {
        chk = _abcdkvnet_tun_read(ctx->virtual_tun_fd,ABCDK_PTR2VPTR(reqbit.data,4),ctx->virtual_tun_mtu);
        if(chk == 0)
            goto END;
        else if(chk < 0)
        {
            if(errno == EAGAIN)
                goto LOOP;
            else 
                goto END;
        }
        else  
        {
            abcdk_bit_write_number(&reqbit,16,chk);
            abcdk_bit_seek(&reqbit,chk*8);//数据已经填充完毕，这里仅移动游标即可。
        }
    }

    chk = abcdk_srpc_request(ctx->rpc_uplink_session,reqbit.data,reqbit.pos/8,NULL);
    if(chk == 0)
        goto LOOP;

END:

    abcdk_object_unref(&reqbuf);
}

static void _abcdkvnet_client_dowork(abcdkvnet_t *ctx)
{
    int chk;
    
LOOP:

    /*检查是否需要退出。*/
    if(abcdk_atomic_compare(&ctx->exit_flag,1))
        return;

    _abcdkvnet_ifconfig_flush(ctx);

    chk = _abcdkvnet_client_connect_uplink(ctx);
    if(chk != 0)
        goto ERR;

    chk = _abcdkvnet_client_logon(ctx);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR,"登录注册失败。");
        goto ERR;
    }

    chk = _abcdkvnet_ifconfig_setup(ctx);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_ERR,"配置虚拟地址失败。");
        abcdk_closep(&ctx->virtual_tun_fd);
        goto ERR;
    }

    _abcdkvnet_client_tun_transfer(ctx);

ERR:

    /*快速关闭。*/
    if(ctx->rpc_uplink_session)
        abcdk_srpc_set_timeout(ctx->rpc_uplink_session,1);
        
    _abcdkvnet_node_free(&ctx->rpc_uplink_session);

    sleep(1);
    goto LOOP;
}


static void _abcdkvnet_process_client(abcdkvnet_t *ctx)
{
    int chk4,chk6;
    int chk;

    ctx->virtual_addr4_type = abcdk_option_get_int(ctx->args, "--virtual-addr4-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_addr6_type = abcdk_option_get_int(ctx->args, "--virtual-addr6-type", 0, ABCDKVNET_IPADDR_TYPE_DHCP);
    ctx->virtual_static_prefix4 = abcdk_option_get_int(ctx->args, "--virtual-static-prefix4", 0, 24);
    ctx->virtual_static_prefix6 = abcdk_option_get_int(ctx->args, "--virtual-static-prefix6", 0, 120);
    ctx->virtual_static_addr4 = abcdk_option_get(ctx->args, "--virtual-static-addr4", 0, "");
    ctx->virtual_static_addr6 = abcdk_option_get(ctx->args, "--virtual-static-addr6", 0, "");
    ctx->virtual_tun_prefix = abcdk_option_get(ctx->args, "--virtual-tun-prefix", 0, "vnet");
    ctx->virtual_tun_mtu = abcdk_option_get_int(ctx->args, "--virtual-tun-mtu", 0, 1500);
    ctx->virtual_default_route = abcdk_option_get_int(ctx->args, "--virtual-default-route", 0, 0);

    ctx->pki_ca_file = abcdk_option_get(ctx->args, "--pki-ca-file", 0, NULL);
    ctx->pki_ca_path = abcdk_option_get(ctx->args, "--pki-ca-path", 0, NULL);
    ctx->pki_cert_file = abcdk_option_get(ctx->args, "--pki-cert-file", 0, NULL);
    ctx->pki_key_file = abcdk_option_get(ctx->args, "--pki-key-file", 0, NULL);
    ctx->pki_check_cert = abcdk_option_get_int(ctx->args, "--pki-check-cert", 0, 1);

    ctx->enigma_key_file = abcdk_option_get(ctx->args, "--enigma-key-file", 0, NULL);
    ctx->enigma_salt_size = abcdk_option_get_int(ctx->args, "--enigma-salt-size", 0, 123);
     
    ctx->uplink_ssl_scheme = abcdk_option_get_int(ctx->args, "--uplink-ssl-scheme", 0, ABCDK_ASIO_SSL_SCHEME_RAW);
    ctx->uplink_addr = abcdk_option_get(ctx->args, "--uplink-addr", 0, "");
    ctx->uplink_gateway = abcdk_option_get(ctx->args, "--uplink-gateway", 0, "");

    ABCDK_CLAMP(ctx->virtual_tun_mtu,1500,65535);

    if (ctx->virtual_addr4_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk4 = abcdk_sockaddr_from_string(&ctx->virtual_local_addr4, ctx->virtual_static_addr4, 0);
        if (chk4 != 0)
        {
            abcdk_trace_output(LOG_WARNING, "静态虚拟地址(%s)解析错误。", ctx->virtual_static_addr4);
            goto END;
        }

        ctx->virtual_mask_prefix4 = ctx->virtual_static_prefix4;

    }

    if (ctx->virtual_addr6_type == ABCDKVNET_IPADDR_TYPE_STATIC)
    {
        chk6 = abcdk_sockaddr_from_string(&ctx->virtual_local_addr6, ctx->virtual_static_addr6, 0);
        if (chk6 != 0)
        {
            abcdk_trace_output(LOG_WARNING, "静态虚拟地址(%s)解析错误。", ctx->virtual_static_addr6);
            goto END;
        }

        ctx->virtual_mask_prefix6 = ctx->virtual_static_prefix6;
    }

    chk = abcdk_sockaddr_from_string(&ctx->rpc_uplink_addr, ctx->uplink_addr, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "上行地址(%s)无法识别。", ctx->uplink_addr);
        goto END;
    }

    chk = abcdk_sockaddr_from_string(&ctx->rpc_uplink_gateway, ctx->uplink_gateway, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_WARNING, "上行网关(%s)无法识别，虚拟地址全局路由配置将被忽略。", ctx->uplink_gateway);
        ctx->virtual_default_route = 0;
    }
    else 
    {
        /*当客户端启用有效网关时才能够配置全局路由，否则会造成数据包“路由回环”错误(在TUN设备中)。*/
        chk = _abcdkvnet_uplink_route_add(ctx);
        if(chk != 0)
            return;
    }

    ctx->rpc_ctx = abcdk_srpc_create(10,-1);
    if(!ctx->rpc_ctx)
        goto END;

    _abcdkvnet_client_dowork(ctx);

END:

    _abcdkvnet_uplink_route_del(ctx);
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

    ctx->virtual_tun_fd = -1;

    ctx->role = abcdk_option_get_int(ctx->args,"--role",0,ABCDKVNET_ROLE_CLIENT);

    if(ctx->role == ABCDKVNET_ROLE_CLIENT)
        _abcdkvnet_process_client(ctx);
    else if(ctx->role == ABCDKVNET_ROLE_SERVER)
        _abcdkvnet_process_server(ctx);

    abcdk_closep(&ctx->virtual_tun_fd);

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
        if(getuid() != 0)
        {
            fprintf(stderr, "此服务需要root权限支持。\n");
            return 1;
        }

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

