/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

typedef struct _abcdk_ipcfg
{
    int errcode;
    abcdk_option_t *args;

    abcdk_logger_t *logger;

    /*0：运行，1：退出。*/
    volatile int exitflag;

    const char *dhclient_cmd;
    const char *udhcpc_cmd;

} abcdk_ipcfg_t;

typedef struct _abcdk_ipcfg_node
{
    /**/
    abcdk_ipcfg_t *father;

    int exitcode;
    int sigcode;

    pid_t dhcp_pid;

    /*
     * 应用状态。
     *
     * 0：未应用。
     * 1：已应用。
     */
    int implement_state;

    /*
     * 物理链路状态。
     *
     * 0：未连接。
     * 1：已连接。
     */
    int link_state;

    /*
     * 逻辑链路状态。
     *
     *  0：未启动。
     *  1：已启动。
     *
     */
    int oper_state;

    /* 地址哈希值。*/
    char addr_hcode[33];

    /* 旧的配置。*/
    abcdk_object_t *old_resolv;

    /* resolv哈希值。*/
    char resolv_hcode[33];

    /* 启动次数。*/
    int start_link_count;

    /* 下一次启动时间。*/
    uint64_t start_link_next;

} abcdk_ipcfg_node_t;

void _abcdk_ipcfg_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的IP配置工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--pid-file < FILE >\n");
    fprintf(stderr, "\t\tPID文件名(包括路径)。默认：/tmp/abcdk/pid/ipconfig.pid\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径。默认：/tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒)，1～60之间有效。默认：30\n");
    fprintf(stderr, "\t\t注：此功能不支持supervisor或类似的工具。\n");

    fprintf(stderr, "\n\t--conf < CONF [CONF ...] >\n");
    fprintf(stderr, "\t\t配置文件名(包括路径)。\n");
    fprintf(stderr, "\t\t注：具体的配置说明见doc/tool/ipconfig/*.sample。\n");

    fprintf(stderr, "\n\t--dhclient-cmd < NAME >\n");
    fprintf(stderr, "\t\tdhclient客户端工具名称(包括路径)。默认：dhclient\n");
    fprintf(stderr, "\t\t注：使能方法为DHCP时有效.\n");

    fprintf(stderr, "\n\t--udhcpc-cmd < NAME >\n");
    fprintf(stderr, "\t\tudhcpc客户端工具名称(包括路径)。默认：udhcpc\n");
    fprintf(stderr, "\t\t注：使能方法为DHCP时有效.\n");
}

void _abcdk_ipcfg_find_dhcp_client(abcdk_ipcfg_t *ctx)
{
    if (!ctx->dhclient_cmd)
    {
        if (access("/sbin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/sbin/dhclient";
        else if (access("/bin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/bin/dhclient";
        else if (access("/usr/sbin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/usr/sbin/dhclient";
        else if (access("/usr/bin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/usr/bin/dhclient";
        else if (access("/usr/local/sbin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/usr/local/sbin/dhclient";
        else if (access("/usr/local/bin/dhclient", X_OK) == 0)
            ctx->dhclient_cmd = "/usr/local/bin/dhclient";
    }

    if (!ctx->udhcpc_cmd)
    {
        if (access("/sbin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/sbin/udhcpc";
        else if (access("/bin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/bin/udhcpc";
        else if (access("/usr/sbin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/usr/sbin/udhcpc";
        else if (access("/usr/bin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/usr/bin/udhcpc";
        else if (access("/usr/local/sbin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/usr/local/sbin/udhcpc";
        else if (access("/usr/local/bin/udhcpc", X_OK) == 0)
            ctx->udhcpc_cmd = "/usr/local/bin/udhcpc";
    }
}

void _abcdk_ipcfg_node_free(abcdk_ipcfg_node_t **ctx)
{
    abcdk_ipcfg_node_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->old_resolv);

    if (ctx_p->dhcp_pid >= 0)
    {
        kill(ctx_p->dhcp_pid, 9);
        waitpid(ctx_p->dhcp_pid, NULL, 0);
    }

    abcdk_heap_free(ctx_p);
}

abcdk_ipcfg_node_t *_abcdk_ipcfg_node_alloc()
{
    abcdk_ipcfg_node_t *ctx = abcdk_heap_alloc(sizeof(abcdk_ipcfg_node_t));

    if (!ctx)
        return NULL;

    ctx->exitcode = 999;
    ctx->sigcode = 999;

    ctx->dhcp_pid = -1;

    ctx->implement_state = 0;
    ctx->link_state = -1;
    ctx->oper_state = -1;

    memset(ctx->addr_hcode, 0xff, 32);

    ctx->old_resolv = NULL;

    memset(ctx->resolv_hcode, 0xff, 32);

    ctx->start_link_count = 0;
    ctx->start_link_next = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,0);

    return ctx;
}

abcdk_option_t *_abcdk_ipcfg_load_conf(const char *conf)
{
    abcdk_option_t *args = abcdk_option_alloc("--");

    if (!args)
        return NULL;

    abcdk_getargs_file(args, conf, '\n', '#', conf);

    return args;
}

int _abcdk_ipcfg_compare_option(abcdk_option_t *a, abcdk_option_t *b)
{
    abcdk_object_t *buf = abcdk_object_alloc3(1024 * 1024, 2);
    ssize_t buflen[2] = {0, 0};
    int chk;

    if (!buf)
        return 0;

    buflen[0] = abcdk_getargs_snprintf(a, buf->pptrs[0], buf->sizes[0], "\n", "");
    buflen[1] = abcdk_getargs_snprintf(b, buf->pptrs[1], buf->sizes[1], "\n", "");

    if (buflen[0] > buflen[1])
        chk = 1;
    else if (buflen[0] < buflen[1])
        chk = -1;
    else
        chk = memcmp(buf->pptrs[0], buf->pptrs[1], buflen[1]);

    abcdk_object_unref(&buf);

    return chk;
}

void _abcdk_ipcfg_dump_option(abcdk_option_t *args, abcdk_logger_t *logger)
{
    abcdk_object_t *buf = abcdk_object_alloc3(1024 * 1024, 1);

    if (!buf)
        return;

    abcdk_getargs_snprintf(args, buf->pptrs[0], buf->sizes[0],"\n", "");
    abcdk_logger_printf(logger, LOG_INFO, "++++++\n%s\n------\n", buf->pstrs[0]);

    abcdk_object_unref(&buf);
}

int _abcdk_ipcfg_eth_hash_code_compare_cb(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque)
{
    if (node1->obj->sizes[0] > node2->obj->sizes[0])
        return 1;
    else if (node1->obj->sizes[0] < node2->obj->sizes[0])
        return -1;

    return memcmp(node1->obj->pptrs[0], node2->obj->pstrs[0], node2->obj->sizes[0]);
}

void _abcdk_ipcfg_eth_hash_code(char buf[33], const char *ifname)
{
    abcdk_tree_t *ip_vec = NULL, *p = NULL, *p2 = NULL;
    ;
    abcdk_ifaddrs_t ifaddr_vec[100] = {0};
    int ifaddr_count = 0;
    abcdk_md5_t *md5_ctx = NULL;
    int chk = -1;

    abcdk_tree_iterator_t it = {0, NULL, NULL, _abcdk_ipcfg_eth_hash_code_compare_cb};

    ip_vec = abcdk_tree_alloc3(1);
    if (!ip_vec)
        goto ERR;

    md5_ctx = abcdk_md5_create();
    if (!md5_ctx)
        goto ERR;

    ifaddr_count = abcdk_ifname_fetch(ifaddr_vec, 100, 1, 1);

    for (int i = 0; i < ifaddr_count; i++)
    {
        if (abcdk_strcmp(ifaddr_vec[i].name, ifname, 1))
            continue;

        if (ifaddr_vec[i].addr.family == AF_INET)
            p = abcdk_tree_alloc4(&ifaddr_vec[i].addr.addr4.sin_addr, 4);
        else if (ifaddr_vec[i].addr.family == AF_INET6)
            p = abcdk_tree_alloc4(&ifaddr_vec[i].addr.addr6.sin6_addr, 16);
        else
            continue;

        if (!p)
            goto END;

        abcdk_tree_insert2(ip_vec, p, 0);
    }

    abcdk_tree_sort(ip_vec, &it, 1);

END:

    p2 = abcdk_tree_child(ip_vec, 1);
    while (p2)
    {
        abcdk_md5_update(md5_ctx, p2->obj->pstrs[0], p2->obj->sizes[0]);

        p2 = abcdk_tree_sibling(p2, 0);
    }

    /*加入接口名字，使得不同的接口在具体相同配置情况下哈希值也不相同。*/
    abcdk_md5_update(md5_ctx, ifname, strlen(ifname));

    abcdk_md5_final2hex(md5_ctx, buf, 0);

ERR:

    abcdk_tree_free(&ip_vec);
    abcdk_md5_destroy(&md5_ctx);

    return;
}

abcdk_object_t *_abcdk_ipcfg_resolv_load()
{
    abcdk_object_t *buf = NULL;
    int chk;

    buf = abcdk_object_alloc2(1024 * 1024);

    if (!buf)
        return NULL;

    abcdk_load("/etc/resolv.conf", buf->pstrs[0], buf->sizes[0], 0);

    return buf;
}

void _abcdk_ipcfg_resolv_hash_code(char buf[33])
{
    abcdk_md5_t *md5_ctx = NULL;
    abcdk_object_t *data = NULL;

    md5_ctx = abcdk_md5_create();
    if (!md5_ctx)
        goto ERR;

    data = abcdk_mmap_filename("/etc/resolv.conf", 0, 0, 0, 0);
    if (!data)
        goto ERR;

    abcdk_md5_update(md5_ctx, data->pptrs[0], data->sizes[0]);

    abcdk_md5_final2hex(md5_ctx, buf, 0);

ERR:

    abcdk_object_unref(&data);
    abcdk_md5_destroy(&md5_ctx);

    return;
}

void _abcdk_ipcfg_start_link(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    int link_state, oper_state;

    link_state = (abcdk_net_get_link_state(ifname) <= 0 ? 0 : 1);
    oper_state = (abcdk_net_get_oper_state(ifname) <= 0 ? 0 : 1);

    if (!link_state || !oper_state)
    {
        /*未到时间不尝试启动。*/
        if (ctx->start_link_next >= abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0))
            return;

        abcdk_trace_output(LOG_INFO, "链路('%s')未活动或被关闭，尝试重新启动。", ifname);
        abcdk_net_up(ifname);

        /*下次启动间隔增加1秒，60次为一个周期。*/
        ctx->start_link_count += 1;
        ctx->start_link_count %= 60;
        ctx->start_link_next = (abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0) + ctx->start_link_count);

    }
    else
    {
        ctx->start_link_count = 0;
        ctx->start_link_next = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);
    }
}

void _abcdk_ipcfg_check_link(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    int prev_link_state, prev_oper_state;

    /*记录旧状态，并读取最新状态。*/
    prev_link_state = ctx->link_state;
    prev_oper_state = ctx->oper_state;
    ctx->link_state = (abcdk_net_get_link_state(ifname) <= 0 ? 0 : 1);
    ctx->oper_state = (abcdk_net_get_oper_state(ifname) <= 0 ? 0 : 1);

    /*检查链路状态是否发生变化，如果发生变化则重新应用配置。*/
    if (prev_link_state != ctx->link_state || prev_oper_state != ctx->oper_state)
    {
        abcdk_trace_output( LOG_INFO, "链路('%s')状态发生变化(link=%d,oper=%d)。", ifname, ctx->link_state, ctx->oper_state);

        if (ctx->link_state && ctx->oper_state)
            ctx->implement_state = 0;

        /*当链路断开后，清理地址和路由表。*/
        if (!ctx->link_state || !ctx->oper_state)
        {
            abcdk_net_address_flush(ifname);
            abcdk_net_route_flush(ifname);
        }
    }
}

void _abcdk_ipcfg_check_address(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    char curt_addr_hcode[33];

    /*链路不可用时，不需要检测地址变化。*/
    if (!ctx->link_state || !ctx->oper_state)
        return;

    /*读取当前配置。*/
    _abcdk_ipcfg_eth_hash_code(curt_addr_hcode, ifname);

    /*检查配置是否已经改变，如果未改变则不需要更新。*/
    if (abcdk_strcmp(curt_addr_hcode, ctx->addr_hcode, 0) == 0)
        return;

    abcdk_trace_output( LOG_INFO, "链路('%s')配置被改变，恢复应用配置。", ifname);

    ctx->implement_state = 0;
}

void _abcdk_ipcfg_check_resolv(abcdk_ipcfg_node_t *ctx)
{
    char curt_resolv_hcode[33];

    _abcdk_ipcfg_resolv_hash_code(curt_resolv_hcode);

    /*检查配置是否已经改变，如果未改变则不需要更新。*/
    if (abcdk_strcmp(curt_resolv_hcode, ctx->resolv_hcode, 0) == 0)
        return;

    abcdk_trace_output( LOG_INFO, "resolv配置被改变，恢复应用配置。");

    ctx->implement_state = 0;
}

int _abcdk_ipcfg_eth_up(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    return abcdk_net_up( ifname);
}

int _abcdk_ipcfg_eth_flush(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    return abcdk_net_address_flush( ifname);
}

int _abcdk_ipcfg_route_flush(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    return abcdk_net_route_flush( ifname);
}

int _abcdk_ipcfg_route_add(abcdk_ipcfg_node_t *ctx, int ver, const char *host, int prefix, const char *gw, int metric, const char *ifname)
{
    return abcdk_net_route_add( ver, host, prefix, gw, metric, ifname);
}

int _abcdk_ipcfg_eth_add(abcdk_ipcfg_node_t *ctx, int idx, int ver, const char *addr, const char *ifname)
{
    abcdk_object_t *addr_conf;
    char *ip = NULL, *gw = NULL;
    int prefix = 0, metric = 100;
    int chk;

    addr_conf = abcdk_strtok2vector(addr, ",");
    if (!addr_conf)
        return -1;

    if (addr_conf->numbers < 2)
    {
        abcdk_trace_output( LOG_ERR, "不符合四段参数(IP,前缀长度,网关,跃点)要求，其中网关和跃点为可选项。");

        chk = -22;
        goto END;
    }

    /*清除字符串两端“空白”字符。*/
    for (int i = 0; i < addr_conf->numbers; i++)
    {
        abcdk_strtrim(addr_conf->pstrs[i], isspace, 2);
    }

    ip = addr_conf->pstrs[0];
    prefix = atoi(addr_conf->pstrs[1]);

    if (addr_conf->numbers > 2)
        gw = addr_conf->pstrs[2];
    if (addr_conf->numbers > 3)
        metric = atoi(addr_conf->pstrs[3]);

    abcdk_net_address_add(ver, ip, prefix, gw, metric, ifname);

    if (gw != NULL && *gw != '\0')
    {
        /*第一个网关，设为默认路由。*/
        if (idx == 0)
        {
            chk = _abcdk_ipcfg_route_add(ctx, ver, (ver == 6 ? "0" : "0.0.0.0"), 0, gw, 0, ifname);
            if (chk != 0)
            {
                chk = -4;
                goto END;
            }
        }
    }

    // OK.
    chk = 0;

END:

    abcdk_object_unref(&addr_conf);

    return chk;
}

void _abcdk_ipcfg_eth_close_dhcp(abcdk_ipcfg_node_t *ctx)
{
    /*关闭DHCP客户端。*/
    if (ctx->dhcp_pid >= 0)
    {
        kill(ctx->dhcp_pid, 9);
        waitpid(ctx->dhcp_pid, NULL, 0);
        ctx->dhcp_pid = -1;
    }
}

int _abcdk_ipcfg_eth_start_dhclient(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    pid_t pid_chk;

    if (ctx->dhcp_pid < 0)
    {
        _abcdk_ipcfg_eth_flush(ctx, ifname);
        _abcdk_ipcfg_route_flush(ctx, ifname);

        ctx->dhcp_pid = abcdk_proc_popen(NULL, NULL, NULL, "%s -1 -q --no-pid %s", ctx->father->dhclient_cmd, ifname);
        if (ctx->dhcp_pid < 0)
            return -1;
    }

    /*非阻塞调用，如果存在但未结束，则返回0。*/
    pid_chk = abcdk_waitpid(ctx->dhcp_pid, WNOHANG, &ctx->exitcode, &ctx->sigcode);
    if (pid_chk != 0)
    {
        ctx->dhcp_pid = -1;

        if (ctx->exitcode != 0)
            return -2;
        else
            return 0;
    }

    /*重试。*/
    return -15;
}

int _abcdk_ipcfg_eth_start_udhcpc(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    pid_t pid_chk;

    if (ctx->dhcp_pid < 0)
    {
        _abcdk_ipcfg_eth_flush(ctx, ifname);
        _abcdk_ipcfg_route_flush(ctx, ifname);

        ctx->dhcp_pid = abcdk_proc_popen( NULL, NULL, NULL, "%s -n -q -f -i %s -p /tmp/udhcpc.%s.pid", ctx->father->udhcpc_cmd, ifname, ifname);
        if (ctx->dhcp_pid < 0)
            return -3;
    }

    /*非阻塞调用，如果存在但未结束，则返回0。*/
    pid_chk = abcdk_waitpid(ctx->dhcp_pid, WNOHANG, &ctx->exitcode, &ctx->sigcode);
    if (pid_chk != 0)
    {
        ctx->dhcp_pid = -1;

        if (ctx->exitcode != 0)
            return -4;
        else
            return 0;
    }

    /*重试。*/
    return -15;
}

int _abcdk_ipcfg_eth_start_dhcp(abcdk_ipcfg_node_t *ctx, const char *ifname)
{
    int chk;

    if (ctx->father->udhcpc_cmd)
    {
        chk = _abcdk_ipcfg_eth_start_udhcpc(ctx, ifname);
    }
    else if (ctx->father->dhclient_cmd)
    {
        chk = _abcdk_ipcfg_eth_start_dhclient(ctx, ifname);
    }
    else
    {
        abcdk_trace_output( LOG_ERR, "未支持的DHCP客户端。");
        return -5;
    }

    if (chk == 0)
        abcdk_trace_output( LOG_INFO, "为'%s'申请动态地址完成。", ifname);
    else if (chk == -15)
        abcdk_trace_output( LOG_INFO, "正在为'%s'申请动态地址……", ifname);
    else
        abcdk_trace_output( LOG_ERR, "为'%s'申请动态地址失败(exit=%d,signal=%d)。", ifname, ctx->exitcode, ctx->sigcode);

    return chk;
}

int _abcdk_ipcfg_resolv_change(abcdk_ipcfg_node_t *ctx, const char *const *nss)
{
    int blen = 1024 * 1024;
    char *buf = NULL;
    int len = 0;
    int chk;

    buf = (char *)abcdk_heap_alloc(blen);
    if (!buf)
        return -1;

    if (ctx->old_resolv)
        len += snprintf(buf + len, blen - len, "%s\n", ctx->old_resolv->pstrs[0]);

    len += snprintf(buf + len, blen - len, "\n#tool-ipconfig-begin");

    for (int i = 0; nss[i]; i++)
    {
        len += snprintf(buf + len, blen - len, "\nnameserver %s", nss[i]);
    }

    len += snprintf(buf + len, blen - len, "\n#tool-ipconfig-end\n");

    truncate("/etc/resolv.conf", 0);
    chk = abcdk_save("/etc/resolv.conf", buf, len, 0);
    if (chk != len)
    {
        abcdk_trace_output( LOG_ERR, "更新'resolv'失败。");
        goto ERR;
    }

    abcdk_trace_output( LOG_INFO, "更新'resolv'完成。");

    abcdk_heap_freep((void **)&buf);
    return 0;

ERR:

    abcdk_heap_freep((void **)&buf);
    return -2;
}

void _abcdk_ipcfg_implement(abcdk_ipcfg_node_t *ctx, abcdk_option_t *opt)
{
    const char *enable, *ifname, *tmp;
    int addr4_count, addr6_count, nss_count;
    char *nss[100] = {0};
    int chk;

    enable = abcdk_option_get(opt, "--enable", 0, "");
    ifname = abcdk_option_get(opt, "--ifname", 0, "");

    if (abcdk_strcmp(enable, "static", 0) == 0)
    {
        _abcdk_ipcfg_start_link(ctx, ifname);
        _abcdk_ipcfg_check_link(ctx, ifname);
        _abcdk_ipcfg_check_address(ctx, ifname);

        /*
         * 1:如果链路未连接或未启动，则不需要应用配置。
         * 2:如果已经应用成功，则不需要重复配置。
         */
        if (!ctx->link_state || !ctx->oper_state || ctx->implement_state)
            return;

        addr4_count = abcdk_option_count(opt, "--address4");
        addr6_count = abcdk_option_count(opt, "--address6");

        /*没有新配置的情况下，保持现有配置不变。*/
        if (addr4_count <= 0 && addr6_count <= 0)
            goto ERROR;

        /*关闭可能存在的DHCP客户端。*/
        _abcdk_ipcfg_eth_close_dhcp(ctx);

        chk = _abcdk_ipcfg_eth_up(ctx, ifname);
        if (chk != 0)
            goto ERROR;

        chk = _abcdk_ipcfg_eth_flush(ctx, ifname);
        if (chk != 0)
            goto ERROR;

        chk = _abcdk_ipcfg_route_flush(ctx, ifname);
        if (chk != 0)
            goto ERROR;

        for (int i = 0; i < addr4_count; i++)
        {
            tmp = abcdk_option_get(opt, "--address4", i, "");
            chk = _abcdk_ipcfg_eth_add(ctx, i, 4, tmp, ifname);
            if (chk != 0)
                goto ERROR;
        }

        for (int i = 0; i < addr6_count; i++)
        {
            tmp = abcdk_option_get(opt, "--address6", i, "");
            chk = _abcdk_ipcfg_eth_add(ctx, i, 6, tmp, ifname);
            if (chk != 0)
                goto ERROR;
        }

        /*记录最新配置。*/
        _abcdk_ipcfg_eth_hash_code(ctx->addr_hcode, ifname);
    }
    else if (abcdk_strcmp(enable, "dhcp", 0) == 0)
    {
        _abcdk_ipcfg_start_link(ctx, ifname);
        _abcdk_ipcfg_check_link(ctx, ifname);
        _abcdk_ipcfg_check_address(ctx, ifname);

        /*
         * 1:如果链路未连接或未启动，则不需要应用配置。
         * 2:如果已经应用成功，则不需要重复配置。
         */
        if (!ctx->link_state || !ctx->oper_state || ctx->implement_state)
            return;

        chk = _abcdk_ipcfg_eth_start_dhcp(ctx, ifname);
        if (chk != 0)
            goto ERROR;

        /*记录最新配置。*/
        _abcdk_ipcfg_eth_hash_code(ctx->addr_hcode, ifname);
    }
    else if (abcdk_strcmp(enable, "resolv", 0) == 0)
    {
        /*旧的配置只加载一次。*/
        if (!ctx->old_resolv)
            ctx->old_resolv = _abcdk_ipcfg_resolv_load();

        _abcdk_ipcfg_check_resolv(ctx);

        /*
         * 1:如果已经应用成功，则不需要重复配置。
         */
        if (ctx->implement_state)
            return;

        nss_count = abcdk_option_count(opt, "--nameserver");
        nss_count = ABCDK_CLAMP(nss_count, 0, ABCDK_ARRAY_SIZE(nss) - 1);
        for (int i = 0; i < nss_count; i++)
        {
            nss[i] = (char *)abcdk_option_get(opt, "--nameserver", i, NULL);
        }

        chk = _abcdk_ipcfg_resolv_change(ctx, (const char *const *)nss);
        if (chk != 0)
            goto ERROR;

        /*记录最新配置。*/
        _abcdk_ipcfg_resolv_hash_code(ctx->resolv_hcode);
    }
    else
    {
        abcdk_trace_output( LOG_INFO, "未支持的使能方案，跳过。");
    }

    ctx->implement_state = 1;
    return;

ERROR:

    ctx->implement_state = 0;
    return;
}

void _abcdk_ipcfg_work_real(abcdk_ipcfg_t *ctx, uint32_t idx)
{
    abcdk_ipcfg_node_t *node_ctx = _abcdk_ipcfg_node_alloc();
    abcdk_option_t *conf_a = NULL, *conf_b = NULL, *conf_prev = NULL;
    int chk;

    abcdk_thread_setname("ipconfig-%d", idx);

    node_ctx->father = ctx;

    const char *conf_file = abcdk_option_get(node_ctx->father->args, "--conf", idx, "");

WATCHDOG:

    /*1秒间隔，太频繁无用。*/
    sleep(1);

    abcdk_option_free(&conf_a);
    conf_a = _abcdk_ipcfg_load_conf(conf_file);
    if (!conf_a)
        goto END;

    /*3秒间隔，太频繁无用。*/
    sleep(3);

    abcdk_option_free(&conf_b);
    conf_b = _abcdk_ipcfg_load_conf(conf_file);
    if (!conf_b)
        goto END;

    /*判断文件是否正在被修改。*/
    chk = _abcdk_ipcfg_compare_option(conf_a, conf_b);
    if (chk != 0)
        goto END;

    /*检查数据是否发生过变更。*/
    if (conf_prev)
    {
        chk = _abcdk_ipcfg_compare_option(conf_b, conf_prev);
        if (chk == 0)
            goto END;

        /*释放旧的配置。*/
        abcdk_option_free(&conf_prev);
        /*复制新配置。*/
        conf_prev = conf_b;
        conf_b = NULL;
    }
    else
    {
        /*可能是第一次运行，直接复制。*/
        conf_prev = conf_b;
        conf_b = NULL;
    }

    /*重置出错码和信号量。*/
    node_ctx->exitcode = 0;
    node_ctx->sigcode = 0;

    /*标记为未应用。*/
    node_ctx->implement_state = 0;

    _abcdk_ipcfg_dump_option(conf_prev, node_ctx->father->logger);
    abcdk_trace_output( LOG_INFO, "配置文件('%s')已经更新，在应用新配置过程中网络连接可能发生抖动。", conf_file);

END:

    if (conf_prev)
        _abcdk_ipcfg_implement(node_ctx, conf_prev);

    if (abcdk_atomic_compare(&node_ctx->father->exitflag, 0))
        goto WATCHDOG;

    abcdk_option_free(&conf_a);
    abcdk_option_free(&conf_b);
    abcdk_option_free(&conf_prev);
    _abcdk_ipcfg_node_free(&node_ctx);
}

void _abcdk_ipcfg_work(void *opaque,uint64_t event,void *item)
{
    abcdk_ipcfg_t *ctx = (abcdk_ipcfg_t *)opaque;

    _abcdk_ipcfg_work_real(ctx, event);
}

void _abcdk_ipcfg_process(abcdk_ipcfg_t *ctx)
{
    abcdk_worker_config_t worker_cfg = {0,ctx,_abcdk_ipcfg_work};
    abcdk_worker_t *worker_ctx = NULL;
    const char *log_path = NULL;
    int conf_num;

    ctx->dhclient_cmd = abcdk_option_get(ctx->args, "--dhclient-cmd", 0, NULL);
    ctx->udhcpc_cmd = abcdk_option_get(ctx->args, "--udhcpc-cmd", 0, NULL);
    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    conf_num = abcdk_option_count(ctx->args, "--conf");

    _abcdk_ipcfg_find_dhcp_client(ctx);

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path,"ipconfig.log", "ipconfig.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace,ctx->logger);

    abcdk_trace_output( LOG_INFO, "启动……");

    if (conf_num <= 0)
    {
        abcdk_trace_output( LOG_ERR, "至少指定一个配置文件。");
        goto ERR;
    }

    /*创建足够数量的工人。*/
    worker_cfg.numbers = conf_num;
    worker_ctx = abcdk_worker_start(&worker_cfg);

    /*每个配置文件将由一个工人处理。*/
    for (int i = 0; i < conf_num; i++)
        abcdk_worker_dispatch(worker_ctx, i, NULL);

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

    /*通知所有线程退出。*/
    abcdk_atomic_store(&ctx->exitflag, 1);

    /*销毁工人。*/
    abcdk_worker_stop(&worker_ctx);

ERR:

    abcdk_trace_output( LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}

int _abcdk_ipcfg_daemon_process_cb(void *opaque)
{
    abcdk_ipcfg_t *ctx = (abcdk_ipcfg_t *)opaque;

    _abcdk_ipcfg_process(ctx);

    return 0;
}

void _abcdk_ipcfg_daemon(abcdk_ipcfg_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path,"ipconfig-daemon.log", "ipconfig-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace,logger);

    abcdk_proc_daemon(interval, _abcdk_ipcfg_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_ipconfig(abcdk_option_t *args)
{
    abcdk_ipcfg_t ctx = {0};
    const char *pid_file = NULL;
    int other_pid = -1, self_pid = -1;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_ipcfg_print_usage(ctx.args);
    }
    else
    {
        pid_file = abcdk_option_get(ctx.args,"--pid-file",0,"/tmp/abcdk/pid/ipconfig.pid");

        /*单实例运行。*/
        self_pid = abcdk_proc_singleton(pid_file, &other_pid);
        if (self_pid < 0)
        {
            fprintf(stderr, "已经有实例(PID=%d)正在运行。\n", other_pid);
            return 1;
        }

        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdk_ipcfg_daemon(&ctx);
        }
        else
        {
            _abcdk_ipcfg_process(&ctx);
        }
    }

    return 0;
}