/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/shell/net.h"

int abcdk_net_get_link_state(const char *ifname)
{
    char carrier_name[NAME_MAX] = {0};
    char buf[2] = {0};
    int chk;

    assert(ifname != NULL && *ifname != '\0');

    snprintf(carrier_name,NAME_MAX,"/sys/class/net/%s/carrier",ifname);

    chk = abcdk_load(carrier_name,buf,2,0);
    if(chk <= 0)
        return -1;
    
    return strtol(buf,NULL,0);
}

int abcdk_net_get_oper_state(const char *ifname)
{
    char oper_name[NAME_MAX] = {0};
    char buf[5] = {0};
    int chk;

    assert(ifname != NULL && *ifname != '\0');

    snprintf(oper_name,NAME_MAX,"/sys/class/net/%s/operstate",ifname);

    chk = abcdk_load(oper_name,buf,5,0);
    if(chk <= 0)
        return -1;

    abcdk_strtrim(buf, isspace, 2);

    if(abcdk_strcmp("up",buf,0) == 0)
        return 1;
    else if(abcdk_strcmp("down",buf,0) == 0)
        return 0;

    /*其它*/
    return -2;
}

int abcdk_net_down(const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;

    assert(ifname != NULL && *ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL, "ip link set %s down", ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "停用IFN(%s)失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);

        return -2;
    }

    abcdk_trace_output(LOG_INFO, "停用IFN(%s)完成。", ifname);

    return 0;
}

int abcdk_net_up(const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;

    assert(ifname != NULL && *ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL, "ip link set %s up", ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "启用IFN(%s)失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);

        return -2;
    }

    abcdk_trace_output( LOG_INFO, "启用IFN(%s)完成。", ifname);

    return 0;
}

int abcdk_net_address_flush(const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;

    assert(ifname != NULL && *ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL, "ip address flush dev %s", ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output(LOG_ERR, "清除IFN(%s)配置失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output(LOG_INFO, "清除IFN(%s)配置完成。", ifname);

    return 0;
}

int abcdk_net_route_flush(const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;

    assert(ifname != NULL && *ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL,"ip route flush dev %s", ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "清除IFN(%s)路由配置失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output( LOG_INFO, "清除IFN(%s)路由配置完成。", ifname);

    return 0;
}

int abcdk_net_route_add(int ver, const char *host, int prefix, const char *gw, int metric, const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    char net[100] = {0};
    pid_t pid = -1;
    int chk;

    assert((ver == 4 || ver == 6) && host != NULL && prefix >= 0 && gw != NULL && metric >= 0 && ifname != NULL);
    assert(*host != '\0' &&  *gw != '\0' && *ifname != '\0');

    abcdk_sockaddr_make_segment2(net, ((ver == 6) ? AF_INET6 : AF_INET), host, prefix);

    pid = abcdk_proc_popen(NULL, NULL, NULL, "ip -%d route add %s/%d via %s metric %d dev %s", ver, net, prefix, gw, metric, ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0 && exitcode != 2)
    {
        abcdk_trace_output( LOG_ERR, "添加IPLAN('IPV%d','%s/%d','%s','%d')到IFN(%s)失败(exit=%d,signal=%d)。",
                                ver, net, prefix, gw, metric, ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output( LOG_INFO, "添加IPLAN('IPV%d','%s/%d','%s','%d')到IFN(%s)完成。",
                            ver, net, prefix, gw, metric, ifname);

    return 0;
}

int abcdk_net_address_add(int ver, const char *host, int prefix, const char *gw, int metric, const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;
    int chk;

    assert((ver == 4 || ver == 6) && host != NULL && prefix > 0 && ifname != NULL);
    assert(*host != '\0' && *ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL, "ip -%d address add %s/%d dev %s", ver, host, prefix, ifname);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "添加IPADDR('IPV%d','%s/%d')到IFN(%s)失败(exit=%d,signal=%d)。",
                                ver, host, prefix, ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output( LOG_INFO, "添加IPADDR('IPV%d','%s/%d')到IFN(%s)完成。",
                            ver, host, prefix, ifname);

    /*可能没有网关。*/
    if (gw != NULL && *gw != '\0')
    {
        metric = ABCDK_CLAMP(metric,0,999);

        chk = abcdk_net_route_add(ver,host,prefix,gw,metric,ifname);
        if(chk != 0)
            return -3;
    }

    return 0;
}

int abcdk_net_set_mtu(uint16_t mtu, const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;
    int chk;

    assert(mtu >= 1400 && ifname != NULL);
    assert(*ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL,"ip link set %s mtu %hu", ifname,mtu);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "更新IFN(%s)最大传输单元失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output( LOG_INFO, "更新IFN(%s)最大传输单元完成。", ifname);

    return 0;
}

int abcdk_net_set_txqueuelen(uint16_t len,const char *ifname)
{
    int exitcode = 0, sigcode = 0;
    pid_t pid = -1;
    int chk;

    assert(len >= 500 && ifname != NULL);
    assert(*ifname != '\0');

    pid = abcdk_proc_popen(NULL, NULL, NULL,"ip link set %s txqueuelen %hu", ifname,len);
    if (pid < 0)
        return -1;

    abcdk_waitpid(pid, 0, &exitcode, &sigcode);
    if (exitcode != 0)
    {
        abcdk_trace_output( LOG_ERR, "更新IFN(%s)队列长度失败(exit=%d,signal=%d)。", ifname, exitcode, sigcode);
        return -2;
    }

    abcdk_trace_output( LOG_INFO, "更新IFN(%s)队列长度完成。", ifname);

    return 0; 
}