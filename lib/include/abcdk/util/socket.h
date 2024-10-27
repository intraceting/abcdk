/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_SOCKET_H
#define ABCDK_UTIL_SOCKET_H

#include "abcdk/util/general.h"
#include "abcdk/util/path.h"
#include "abcdk/util/io.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/bloom.h"

__BEGIN_DECLS

/***/
#ifndef SUN_LEN
#define SUN_LEN(ptr) ((size_t)(((struct sockaddr_un *)0)->sun_path) + strlen((ptr)->sun_path))
#endif

/** Socket地址 */
typedef union _abcdk_sockaddr
{
    /** 预留空间。*/
    uint8_t padding[255];

    /** 协议。*/
    sa_family_t family;

    /** 通用的地址。*/
    struct sockaddr addr;

    /** UNIX地址。*/
    struct sockaddr_un addr_un;

    /** IPv4地址。*/
    struct sockaddr_in addr4;

    /** IPv6地址。*/
    struct sockaddr_in6 addr6;

} abcdk_sockaddr_t;


/*
 * 网卡接口地址。
 */
typedef struct _abcdk_ifaddrs
{
    /**
     * 接口名称
    */
    char name[IF_NAMESIZE];

    /** 接口地址。*/
    abcdk_sockaddr_t addr;

    /** 掩码地址。*/
    abcdk_sockaddr_t mark;

    /** 
     * 广播地址。 
     * 
     * @note IPv6无效。
    */
    abcdk_sockaddr_t broa;
} abcdk_ifaddrs_t;

/** Socket多播地址。*/
typedef union _abcdk_mreq
{
    /** IPv4地址。*/
    struct ip_mreq st_mreq4;

    /** IPv6地址。*/
    struct ipv6_mreq st_mreq6;
} abcdk_mreq_t;

/**
 * SOCKADDR复制。
 */
void abcdk_sockaddr_copy(const abcdk_sockaddr_t *src,abcdk_sockaddr_t *dst);

/**
 * 域名解析。
 * 
 * @param addrs IP地址数组的指针。
 * @param max IP地址数组元素最大数量。
 * @param canonname 规范名称，NULL(0) 忽略。
 *
 * @return >= 0 IP地址数量， < 0 出错。
*/
int abcdk_gethostbyname(const char *name, sa_family_t family, abcdk_sockaddr_t *addrs, int max, char canonname[1000]);

/**
 * IP字符串转IP地址。
 *
 * @return 0 成功，-1 失败。
*/
int abcdk_inet_pton(const char *name, sa_family_t family, abcdk_sockaddr_t *addr);

/**
 * IP地址转IP字符串。
 *
 * @return !NULL(0) IP字符串指针，NULL(0) 失败。
*/
char *abcdk_inet_ntop(const abcdk_sockaddr_t *addr, char *name, size_t max);

/**
 * 获取网络接口信息
 * 
 * @param ex_loopback 0 包括回环接口，!0 排除回环接口。
 * @param ex_virtual 0 包括虚拟接口，!0 排除虚拟接口。
 * 
 * @return >= 0 网络接口数量， < 0 出错。
*/
int abcdk_ifname_fetch(abcdk_ifaddrs_t *addrs, int max, int ex_loopback,int ex_virtual);

/**
 * SOCKET IO control
 * 
 * @return !-1 成功，-1 失败。
*/
int abcdk_socket_ioctl(uint32_t cmd, void *args);

/**
 * 查询网卡地址。
 * 
 * 格式化为十六进制字符串。AABBCCDDEEFF
 * 
 * @return !NULL(0) 成功(网卡地址字符串的指针)，NULL(0) 失败。
 * 
*/
char *abcdk_mac_fetch(const char *ifname, char addr[12]);

/**
 * 查询网卡连接状态。
 * 
 * @param [out] flag 状态。
 * 
 * @return 0 成功，-1 失败。
 * 
*/
int abcdk_netlink_fetch(const char *ifname, int *flag);

/**
 * 获取或设置SOCKET选项。
 * 
 * @param direction 方向。 1 读，2 写。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_socket_option(int fd, int level, int name, void *data, int *len, int direction);

/**
 * 获取或设置SOCKET选项(integer)。
 * 
 * @param direction 方向。 1 读，2 写。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sockopt_option_int(int fd, int level, int name, int *flag, int direction);

/**
 * 获取或设置SOCKET选项(timeout)。
 * 
 * @param direction 方向。 1 读，2 写。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sockopt_option_timeout(int fd, int name, struct timeval *tv, int direction);

/**
 * 设置SOCKET选项(timeout)。
 * 
 * @param tv 时长(微秒)。> 0 有效，<= 0 关闭。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sockopt_option_timeout_set(int fd, int name, time_t tv);

/**
 * 获取或设置SOCKET选项(linger)。
 * 
 * @param direction 方向。 1 读，2 写。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_socket_option_linger(int fd, struct linger *lg, int direction);

/**
 * 设置SOCKET选项(linger)。
 *  
 * @return 0 成功，-1 失败。
*/
int abcdk_socket_option_linger_set(int fd, int l_onoff, int l_linger);

/**
 * 启用或禁用SOCKET组播选项(multicast)。
 * 
 * @param ifaddr 地址的指针，为NULL(0)使用默认值。IPv4 点分十进制字符串地址；IPv6 网络接口名称。
 * @param enable 开关。!0 启用，0 禁用。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_socket_option_multicast(int fd,abcdk_sockaddr_t *multiaddr, const char *ifaddr,int enable);

/**
 * TCP快速确认开关。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_socket_option_tcp_quickack(int fd,int enable);

/**
 * 创建一个SOCKET句柄。
 * 
 * @return >= 0 成功(SOCKET句柄)，-1 失败。
*/
int abcdk_socket(sa_family_t family, int udp);

/**
 * 绑定地址到SOCKET句柄。
 *
 * @return 0 成功，!0 失败。
*/
int abcdk_bind(int fd, const abcdk_sockaddr_t *addr);

/**
 * 接收一个已经连接的SOCKET句柄。
 * 
 * @param addr 远程地址的指针，NULL(0) 忽略。
 * 
 * @return >= 0 成功(SOCKET句柄)，-1 失败。
*/
int abcdk_accept(int fd, abcdk_sockaddr_t *addr);

/**
 * 使用已经打开的SOCKET句柄创建到远程地址的连接。
 * 
 * @param timeout 超时(毫秒)。>= 0 连接成功或时间过期，< 0 直到连接成功或出错。
 * 
 * @return 0 成功，-1 失败(或超时)。
 * 
*/
int abcdk_connect(int fd, abcdk_sockaddr_t *addr, time_t timeout);

/**
 * 字符地址转SOCKET地址。
 * 
 * @code
 * Unix: /File 
 * Unix: /Path/File
 * Unix: unix:///Path/File 
 * IPv4：Address:Port 
 * IPv4：ipv4://Address:Port 
 * IPv6：Address,Port
 * IPv6：ipv6://Address,Port
 * IPv6：[Address]:Port
 * IPv6：ipv6://[Address]:Port
 * @endcode
 * 
 * @param [in out] dst 地址。可以指定地址家族。
 * @param [in] try_lookup 尝式域名解析。!0 是，0 否。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_sockaddr_from_string(abcdk_sockaddr_t *dst,const char *src, int try_lookup);

/**
 * SOCKET地址转字符地址。
 * 
 * Unix: /File
 * Unix: /Path/File
 * IPv4：*.*.*.* 
 * IPv6：*:*::*
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
char *abcdk_sockaddr_to_string(char dst[NAME_MAX],const abcdk_sockaddr_t *src,int ex_port);

/**
 * 判断SOCKET地址位置。
 * 
 * @param where 1 是否在本地，2 是否在远程。
 * 
 * @return !0 是，0 否。
*/
int abcdk_sockaddr_where(const abcdk_sockaddr_t *test,int where);

/**
 * 比较SOCKET地址。
 * 
 * @param [in] care_port 是否比较端口。!0 是，0 否。
 *
 * @return 0 相同，!0 不同。
 */
int abcdk_sockaddr_compare(const abcdk_sockaddr_t *addr1, const abcdk_sockaddr_t *addr2,int care_port);

/**
 * 生成网络掩码。
*/
void abcdk_sockaddr_make_netmask(abcdk_sockaddr_t *mask, const abcdk_sockaddr_t *addr, int prefix);

/**
 * 生成网络掩码。
*/
char *abcdk_sockaddr_make_netmask2(char buf[100], sa_family_t family, const char *host, int prefix);

__END_DECLS

#endif //ABCDK_UTIL_SOCKET_H