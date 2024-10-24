/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/socket.h"

void abcdk_sockaddr_copy(const abcdk_sockaddr_t *src,abcdk_sockaddr_t *dst)
{
    assert(src != NULL && dst != NULL);

    if(src->family == AF_UNIX)
    {
        dst->family = src->family;
        strcpy(dst->addr_un.sun_path,src->addr_un.sun_path);
    }
    else
    {
        *dst = *src;
    }
}

int abcdk_gethostbyname(const char *name, sa_family_t family, abcdk_sockaddr_t *addrs, int max, char canonname[1000])
{
    struct addrinfo *results = NULL;
    struct addrinfo *it = NULL;
    struct addrinfo hint = {0};
    int chk;
    int count = 0;

    assert(name != NULL && (family == AF_UNSPEC || family == AF_INET || family == AF_INET6) && addrs != NULL && max > 0);

    hint.ai_flags = AI_ADDRCONFIG | AI_CANONNAME;
    hint.ai_family = family;

    chk = getaddrinfo(name, NULL, &hint, &results);
    if (chk != 0 || results == NULL)
        return -1;

    for (it = results; it != NULL && count < max; it = it->ai_next)
    {
        if (it->ai_socktype != SOCK_STREAM || it->ai_protocol != IPPROTO_TCP)
            continue;

        if (it->ai_addr->sa_family != AF_INET && it->ai_addr->sa_family != AF_INET6)
            continue;

        memcpy(&addrs[count++], it->ai_addr, it->ai_addrlen);
    }

    if (canonname && results->ai_canonname)
        strncpy(canonname,results->ai_canonname, 1000);

    freeaddrinfo(results);

    return count;
}

int abcdk_inet_pton(const char *name, sa_family_t family, abcdk_sockaddr_t *addr)
{
    int chk = -1;

    assert(name != NULL && (family == AF_INET || family == AF_INET6) && addr != NULL);

    if (family == AF_INET)
        chk = (inet_pton(family, name, &addr->addr4.sin_addr) == 1 ? 0 : -1);
    if (family == AF_INET6)
        chk = (inet_pton(family, name, &addr->addr6.sin6_addr) == 1 ? 0 : -1);

    if(chk == 0)
        addr->family = family;

    return chk;
}

char *abcdk_inet_ntop(const abcdk_sockaddr_t *addr, char *name, size_t max)
{
    assert(addr != NULL && name != NULL && max > 0);
    
    assert(addr->family == AF_INET || addr->family == AF_INET6);
    assert((addr->family == AF_INET) ? (max >= INET_ADDRSTRLEN) : 1);
    assert((addr->family == AF_INET6) ? (max >= INET6_ADDRSTRLEN) : 1);

    if (addr->family == AF_INET)
        return (char *)inet_ntop(addr->family, &addr->addr4.sin_addr, name, max);
    if (addr->family == AF_INET6)
        return (char *)inet_ntop(addr->family, &addr->addr6.sin6_addr, name, max);

    return NULL;
}

int abcdk_ifname_fetch(abcdk_ifaddrs_t *addrs, int max, int ex_loopback,int ex_virtual)
{
    struct ifaddrs *results = NULL;
    struct ifaddrs *it = NULL;
    abcdk_ifaddrs_t *p = NULL;
    char tmp[255] = {0};
    int chk;
    int count = 0;

    assert(addrs != NULL && max > 0);

    chk = getifaddrs(&results);
    if (chk != 0 || results == NULL)
        return -1;

    for (it = results; it != NULL && count < max; it = it->ifa_next)
    {
        if (it->ifa_addr == NULL)
            continue;

        if (it->ifa_addr->sa_family != AF_INET && it->ifa_addr->sa_family != AF_INET6)
            continue;

        if (ex_loopback)
        {
            /*跳过回环接口。*/
            if ((it->ifa_flags & IFF_LOOPBACK))
                continue;
        }

        if (ex_virtual)
        {
            /*虚拟接口会在这个目录存在相同名字的目录。*/
            memset(tmp, 0, sizeof(tmp));
            abcdk_dirdir(tmp, "/sys/devices/virtual/net/");
            abcdk_dirdir(tmp, it->ifa_name);

            /*跳过虚拟接口。*/
            if (access(tmp, F_OK) == 0)
                continue;
        }

        p = &addrs[count++];

        strncpy(p->name, it->ifa_name, IFNAMSIZ);

        if (AF_INET == it->ifa_addr->sa_family)
        {
            memcpy(&p->addr, it->ifa_addr, sizeof(struct sockaddr_in));
            memcpy(&p->mark, it->ifa_netmask, sizeof(struct sockaddr_in));

            if (it->ifa_flags & IFF_BROADCAST)
                memcpy(&p->broa, it->ifa_broadaddr, sizeof(struct sockaddr_in));
            else
                p->broa.family = PF_UNSPEC;
        }
        else if (AF_INET6 == it->ifa_addr->sa_family)
        {
            memcpy(&p->addr, it->ifa_addr, sizeof(struct sockaddr_in6));
            memcpy(&p->mark, it->ifa_netmask, sizeof(struct sockaddr_in6));

            /*IPv6 not support. */
            p->broa.family = PF_UNSPEC;
        }
    }

    freeifaddrs(results);

    return count;
}

int abcdk_socket_ioctl(uint32_t cmd, void *args)
{
    int sock = -1;
    int chk;

    assert(cmd != 0 && args != NULL);

    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock == -1)
        return -1;

    chk = ioctl(sock, cmd, args);

final:

    abcdk_closep(&sock);

    return chk;
}

char *abcdk_mac_fetch(const char *ifname, char addr[12])
{
    struct ifreq args;
    int chk;

    assert(ifname != NULL && addr != NULL);

    memset(&args,0,sizeof(args));
    strncpy(args.ifr_ifrn.ifrn_name, ifname, IFNAMSIZ);

    chk = abcdk_socket_ioctl(SIOCGIFHWADDR, &args);
    if (chk == -1)
        return NULL;

    for (int i = 0; i < 6; i++)
        sprintf(addr + 2 * i, "%02X", (uint8_t)args.ifr_hwaddr.sa_data[i]);

    return addr;
}

int abcdk_netlink_fetch(const char *ifname, int *flag)
{
    struct ifreq args;
    int chk;

    assert(ifname != NULL && flag != NULL);

    memset(&args,0,sizeof(args));
    strncpy(args.ifr_ifrn.ifrn_name, ifname, IFNAMSIZ);

    chk = abcdk_socket_ioctl(SIOCGIFFLAGS, &args);
    if (chk == -1)
        return -1;

    *flag = args.ifr_flags;

    return 0;
}

int abcdk_socket_option(int fd,int level, int name,void *data,int *len,int direction)
{
    assert(fd >= 0 && level >= 0 && name > 0 && data != NULL && len != NULL && (direction == 1 || direction == 2));

    if(direction == 1)
        return getsockopt(fd,level,name,data,len);
    
    return setsockopt(fd,level,name,data,*len);
}

int abcdk_sockopt_option_int(int fd,int level, int name,int *flag,int direction)
{
    socklen_t len = sizeof(int);

    assert(fd >= 0 && level >= 0 && name > 0 && flag != NULL && (direction == 1 || direction == 2));

    return abcdk_socket_option(fd,level,name,flag,&len,direction);
}

int abcdk_sockopt_option_timeout(int fd,int name, struct timeval *tv,int direction)
{
    socklen_t len = sizeof(struct timeval);

    assert(fd >= 0 && name > 0 && tv != NULL && (direction == 1 || direction == 2));

    return abcdk_socket_option(fd,SOL_SOCKET,name,tv,&len,direction);
}

int abcdk_sockopt_option_timeout_set(int fd, int name, time_t tv)
{
    struct timeval tmp = {0, 0};

    assert(fd >= 0 && name > 0);

    if (tv > 0)
    {
        tmp.tv_sec = tv / 1000000;
        tmp.tv_usec = tv % 1000000;
    }

    return abcdk_sockopt_option_timeout(fd, name, &tmp, 2);
}

int abcdk_socket_option_linger(int fd,struct linger *lg,int direction)
{
    socklen_t len = sizeof(struct linger);

    assert(fd >= 0 && lg != NULL && (direction == 1 || direction == 2));

    return abcdk_socket_option(fd,SOL_SOCKET,SO_LINGER,lg,&len,direction);  
}

int abcdk_socket_option_linger_set(int fd, int l_onoff, int l_linger)
{
    struct linger l;

    assert(fd >= 0);

    l.l_onoff = l_onoff;
    l.l_linger = l_linger;

    return abcdk_socket_option_linger(fd,&l,2);
}

int abcdk_socket_option_multicast(int fd,abcdk_sockaddr_t *multiaddr, const char *ifaddr,int enable)
{
    socklen_t len = sizeof(struct ip_mreq);
    socklen_t len6 = sizeof(struct ipv6_mreq);
    struct ip_mreq st_mreq;
    struct ipv6_mreq st_mreq6;
    int name;
    int chk = -1;

    assert(fd >= 0 && multiaddr != NULL);
    assert(multiaddr->family == AF_INET || multiaddr->family == AF_INET6);

    memset(&st_mreq,0,sizeof(st_mreq));
    memset(&st_mreq6,0,sizeof(st_mreq6));

    if(multiaddr->family == AF_INET)
    {
        st_mreq.imr_multiaddr = multiaddr->addr4.sin_addr;
        st_mreq.imr_interface.s_addr = (ifaddr ? inet_addr(ifaddr) : INADDR_ANY);

        name = (enable ? IP_ADD_MEMBERSHIP : IP_DROP_MEMBERSHIP);

        chk = abcdk_socket_option(fd,IPPROTO_IP,name,&st_mreq,&len,2);  
    }
    else if(multiaddr->family == AF_INET6)
    {
        st_mreq6.ipv6mr_multiaddr = multiaddr->addr6.sin6_addr;
        st_mreq6.ipv6mr_interface = (ifaddr ? if_nametoindex(ifaddr) : 0);

        name = (enable ? IPV6_JOIN_GROUP : IPV6_LEAVE_GROUP);

        chk = abcdk_socket_option(fd,IPPROTO_IP,name,&st_mreq6,&len,2); 
    }

    return chk;
}

int abcdk_socket_option_tcp_quickack(int fd,int enable)
{
    int flag = (enable?1:0);

    assert(fd >= 0);

    return abcdk_sockopt_option_int(fd, IPPROTO_TCP, TCP_QUICKACK, &flag, 2);
}

int abcdk_socket(sa_family_t family, int udp)
{
    int type = SOCK_CLOEXEC;

    type |= (udp ? SOCK_DGRAM : SOCK_STREAM);

    return socket(family, type, 0);
}

int abcdk_bind(int fd, const abcdk_sockaddr_t *addr)
{
    socklen_t len;

    assert(fd >= 0 && addr != NULL);

    len = sizeof(abcdk_sockaddr_t);
    if(addr->family == AF_UNIX)
    {
        len = SUN_LEN(&addr->addr_un);
    }
    else if(addr->family == AF_INET)
    {
        len = sizeof(struct sockaddr_in);
    }
    else if(addr->family == AF_INET6)
    {
        len = sizeof(struct sockaddr_in6);
    }

    return bind(fd, &addr->addr,len);
}

int abcdk_accept(int fd, abcdk_sockaddr_t *addr)
{
    int sub_fd = -1;
    socklen_t addrlen = sizeof(abcdk_sockaddr_t);

    assert(fd >= 0);

    if (addr)
        sub_fd = accept(fd, &addr->addr, &addrlen);
    else
        sub_fd = accept(fd, NULL, NULL);

    if (sub_fd < 0)
        return -1;

    /* 添加个非必要标志，忽略可能的出错信息。 */
    abcdk_fflag_add(sub_fd, O_CLOEXEC);

    return sub_fd;
}

int abcdk_connect(int fd, abcdk_sockaddr_t *addr, time_t timeout)
{
    socklen_t len;
    int flags = 0;
    int eno = 0;
    int chk;

    assert(fd >= 0 && addr != NULL);

    flags = abcdk_fflag_get(fd);
    if (flags == -1)
        return -1;

    /* 添加非阻塞标志，用于异步连接。*/
    chk = 0;
    if (!(flags & O_NONBLOCK))
        chk = abcdk_fflag_add(fd, O_NONBLOCK);

    if (chk != 0)
        return -1;

    len = sizeof(abcdk_sockaddr_t);
    if(addr->family == AF_UNIX)
    {
        len = SUN_LEN(&addr->addr_un);
    }
    else if(addr->family == AF_INET)
    {
        len = sizeof(struct sockaddr_in);
    }
    else if(addr->family == AF_INET6)
    {
        len = sizeof(struct sockaddr_in6);
    }

    chk = connect(fd, &addr->addr, len);
    if(chk == 0)
        goto final;

    if (errno != EINPROGRESS && errno != EWOULDBLOCK && errno != EAGAIN)
        goto final;

    /* 等待写事件(允许)。 */
    chk = (abcdk_poll(fd, 0x02, timeout) > 0 ? 0 : -1);
    if(chk != 0)
        goto final;

    /* 获取SOCKET句柄的出错码。 */
    chk = abcdk_sockopt_option_int(fd, SOL_SOCKET, SO_ERROR, &eno, 1);
    chk = (eno == 0 ? 0 : -1);

final:
    
    /* 恢复原有的标志，忽略可能的出错信息。*/
    if (!(flags & O_NONBLOCK))
        abcdk_fflag_del(fd, O_NONBLOCK);

    return chk;
}

int abcdk_sockaddr_from_string(abcdk_sockaddr_t *dst, const char *src, int try_lookup)
{
    sa_family_t family = AF_UNSPEC;
    char name[NAME_MAX] = {0};
    uint16_t port = 0;
    int chk;

    assert(dst != NULL && src != NULL);

    if (abcdk_strncmp("uinx://", src, 7, 0) == 0)
    {
        dst->family = AF_UNIX;
        return abcdk_sockaddr_from_string(dst, src + 7, try_lookup);
    }
    else if (abcdk_strncmp("ipv4://", src, 7, 0) == 0)
    {
        dst->family = AF_INET;
        return abcdk_sockaddr_from_string(dst, src + 7, try_lookup);
    }
    else if (abcdk_strncmp("ipv6://", src, 7, 0) == 0)
    {
        dst->family = AF_INET6;
        return abcdk_sockaddr_from_string(dst, src + 7, try_lookup);
    }
    else if (src[0] == '/')
    {
        if(dst->family != AF_UNIX)
            dst->family = AF_UNIX;

        strncpy(dst->addr_un.sun_path, src, 108);
        return 0;
    }
    else if (strchr(src, '['))
    {
        if(dst->family != AF_INET6)
            dst->family = AF_INET6;

        sscanf(src, "%*[[ ]%[^] ]%*[] :,]%hu", name, &port);
    }
    else if (strchr(src, ','))
    {
         if(dst->family != AF_INET6)
            dst->family = AF_INET6;

        sscanf(src, "%[^, ]%*[, ]%hu", name, &port);
    }
    else if (strchr(src, ':'))
    {
        if(dst->family != AF_INET && dst->family != AF_INET6)
            dst->family = AF_UNSPEC;

        sscanf(src, "%[^: ]%*[: ]%hu", name, &port);
    }
    else
    {
        if(dst->family != AF_INET && dst->family != AF_INET6)
            dst->family = AF_UNSPEC;
            
        strncpy(name,src,NAME_MAX);
    }

    if(dst->family == AF_UNSPEC)
    {
        /*Try to IPV4.*/
        chk = abcdk_inet_pton(name, AF_INET, dst);
        if(chk == 0)
            goto TRY_PORT;

        /*Try to IPV6.*/
        chk = abcdk_inet_pton(name, AF_INET6, dst);
        if(chk == 0)
            goto TRY_PORT;
    }
    else 
    {
        chk = abcdk_inet_pton(name, dst->family, dst);
        if(chk == 0)
            goto TRY_PORT;
    }

    /*走到这里表示地址转换失败，尝试域名解析。*/
    if(try_lookup)
    {
        chk = (abcdk_gethostbyname(name, dst->family, dst, 1, NULL) == 1 ? 0 : -1);
        if(chk == 0)
            goto TRY_PORT;
    }

    /*地址换转和域名解析都失败。*/
    return -1;

TRY_PORT:

    /*地址转换成功后，再转换端口号。*/

    if (dst->family == AF_INET6)
        dst->addr6.sin6_port = abcdk_endian_h_to_b16(port);
    else if (dst->family == AF_INET)
        dst->addr4.sin_port = abcdk_endian_h_to_b16(port);
    else 
        return -22;

    return 0;
}

char *abcdk_sockaddr_to_string(char dst[NAME_MAX],const abcdk_sockaddr_t *src,int ex_port)
{
    char buf[INET6_ADDRSTRLEN] = {0};

    assert(dst != NULL && src != NULL);
    assert(src->family == AF_UNIX || src->family == AF_INET || src->family == AF_INET6);

    if (src->family == AF_UNIX)
    {
        strcpy(dst, src->addr_un.sun_path);
    }
    else
    {
        if(ex_port)
        {
            if (abcdk_inet_ntop(src, dst, INET6_ADDRSTRLEN) == NULL)
                return NULL;
        }
        else if(src->family == AF_INET)
        {
            if (abcdk_inet_ntop(src, dst, INET6_ADDRSTRLEN) == NULL)
                return NULL;

            sprintf(dst+strlen(dst),":%d",abcdk_endian_b_to_h16(src->addr4.sin_port));
        }
        else if(src->family == AF_INET6)
        {
            strcat(dst,"[");

            if (abcdk_inet_ntop(src+1, dst, INET6_ADDRSTRLEN) == NULL)
                return NULL;
            
            strcat(dst,"]");

            sprintf(dst+strlen(dst),":%d",abcdk_endian_b_to_h16(src->addr6.sin6_port));
        }
    }

    return dst;
}

int abcdk_sockaddr_where(const abcdk_sockaddr_t *test,int where)
{
    int addr_num = 0;
    int addr_max = 100;
    abcdk_ifaddrs_t *addrs = NULL;
    abcdk_ifaddrs_t *addr_p = NULL;
    int match_num = 0;
    int chk = -1;

    assert(test != NULL && (where ==1 || where ==2));

    addrs = (abcdk_ifaddrs_t *)abcdk_heap_alloc(sizeof(abcdk_ifaddrs_t)*addr_max);
    if(!addrs)
        ABCDK_ERRNO_AND_RETURN1(errno,0);

    addr_num = abcdk_ifname_fetch(addrs,addr_max,0,0);

    for (int i = 0; i < addr_num; i++)
    {
        addr_p = addrs+i;// &addrs[i]

        /*只比较同类型的地址。*/
        if(addr_p->addr.family != test->family)
            continue;

        if (addr_p->addr.family == AF_INET6)
            chk = memcmp(&addr_p->addr.addr6.sin6_addr, &test->addr6.sin6_addr, sizeof(struct in6_addr));
        if (addr_p->addr.family == AF_INET)
            chk = memcmp(&addr_p->addr.addr4.sin_addr, &test->addr4.sin_addr, sizeof(struct in_addr));

        /*地址相同则计数。*/
        if (chk == 0)
            match_num += 1;
    }

    abcdk_heap_freep((void**)&addrs);

    if (where == 1)
        return ((match_num > 0) ? 1 : 0);
    if (where == 2)
        return ((match_num <= 0) ? 1 : 0);

    return 0;
}

int abcdk_sockaddr_compare(const abcdk_sockaddr_t *addr1, const abcdk_sockaddr_t *addr2,int care_port)
{
    uint32_t a1,a2;
    int32_t p1,p2;
    int chk;

    assert(addr1 != NULL && addr2 != NULL);
    assert(addr1->family == AF_INET || addr1->family == AF_INET6);
    assert(addr2->family == AF_INET || addr2->family == AF_INET6);
    
    chk = -1;
    if (addr1->family == AF_INET)
    {
        p1 = abcdk_endian_b_to_h16(addr1->addr4.sin_port);
        p2 = abcdk_endian_b_to_h16(addr2->addr4.sin_port);

        chk = memcmp(&addr1->addr4.sin_addr, &addr2->addr4.sin_addr, sizeof(struct in_addr));
    }
    else if (addr1->family == AF_INET6)
    {
        p1 = abcdk_endian_b_to_h16(addr1->addr6.sin6_port);
        p2 = abcdk_endian_b_to_h16(addr2->addr6.sin6_port);

        chk = memcmp(&addr1->addr6.sin6_addr, &addr2->addr6.sin6_addr, sizeof(struct in6_addr));
    }

    if (chk == 0 && care_port)
        chk = p1 - p2;

    return chk;
}

void abcdk_sockaddr_make_netmask(abcdk_sockaddr_t *mask, const abcdk_sockaddr_t *addr, int prefix)
{
    int iplen;

    assert(mask != NULL && addr != NULL && prefix >=0);
    assert(addr->family == AF_INET || addr->family == AF_INET6);

    iplen = ((addr->family == AF_INET6) ? 16 * 8 : 4 * 8);

    assert(iplen >= prefix);

    /*copy*/
    *mask = *addr;

    for (int i = prefix; i < iplen; i++)
    {
        if (mask->family == AF_INET6)
            abcdk_bloom_write((uint8_t *)&mask->addr6.sin6_addr, 16, i, 0);
        else
            abcdk_bloom_write((uint8_t *)&mask->addr4.sin_addr, 4, i, 0);
    }
}

char *abcdk_sockaddr_make_netmask2(char buf[100], sa_family_t family, const char *host, int prefix)
{
    abcdk_sockaddr_t mask,addr;

    assert(buf != NULL && (family == AF_INET||family == AF_INET6) && host != NULL && prefix >=0);

    abcdk_inet_pton(host, family, &addr);
    abcdk_sockaddr_make_netmask(&mask,&addr,prefix);

    return abcdk_inet_ntop(&mask, buf, 100);
}


