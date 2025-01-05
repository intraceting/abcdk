/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/if_tun.h>
#include <arpa/inet.h>

// 打开TUN设备并配置名称、IP地址和掩码
static int tun_open(const char *dev_name, const char *ip_addr, const char *netmask)
{
    int tun_fd;
    struct ifreq ifr;
    int ret;

    // 打开TUN设备
    if ((tun_fd = open("/dev/net/tun", O_RDWR)) < 0)
    {
        perror("Opening /dev/net/tun");
        return -1;
    }

    // 设置ifreq结构体中的设备名字段
    memset(&ifr, 0, sizeof(ifr));
    ifr.ifr_flags = IFF_TUN | IFF_NO_PI;
    strncpy(ifr.ifr_name, dev_name, IFNAMSIZ);

    // 通过ioctl调用将TUN设备添加到系统中
    if ((ret = ioctl(tun_fd, TUNSETIFF, (void *)&ifr)) < 0)
    {
        perror("ioctl(TUNSETIFF)");
        close(tun_fd);
        return -1;
    }

    // 使用系统调用配置TUN设备的IP地址和掩码
    int fd;
    struct sockaddr_in sin;

    // 创建一个socket
    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("socket");
        close(tun_fd);
        return -1;
    }

    // 设置ifreq结构体中的设备名字段
    strncpy(ifr.ifr_name, dev_name, IFNAMSIZ);

    // 将IP地址转换为网络字节序
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = inet_addr(ip_addr);
    memcpy(&ifr.ifr_addr, &sin, sizeof(struct sockaddr));

    // 使用ioctl调用设置TUN设备的IP地址
    if ((ret = ioctl(fd, SIOCSIFADDR, &ifr)) < 0)
    {
        perror("ioctl(SIOCSIFADDR)");
        close(fd);
        close(tun_fd);
        return -1;
    }

    // 将掩码转换为网络字节序
    sin.sin_addr.s_addr = inet_addr(netmask);
    memcpy(&ifr.ifr_netmask, &sin, sizeof(struct sockaddr));

    // 使用ioctl调用设置TUN设备的掩码
    if ((ret = ioctl(fd, SIOCSIFNETMASK, &ifr)) < 0)
    {
        perror("ioctl(SIOCSIFNETMASK)");
        close(fd);
        close(tun_fd);
        return -1;
    }

    // 启用TUN设备
    ifr.ifr_flags |= IFF_UP;
    if ((ret = ioctl(fd, SIOCSIFFLAGS, &ifr)) < 0) {
        perror("ioctl(SIOCSIFFLAGS)");
        close(fd);
        close(tun_fd);
        return -1;
    }

    // 关闭socket并返回TUN设备的文件描述符
    close(fd);
    return tun_fd;
}

#define MAX_BUF_SIZE 2000

static int test1()
{
    int tun_fd1 = tun_open("tun0", "10.10.0.1", "255.255.255.0");
    int tun_fd2 = tun_open("tun1", "10.10.0.2", "255.255.255.0");

    //abcdk_net_route_add(4,"0.0.0.0",0,"10.10.0.254",0,"tun0");

    char buf1[MAX_BUF_SIZE];
    char buf2[MAX_BUF_SIZE];

    fd_set read_fds;
    struct timeval tv;
    int max_fd = (tun_fd1 > tun_fd2) ? tun_fd1 : tun_fd2;

    while (1)
    {
        FD_ZERO(&read_fds);
        FD_SET(tun_fd1, &read_fds);
        FD_SET(tun_fd2, &read_fds);

        tv.tv_sec = 1; // 设置超时时间为1秒
        tv.tv_usec = 0;

        int ret = select(max_fd + 1, &read_fds, NULL, NULL, &tv);
        if (ret == -1)
        {
            perror("select");
            break;
        }
        else if (ret == 0)
        {
            // 超时
            printf("Timeout\n");
            continue;
        }

        // 检查每个TUN设备的读写事件
        if (FD_ISSET(tun_fd1, &read_fds))
        {
            ssize_t nread = read(tun_fd1, buf1, sizeof(buf1));
            if (nread < 0)
            {
                perror("Reading from TUN device tun0");
                break;
            }
            printf("Read %zd bytes from TUN device tun0\n", nread);

            write(tun_fd2, buf1, nread);
        }

        if (FD_ISSET(tun_fd2, &read_fds))
        {
            ssize_t nread = read(tun_fd2, buf2, sizeof(buf2));
            if (nread < 0)
            {
                perror("Reading from TUN device tun1");
                break;
            }
            printf("Read %zd bytes from TUN device tun1\n", nread);

            write(tun_fd1, buf2, nread);
        }
    }

    close(tun_fd1);
    close(tun_fd2);
    return 0;
}

int abcdk_test_tun(abcdk_option_t *args)
{
    test1();

    return 0;
}