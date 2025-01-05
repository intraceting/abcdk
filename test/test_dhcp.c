/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
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
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#define DHCP_SERVER_PORT 67
#define DHCP_CLIENT_PORT 6800

// DHCP消息类型
#define DHCPDISCOVER 1
#define DHCPOFFER 2
#define DHCPREQUEST 3
#define DHCPACK 5
#define DHCPNAK 6

struct dhcp_message
{
    uint8_t op;
    uint8_t htype;
    uint8_t hlen;
    uint8_t hops;
    uint32_t xid;
    uint16_t secs;
    uint16_t flags;
    uint32_t ciaddr;
    uint32_t yiaddr;
    uint32_t siaddr;
    uint32_t giaddr;
    unsigned char chaddr[16];
    unsigned char padding[192];
    uint32_t magic_cookie;
    unsigned char options[64];
};

static void fill_dhcp_discover(struct dhcp_message *packet, uint32_t xid)
{
    // 设置DHCP消息头
    packet->op = 1;    // BOOTREQUEST
    packet->htype = 1; // Ethernet
    packet->hlen = 6;  // MAC address length
    packet->hops = 0;
    packet->xid = htonl(xid);
    packet->secs = htons(0);
    packet->flags = htons(0);
    packet->ciaddr = 0;
    packet->yiaddr = 0;
    packet->siaddr = 0;
    packet->giaddr = 0;
    memset(packet->chaddr, 0, sizeof(packet->chaddr));
    // 填充Magic Cookie
    packet->magic_cookie = htonl(0x63825363);
    // 填充DHCP消息类型
    packet->options[0] = 53; // DHCP Message Type
    packet->options[1] = 1;  // Option length
    packet->options[2] = DHCPDISCOVER;
    // 添加结束标志
    packet->options[3] = 255;
}

static void dhcpv4_client()
{
    // 创建IPv4 UDP套接字
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == -1)
    {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // 设置广播选项
    int broadcast = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) == -1)
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    // 准备服务器和客户端地址结构
    struct sockaddr_in server_addr, client_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    memset(&client_addr, 0, sizeof(client_addr));

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(DHCP_SERVER_PORT);
    server_addr.sin_addr.s_addr = INADDR_BROADCAST;

    client_addr.sin_family = AF_INET;
    client_addr.sin_port = htons(DHCP_CLIENT_PORT);
    client_addr.sin_addr.s_addr = INADDR_ANY;

    // 绑定客户端地址
    if (bind(sockfd, (struct sockaddr *)&client_addr, sizeof(client_addr)) == -1)
    {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    // 构造DHCP Discover消息
    struct dhcp_message discover_packet;
    fill_dhcp_discover(&discover_packet, 123456); // 123456 是随机选择的XID

    // 发送DHCP Discover消息
    if (sendto(sockfd, &discover_packet, sizeof(discover_packet), 0, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)
    {
        perror("sendto");
        exit(EXIT_FAILURE);
    }

    // 接收服务器的DHCP Offer消息
    struct dhcp_message offer_packet;
    ssize_t rlen = recvfrom(sockfd,&offer_packet,sizeof(offer_packet),0,NULL,NULL);
    if(rlen == -1)
    {
        perror("sendto");
        exit(EXIT_FAILURE);
    }

    // 此处需要解析DHCP Offer消息的格式，获取IP等信息

    // 关闭套接字
    close(sockfd);
}

int abcdk_test_dhcp(abcdk_option_t *args)
{
    dhcpv4_client();

    return 0;
}