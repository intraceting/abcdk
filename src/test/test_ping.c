/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "entry.h"

#if 0
 
#define ICMP_SIZE (sizeof(struct icmp))
#define ICMP_ECHO 8
#define ICMP_ECHOREPLY 0
#define BUF_SIZE 1024
#define NUM   5    // 发送报文次数
 
#define UCHAR  unsigned char
#define USHORT unsigned short
#define UINT   unsigned int 
 
// ICMP报文数据结构
struct icmp{
    UCHAR           type;      // 类型
    UCHAR           code;      // 代码
    USHORT          checksum;  // 校验和
    USHORT          id;        // 标识符
    USHORT          sequence;  // 序号 
    struct timeval  timestamp; // 时间戳
};
 
 
// IP首部数据结构
struct ip{
    // 主机字节序判断
    #if __BYTE_ORDER == __LITTLE_ENDIAN
    UCHAR   hlen:4;        // 首部长度
    UCHAR   version:4;     // 版本      
    #endif
    #if __BYTE_ORDER == __BIG_ENDIAN
    UCHAR   version:4;       
    UCHAR   hlen:4;    
    #endif    
    UCHAR   tos;             // 服务类型
    USHORT  len;             // 总长度
    USHORT  id;                // 标识符
    USHORT  offset;            // 标志和片偏移
    UCHAR   ttl;            // 生存时间
    UCHAR   protocol;       // 协议
    USHORT  checksum;       // 校验和
    struct in_addr ipsrc;    // 32位源ip地址
    struct in_addr ipdst;   // 32位目的ip地址
};
 
 
char buf[BUF_SIZE] = {0};
 
USHORT checkSum(USHORT *, int); // 计算校验和
float timediff(struct timeval *, struct timeval *); // 计算时间差
void pack(struct icmp *, int);  // 封装一个ICMP报文
int unpack(char *, int, char *);        // 对接收到的IP报文进行解包
 
 
int abcdk_test_ping(abcdk_option_t *args)
{
    struct hostent *host;
    struct icmp sendicmp;
    struct sockaddr_in from;
    struct sockaddr_in to;
    int fromlen = 0;
    int sockfd;
    int nsend = 0;
    int nreceived = 0;
    int i, n;
    in_addr_t inaddr;
 
    memset(&from, 0, sizeof(struct sockaddr_in));
    memset(&to, 0, sizeof(struct sockaddr_in));
 
    const char *dst_p = abcdk_option_get(args,"--dst",0,"127.0.0.1");
 
    // 生成原始套接字
    if((sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) == -1){
        printf("socket() error \n");
        return errno;
    }
 
    // 设置目的地址信息
    to.sin_family = AF_INET;
 
    // 判断是域名还是ip地址
    if(inaddr = inet_addr(dst_p) == INADDR_NONE){
        // 是域名
        if((host = gethostbyname(dst_p)) == NULL){
            printf("gethostbyname() error \n");
            return errno;
        }
        to.sin_addr = *(struct in_addr *)host->h_addr_list[0];
    }else{
        // 是ip地址
        to.sin_addr.s_addr = inaddr;
    }
 
    // 输出域名ip地址信息
    printf("ping %s (%s) : %d bytes of data.\n", dst_p, inet_ntoa(to.sin_addr), (int)ICMP_SIZE);
 
    //循环发送报文, 接收报文 
    for(i = 0; i < NUM; i++){
        nsend++;  // 发送次数加1
        memset(&sendicmp, 0, ICMP_SIZE);
        pack(&sendicmp, nsend);
 
        // 发送报文
        if(sendto(sockfd, &sendicmp, ICMP_SIZE, 0, (struct sockaddr *)&to, sizeof(to)) == -1){
            printf("sendto() error \n");
            continue;
        }
 
        // 接收报文
        if((n = recvfrom(sockfd, buf, BUF_SIZE, 0, (struct sockaddr *)&from, &fromlen)) < 0){
            printf("recvform() error \n");
            continue;
        }
        nreceived++;  // 接收次数加1
        if(unpack(buf, n, inet_ntoa(from.sin_addr)) == -1){
            printf("unpack() error \n");
        }
 
        sleep(1);
    }
 
    // 输出统计信息
    printf("---  %s ping statistics ---\n", dst_p);
    printf("%d packets transmitted, %d received, %%%d packet loss\n", nsend, nreceived, 
            (nsend - nreceived) / nsend * 100);
 
    return 0;
}
 
/**
 * addr 指向需校验数据缓冲区的指针
 * len  需校验数据的总长度(字节单位)
 */
USHORT checkSum(USHORT *addr, int len){
    UINT sum = 0;  
    while(len > 1){
        sum += *addr++;
        len -= 2;
    }
 
    // 处理剩下的一个字节
    if(len == 1){
        sum += *(UCHAR *)addr;
    }
 
    // 将32位的高16位与低16位相加
    sum = (sum >> 16) + (sum & 0xffff);
    sum += (sum >> 16);
 
    return (USHORT) ~sum;
}
 
/**
 * 返回值单位: ms
 * begin 开始时间戳
 * end   结束时间戳
 */
float timediff(struct timeval *begin, struct timeval *end){
    int n;
    // 先计算两个时间点相差多少微秒
    n = ( end->tv_sec - begin->tv_sec ) * 1000000
        + ( end->tv_usec - begin->tv_usec );
 
    // 转化为毫秒返回
    return (float) (n / 1000);
}
 
/**
 * icmp 指向需要封装的ICMP报文结构体的指针
 * sequence 该报文的序号
 */ 
void pack(struct icmp * icmp, int sequence){
    icmp->type = ICMP_ECHO;
    icmp->code = 0;
    icmp->checksum = 0;
    icmp->id = getpid();
    icmp->sequence = sequence;
    gettimeofday(&icmp->timestamp, 0);
    icmp->checksum = checkSum((USHORT *)icmp, ICMP_SIZE);
}
 
/**
 * buf  指向接收到的IP报文缓冲区的指针
 * len  接收到的IP报文长度
 * addr 发送ICMP报文响应的主机IP地址
 */ 
int unpack(char * buf, int len, char * addr){
   int i, ipheadlen;
   struct ip * ip;
   struct icmp * icmp;
   float rtt;          // 记录往返时间
   struct timeval end; // 记录接收报文的时间戳
 
   ip = (struct ip *)buf;
 
   // 计算ip首部长度, 即ip首部的长度标识乘4
   ipheadlen = ip->hlen << 2;
 
   // 越过ip首部, 指向ICMP报文
   icmp = (struct icmp *)(buf + ipheadlen);
 
   // ICMP报文的总长度
   len -= ipheadlen;
 
   // 如果小于ICMP报文首部长度8
   if(len < 8){
        printf("ICMP packets\'s length is less than 8 \n"); 
        return -1;
   }
 
   // 确保是我们所发的ICMP ECHO回应
   if(icmp->type != ICMP_ECHOREPLY ||
           icmp->id != getpid()){    
       printf("ICMP packets are not send by us \n");
       return -1;
   }
 
   // 计算往返时间
   gettimeofday(&end, 0);
   rtt = timediff(&icmp->timestamp, &end);
 
   // 打印ttl, rtt, seq
   printf("%d bytes from %s : icmp_seq=%u ttl=%d rtt=%fms \n",
           len, addr, icmp->sequence, ip->ttl, rtt);
 
   return 0;
}

#elif 1

#define DATA_SIZE 32
#define MAX_RECV_SIZE 1024

typedef struct _TAG_IP_HEADER
{
    u_int8_t    ip_head_verlen;
    u_int8_t    ip_tos;
    u_int16_t   ip_length;
    u_int16_t   ip_id;
    u_int16_t   ip_flags;
    u_int8_t    ip_ttl;
    u_int8_t    ip_protacol;
    u_int16_t   ip_checksum;
    u_int32_t   ip_source;
    u_int32_t   ip_destination;
} IP_HEADER;

typedef struct _TAG_IMCP_HEADER
{
    u_int8_t    icmp_type;
    u_int8_t    icmp_code;
    u_int16_t   check_sum;
    u_int16_t   echo_id;
    u_int16_t   echo_seq;
} ICMP_HEADER;

typedef struct _TAG_ICMP_PACKET
{
    ICMP_HEADER     icmp_header;
    struct timeval  icmp_time;
    u_int16_t       icmp_sum_flag;
    u_int8_t        imcp_data[DATA_SIZE];
} ICMP_PACKET;

typedef struct _TAG_THREAD_DATA
{
    int         fd;
    u_int32_t   times;
    ICMP_PACKET * icmp_packet;
    char        * buffer;
    u_int32_t   buffer_len;
    struct sockaddr_in * sockaddr;
    u_int8_t    send_flag;
} THREAD_DATA;

static u_int16_t generation_checksum(u_int16_t * buf, u_int32_t size);
static double get_time_interval(struct timeval * start, struct timeval * end);

u_int16_t generation_checksum(u_int16_t * buf, u_int32_t size)
{
    u_int64_t cksum = 0;
    while(size > 1)
    {
        cksum += *buf++;
        size -= sizeof(u_int16_t);
    }

    if(size)
    {
        cksum += *buf++;
    }

    cksum =  (cksum>>16) + (cksum & 0xffff);
    cksum += (cksum>>16);

    return (u_int16_t)(~cksum);
}

static double get_time_interval(struct timeval * start, struct timeval * end)
{
    double interval;
    struct timeval tp;

    tp.tv_sec = end->tv_sec - start->tv_sec;
    tp.tv_usec = end->tv_usec - start->tv_usec;
    if(tp.tv_usec < 0)
    {
        tp.tv_sec -= 1;
        tp.tv_usec += 1000000;
    }

    interval = tp.tv_sec * 1000 + tp.tv_usec * 0.001;
    return interval;
}

static void * send_imcp(void *arg)
{
    u_int8_t *flag  = NULL;
    int times       = -1;
    int fd          = -1;
    char * buffer   = NULL;
    struct sockaddr_in * dest_socket_addr;
    ICMP_HEADER *icmp_header = NULL;
    ICMP_PACKET *icmp_packet = NULL;

    THREAD_DATA *thread_data = (THREAD_DATA *)arg;
    if (thread_data == NULL)
    {
        return NULL;
    }

    dest_socket_addr = thread_data->sockaddr;
    if (dest_socket_addr == NULL)
    {
        return NULL;
    }

    flag  = &thread_data->send_flag;
    if (flag == NULL)
    {
        return NULL;
    }

    times = thread_data->times;
    fd = thread_data->fd;
    if (fd <= 0)
    {
        return NULL;
    }

    icmp_packet = thread_data->icmp_packet;
    if (icmp_packet == NULL)
    {
        return NULL;
    }

    icmp_header = &(icmp_packet->icmp_header);
    if (icmp_header == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < thread_data->times; i++)
    {
        long result = -1;
        icmp_header->echo_seq = htons(i);
        icmp_header->check_sum = 0;

        //fprintf(stderr,"send packet. %s\n", inet_ntoa(*((struct in_addr*)&(dest_socket_addr->sin_addr.s_addr))));
        gettimeofday(&icmp_packet->icmp_time, NULL);
        icmp_packet->icmp_sum_flag = 0;
        icmp_header->check_sum = generation_checksum((u_int16_t *) icmp_packet, sizeof(ICMP_PACKET));
        //fprintf(stderr,"send sum: %x\n", icmp_header->check_sum);
        result = sendto(fd, icmp_packet, sizeof(ICMP_PACKET), 0, (struct sockaddr *)dest_socket_addr,
                        sizeof(struct sockaddr_in));
        if (result == -1)
        {
            fprintf(stderr,"PING: sendto: Network is unreachable\n");
            continue;
        }

        sleep(1);
    }

    *flag = 0;
    return NULL;
}

static void * recv_imcp(void *arg)
{
    u_int8_t *flag  = NULL;
    int times       = -1;
    int fd          = -1;
    char * buffer   = NULL;
    ICMP_HEADER *icmp_header = NULL;
    ICMP_PACKET *icmp_packet = NULL;

    THREAD_DATA *thread_data = (THREAD_DATA *)arg;
    if (thread_data == NULL)
    {
        return NULL;
    }

    flag  = &thread_data->send_flag;
    if (flag == NULL)
    {
        return NULL;
    }

    times = thread_data->times;
    fd = thread_data->fd;
    if (fd <= 0)
    {
        return NULL;
    }

    icmp_packet = thread_data->icmp_packet;
    if (icmp_packet == NULL)
    {
        return NULL;
    }

    icmp_header = &(icmp_packet->icmp_header);
    if (icmp_header == NULL)
    {
        return NULL;
    }

    struct sockaddr_in from;
    socklen_t from_packet_len;
    long read_length;
    char recv_buf[MAX_RECV_SIZE];
    struct timeval end;

    from_packet_len = sizeof(struct sockaddr_in);
    for (int index = 0; index < times && *flag == 1;)
    {
        read_length = recvfrom(fd, recv_buf, MAX_RECV_SIZE, 0,
                               (struct sockaddr*)&from, &from_packet_len);
        gettimeofday( &end, NULL );
        if(read_length != -1)
        {
            IP_HEADER * recv_ip_header = (IP_HEADER*)recv_buf;
            int ip_ttl = (int)recv_ip_header->ip_ttl;
            ICMP_PACKET * recv_icmp = (ICMP_PACKET *)(recv_buf +
                                                      (recv_ip_header->ip_head_verlen & 0x0F) * 4);
            //icmp_header * recv_icmp_header = (icmp_header *)(recv_buf +
            //    (recv_ip_header->ip_head_verlen & 0x0F) * 4);

            //fprintf(stderr,"id: %d, seq: %d\n", recv_icmp->icmp_header.icmp_id, recv_icmp->icmp_header.icmp_seq);
            if(recv_icmp->icmp_header.icmp_type != 0)
            {
                fprintf(stderr,"error type %d received, error code %d \n", recv_icmp->icmp_header.icmp_type, recv_icmp->icmp_header.icmp_code);
                continue;
            }

            //if(recv_icmp->icmp_header.id != icmp_head->id)
            //{
            //fprintf(stderr,"some else's packet\n");
            //break;
            //}

            if (recv_icmp->icmp_sum_flag != icmp_packet->icmp_sum_flag)
            {
                fprintf(stderr,"check sum fail.\n");
                continue;
            }

            if(read_length >= (0 + sizeof(ICMP_PACKET)))
            {
                index++;
                snprintf(thread_data->buffer, thread_data->buffer_len, "%s%ld bytes from (%s): icmp_seq=%d time=%.2f ms\n",
                 thread_data->buffer, read_length, inet_ntoa(from.sin_addr),
                 recv_icmp->icmp_header.echo_seq / 256, get_time_interval(&recv_icmp->icmp_time, &end));

                fprintf(stderr,"%ld bytes from (%s): icmp_seq=%d ttl=%d time=%.2f ms\n",
                          read_length, inet_ntoa(from.sin_addr), recv_icmp->icmp_header.echo_seq / 256,
                          ip_ttl, get_time_interval(&recv_icmp->icmp_time, &end));
            }
        }
        else
        {
            if (errno != EAGAIN)
            {
                fprintf(stderr,"receive data error: %s\n", strerror(errno));
                snprintf(thread_data->buffer, thread_data->buffer_len, "receive data error: %s\n", strerror(errno));
            }
        }
    }

    return NULL;
}

int get_ping_result(const char * domain, u_int32_t times, char * res_buffer, int buffer_len)
{
    int ret = 0;
    int client_fd = -1;
    int size = 50 * MAX_RECV_SIZE;
    struct timeval timeout;

    ICMP_PACKET * icmp_packet = NULL;
    ICMP_HEADER * icmp_header = NULL;
    struct timeval * icmp_send_time = NULL;

    in_addr_t dest_ip;
    struct sockaddr_in dest_socket_addr;

    pthread_t send_pid;
    pthread_t recv_pid;

    THREAD_DATA thread_data;

    if (res_buffer == NULL || domain == NULL || buffer_len == 0)
    {
        fprintf(stderr,"res_buffer: %s, domain: %s, buffer_len: %d\n", res_buffer, domain, buffer_len);
        return ret;
    }

    dest_ip = inet_addr(domain);
    if (dest_ip == INADDR_NONE)
    {
        struct hostent* p_hostent = gethostbyname(domain);
        if(p_hostent)
        {
            dest_ip = (*(in_addr_t*)p_hostent->h_addr);
        }
    }

    if (dest_ip == INADDR_NONE)
    {
        return ret;
    }

    client_fd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (client_fd == -1)
    {
        fprintf(stderr,"socket error: %s!\n", strerror(errno));
        return ret;
    }

    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
    if(setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(struct timeval)))
    {
        fprintf(stderr,"setsocketopt SO_RCVTIMEO error: %s\n", strerror(errno));
        return ret;
    }

    if(setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(struct timeval)))
    {
        fprintf(stderr,"setsockopt SO_SNDTIMEO error: %s\n", strerror(errno));
        return ret;
    }

    memset(dest_socket_addr.sin_zero, 0, sizeof(dest_socket_addr.sin_zero));
    dest_socket_addr.sin_family = AF_INET;
    dest_socket_addr.sin_addr.s_addr = dest_ip;
    dest_socket_addr.sin_port = htons(0);

    icmp_packet = (ICMP_PACKET *)malloc(sizeof(ICMP_PACKET));
    if (icmp_packet == NULL)
    {
        fprintf(stderr,"malloc error.\n");
        return ret;
    }

    memset(icmp_packet, 0, sizeof(ICMP_PACKET));

    icmp_header = &icmp_packet->icmp_header;
    icmp_header->icmp_type = 8;
    icmp_header->icmp_code = 0;
    icmp_header->echo_id = getpid();

    icmp_packet->icmp_sum_flag = generation_checksum((u_int16_t *)icmp_packet, sizeof(ICMP_PACKET));
    fprintf(stderr,"PING %s (%s).\n", domain, inet_ntoa(*((struct in_addr*)&dest_ip)));
    snprintf(res_buffer, buffer_len, "PING %s (%s).\n", domain, inet_ntoa(*((struct in_addr*)&dest_ip)));

    thread_data.send_flag   = 1;
    thread_data.sockaddr    = &dest_socket_addr;
    thread_data.fd          = client_fd;
    thread_data.buffer      = res_buffer;
    thread_data.times       = times;
    thread_data.icmp_packet = icmp_packet;

    ret = pthread_create(&send_pid, NULL, send_imcp, (void *)&thread_data);
    if (ret < 0)
    {
        fprintf(stderr,"pthread create error: %s\n", strerror(errno));
        goto FAIL_EXIT;
    }

    ret = pthread_create(&recv_pid, NULL, recv_imcp, (void *)&thread_data);
    if (ret < 0)
    {
        fprintf(stderr,"pthread create error: %s\n", strerror(errno));
        pthread_detach(send_pid);
        goto FAIL_EXIT;
    }

    pthread_join(send_pid, NULL);
    pthread_join(recv_pid, NULL);

    FAIL_EXIT:
    if (icmp_packet != NULL)
    {
        free(icmp_packet);
        icmp_packet = NULL;
    }

    if (client_fd >= 0)
    {
        close(client_fd);
        client_fd = -1;
    }

    return ret;
}

int abcdk_test_ping(abcdk_option_t *args)
{
    char buffer[MAX_RECV_SIZE];

#ifndef _SYS_CAPABILITY_H
    setuid(getuid());
#else
    abcdk_cap_set_pid(getpid(),CAP_NET_RAW,CAP_EFFECTIVE,CAP_SET);
    abcdk_cap_set_pid(getpid(),CAP_NET_RAW,CAP_PERMITTED,CAP_SET);

    assert(abcdk_cap_get_pid(getpid(),CAP_NET_RAW,CAP_EFFECTIVE)==1);
    assert(abcdk_cap_get_pid(getpid(),CAP_NET_RAW,CAP_PERMITTED)==1);
#endif

    const char *dst_p = abcdk_option_get(args,"--dst",0,"127.0.0.1");

    int ret = get_ping_result(dst_p,10, buffer, 10);

    fprintf(stderr,"ping result: %d\n", ret);

    return 0;
}
#endif

