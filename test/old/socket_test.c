/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "abcdk/util/socket.h"

void test_gethostbyname()
{
    abcdk_sockaddr_t * addrs = abcdk_heap_alloc(sizeof(abcdk_sockaddr_t)*10);

    char*canonname = NULL;
    //int n = abcdk_gethostbyname("ip6-localhost",ABCDK_IPV6,addr,10);
    int n = abcdk_gethostbyname("www.taobao.com",ABCDK_IPV4,addrs,10,&canonname);
    
    printf("%s\n",canonname);

    for (int i = 0; i < n; i++)
    {
        char buf[100];

        assert(abcdk_inet_ntop(&addrs[i], buf, 100));

        printf("%s\n",buf);
    }

    abcdk_heap_free(addrs);
}

void test_ifname()
{
    abcdk_ifaddrs_t * addrs = abcdk_heap_alloc(sizeof(abcdk_ifaddrs_t)*10);

    int n = abcdk_ifname_fetch(addrs,10,0);

    for (int i = 0; i < n; i++)
    {
        char addr[100];
        char mark[100];
        char broa[100];
        char mac[100];

        assert(abcdk_inet_ntop(&addrs[i].addr, addr, 100) == addr);
        assert(abcdk_inet_ntop(&addrs[i].mark, mark, 100));

        if(addrs[i].broa.family == ABCDK_IPV4 || addrs[i].broa.family == ABCDK_IPV6)
            assert(abcdk_inet_ntop(&addrs[i].broa, broa, 100));

        printf("Name: %s\n",addrs[i].name);
        printf("Addr: %s\n",addr);
        printf("Mark: %s\n",mark);
        printf("Broa: %s\n",broa);

        assert(abcdk_mac_fetch(addrs[i].name,mac)==mac);

        printf("MAC: %s\n",mac);

    }

    abcdk_heap_free(addrs);
}

void test_connect()
{
    abcdk_sockaddr_t addr={0};

    //int n = abcdk_gethostbyname("www.taobao.com",ABCDK_IPV4,&addr,1,NULL);
    //int n = abcdk_gethostbyname("localhost",ABCDK_IPV4,&addr,1,NULL);
    //abcdk_inet_pton("192.168.100.4",ABCDK_IPV4,&addr);
    //addr.addr4.sin_port = abcdk_endian_h_to_b16(8090);

    assert(abcdk_sockaddr_from_string(&addr,"www.taobao.com:443",1)==0);
  //  assert(abcdk_sockaddr_from_string(&addr,"192.168.100.4:8090",0)==0);
    //assert(abcdk_sockaddr_from_string(&addr,"[www.taobao.com]:443",1)==0);
    //assert(abcdk_sockaddr_from_string(&addr,"[240e:cf:9000:1::3fb]:443",1)==0);
   // assert(abcdk_sockaddr_from_string(&addr,"[240e:cf:9000:1::3fb],443",1)==0);


    char buf[100]={0};
    printf("%s\n",abcdk_sockaddr_to_string(buf,&addr));

    assert(abcdk_sockaddr_where(&addr,1)==0);

    assert(abcdk_sockaddr_where(&addr,2)!=0);

    int s = abcdk_socket(addr.family,0);

  //  abcdk_fflag_add(s,O_NONBLOCK);
    
    assert(abcdk_connect(s,&addr,10000)==0);
    



    abcdk_closep(&s);
}

void test_group()
{
    int s = abcdk_socket(ABCDK_IPV4,1);

    abcdk_sockaddr_t addr={0};
    abcdk_inet_pton("224.0.0.5",ABCDK_IPV4,&addr);
    addr.addr4.sin_port = abcdk_endian_h_to_b16(8090);

    assert(abcdk_socket_option_multicast(s,&addr,NULL,1)==0);

    assert(abcdk_bind(s,&addr)==0);


    abcdk_closep(&s);
}

int main(int argc, char **argv)
{
    //test_gethostbyname();

    //test_ifname();

    test_connect();

  //  test_group();

    return 0;
}