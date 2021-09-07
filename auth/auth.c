/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-auth/auth.h"

int abcdk_auth_collect_dmi(abcdk_tree_t *opt)
{
    char tmp[255];
    ssize_t rsize;
    int fd = -1;

    assert(opt != NULL);
    
    fd = abcdk_open("/sys/devices/virtual/dmi/id/product_serial", 0, 0, 0);
    if (fd >= 0)
    {
        memset(tmp,0,sizeof(tmp));

        rsize = abcdk_read(fd, tmp, 254);
        if (rsize > 0)
            abcdk_option_set(opt,"--system-serial-number",tmp);

        abcdk_closep(&fd);
    }


    fd = abcdk_open("/sys/devices/virtual/dmi/id/product_uuid", 0, 0, 0);
    if (fd >= 0)
    {
        memset(tmp,0,sizeof(tmp));

        rsize = abcdk_read(fd, tmp, 254);
        if (rsize > 0)
            abcdk_option_set(opt,"--system-uuid",tmp);

        abcdk_closep(&fd);
    }

    return 0;
}

int abcdk_auth_collect_mac(abcdk_tree_t *opt)
{
    int max = 100;
    int count = 0;
    abcdk_ifaddrs_t *addrs = NULL;
    char tmp[255] = {0};
    char mac[20];

    assert(opt != NULL);

    addrs = abcdk_heap_alloc(sizeof(abcdk_ifaddrs_t)*max);
    if(!addrs)
        return -1;

    count = abcdk_ifname_fetch(addrs, max, 1);
    if (count <= 0)
        goto final;

    for (int i = 0; i < count; i++)
    {
        memset(tmp,0,sizeof(tmp));

        /*虚拟网卡会在这个目录存在相同名字的目录。*/
        abcdk_dirdir(tmp,"/sys/devices/virtual/net/");
        abcdk_dirdir(tmp,addrs[i].name);

        /*跳过虚拟网卡。*/
        if(access(tmp,F_OK)==0)
            continue;

        memset(mac,0,sizeof(mac));
        abcdk_mac_fetch(addrs[i].name,mac);
        
        /*跳过无效地址。*/
        if(mac[0] == '\0')
            continue;

        abcdk_option_set2(opt,"--network-interface-address",mac,1);
    }

final:

    abcdk_heap_free(addrs);

    return 0;
}

int abcdk_auth_make_valid_period(abcdk_tree_t *opt, uintmax_t days, struct tm *begin)
{
    struct tm b = {0}, e = {0};
    time_t b_sec = 0, e_sec = 0;
    char tmp[100];

    assert(opt != NULL && days > 0);

    if (begin)
        b = *begin;
    else 
        abcdk_time_get(&b,1);

    b_sec = timegm(&b);
    e_sec = b_sec + days * 24 * 60 * 60;

    abcdk_sec2time(&e,e_sec,1);

    memset(tmp,0,sizeof(tmp));
    strftime(tmp,100,"%Y-%m-%dT%H:%M:%SZ",&b);

    abcdk_option_set(opt,"--being-time",tmp);

    memset(tmp,0,sizeof(tmp));
    strftime(tmp,100,"%Y-%m-%dT%H:%M:%SZ",&e);

    abcdk_option_set(opt,"--end-time",tmp);

    return 0;
}