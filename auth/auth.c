/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-auth/auth.h"

int abcdk_auth_add_dmi(abcdk_tree_t *auth,const char *serial,const char * uuid)
{
    int chk;
    assert(auth != NULL);

    if(serial)
    {
        chk = abcdk_option_set2(auth,"--product-serial",serial,1);
        if(chk != 0)
            return -1;
    }

    if(uuid)
    {
        chk = abcdk_option_set2(auth,"--product-uuid",uuid,1);
        if(chk != 0)
            return -2;
    }

    return 0;
}

int abcdk_auth_add_mac(abcdk_tree_t *auth,const char *mac)
{
    int chk;
    assert(auth != NULL);

    if(mac)
    {
        chk = abcdk_option_set2(auth,"--physical-mac",mac,1);
        if(chk != 0)
            return -1;
    }

    return 0;
}

int abcdk_auth_add_valid_period(abcdk_tree_t *auth, uintmax_t days, struct tm *begin)
{
    struct tm b = {0}, e = {0};
    time_t b_sec = 0, e_sec = 0;
    char tmp[100];

    assert(auth != NULL && days > 0);

    if (begin)
        b = *begin;
    else 
        abcdk_time_get(&b,1);

    b_sec = timegm(&b);
    e_sec = b_sec + days * 24 * 60 * 60;

    abcdk_time_sec2tm(&e,e_sec,1);

    memset(tmp,0,sizeof(tmp));
    strftime(tmp,100,"%Y-%m-%dT%H:%M:%SZ",&b);

    abcdk_option_set(auth,"--being-time",tmp);

    memset(tmp,0,sizeof(tmp));
    strftime(tmp,100,"%Y-%m-%dT%H:%M:%SZ",&e);

    abcdk_option_set(auth,"--end-time",tmp);

    return 0;
}

int abcdk_auth_add_valid_period2(abcdk_tree_t *auth, uintmax_t days, uintmax_t delay)
{
    struct tm b = {0};
    time_t b_sec = 0;

    assert(auth != NULL && days > 0);

    abcdk_time_get(&b,1);
    
    b_sec = timegm(&b);
    b_sec += delay * 24 * 60 * 60;

    abcdk_time_sec2tm(&b,b_sec,1);

    return abcdk_auth_add_valid_period(auth,days,&b);
}

int abcdk_auth_add_salt(abcdk_tree_t *auth)
{
    char tmp[100] = {0};

    assert(auth != NULL);

    srand(time(NULL));

    for (int i = 0; i < 99; i++)
        tmp[i] = 32 + rand() % 95;//32~126

    abcdk_option_set(auth,"--",tmp);

    return 0;
}

int abcdk_auth_collect_dmi(abcdk_tree_t *auth)
{
    char tmp[255];
    ssize_t rsize;
    int fd = -1;
    int count = 0;

    assert(auth != NULL);
    
    fd = abcdk_open("/sys/devices/virtual/dmi/id/product_serial", 0, 0, 0);
    if (fd >= 0)
    {
        memset(tmp,0,sizeof(tmp));

        rsize = abcdk_read(fd, tmp, 254);
        if (rsize > 0)
        {
            abcdk_strtrim(tmp,isspace,0);
            abcdk_auth_add_dmi(auth,tmp,NULL);
        }

        count += 1;
        abcdk_closep(&fd);
    }


    fd = abcdk_open("/sys/devices/virtual/dmi/id/product_uuid", 0, 0, 0);
    if (fd >= 0)
    {
        memset(tmp,0,sizeof(tmp));

        rsize = abcdk_read(fd, tmp, 254);
        if (rsize > 0)
        {
            abcdk_strtrim(tmp,isspace,0);
            abcdk_auth_add_dmi(auth,NULL,tmp);
        }

        count += 1;
        abcdk_closep(&fd);
    }

    return ((count > 0) ? 0 : -1);
}

int abcdk_auth_collect_mac(abcdk_tree_t *auth)
{
    int max = 100;
    int count = 0;
    int count2 = 0;
    abcdk_ifaddrs_t *addrs = NULL;
    char tmp[255] = {0};
    char mac[20];

    assert(auth != NULL);

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

        abcdk_auth_add_mac(auth,mac);
        count2 += 1;
    }

final:

    abcdk_heap_free(addrs);

    return ((count > 0) ? 0 : -1);
}

int _abcdk_auth_verify_dmi(abcdk_tree_t *auth,abcdk_tree_t *local)
{
    const char *p1 = NULL, *p2 = NULL;

    p1 = abcdk_option_get(auth, "--product-serial", 0, NULL);
    p2 = abcdk_option_get(local, "--product-serial", 0, NULL);
    if (p1 && p2)
    {
        if (abcdk_strcmp(p1, p2, 1) != 0)
            return -1;
    }

    p1 = abcdk_option_get(auth, "--product-uuid", 0, NULL);
    p2 = abcdk_option_get(local, "--product-uuid", 0, NULL);
    if (p1 && p2)
    {
        if (abcdk_strcmp(p1, p2, 1) != 0)
            return -1;
    }

    return 0;
}

int _abcdk_auth_verify_mac(abcdk_tree_t *auth, abcdk_tree_t *local)
{
    const char *p1 = NULL, *p2 = NULL;
    ssize_t c1 = 0, c2 = 0;

    c1 = abcdk_option_count(auth,"--physical-mac");
    c2 = abcdk_option_count(local,"--physical-mac");

    if (c1 <= 0)
        return 0;

    for (size_t i = 0; i < c1; i++)
    {
        p1 = abcdk_option_get(auth, "--physical-mac", i, NULL);
        if (!p1)
            continue;

        for (size_t j = 0; j < c2; j++)
        {
            p2 = abcdk_option_get(local, "--physical-mac", j, NULL);
            if (!p1)
                continue;

            if (abcdk_strcmp(p1, p2, 0) == 0)
                return 0;
        }
    }

    return -1;
}

int _abcdk_auth_verify_valid_period(abcdk_tree_t *auth)
{
    const char *p1 = NULL, *p2 = NULL;
    struct tm b = {0}, e = {0}, l = {0};
    time_t b_sec = 0, e_sec = 0, l_sec = 0;
    double d = 0.0;

    p1 = abcdk_option_get(auth, "--being-time", 0, "2011-01-01T00:00:00Z");
    p2 = abcdk_option_get(auth, "--end-time", 0, "2110-12-31T23:59:59Z");

    strptime(p1,"%Y-%m-%dT%H:%M:%SZ",&b);
    strptime(p2,"%Y-%m-%dT%H:%M:%SZ",&e);
    abcdk_time_get(&l,1);

    b_sec = timegm(&b);
    e_sec = timegm(&e);
    l_sec = timegm(&l);

    d = difftime(l_sec, b_sec);
    if (d < 0)
        return -2;

    d = difftime(l_sec, e_sec);
    if (d >= 0)
        return -2;

    return 0;
}

int abcdk_auth_verify(abcdk_tree_t *auth)
{
    const char *p1 = NULL, *p2 = NULL;
    abcdk_tree_t *local = NULL;
    int chk = 0;

    assert(auth != NULL);

    local = abcdk_tree_alloc3(1);
    if (!local)
        return -3;

    abcdk_auth_collect_dmi(local);
    abcdk_auth_collect_mac(local);

    chk = _abcdk_auth_verify_dmi(auth,local);
    if (chk != 0)
        goto final;

    chk = _abcdk_auth_verify_mac(auth,local);
    if (chk != 0)
        goto final;

    chk = _abcdk_auth_verify_valid_period(auth);

final:

    abcdk_tree_free(&local);

    return chk;
}

abcdk_allocator_t *abcdk_auth_serialize(abcdk_tree_t *auth)
{
    size_t max = 1024 * 1024;
    abcdk_allocator_t *buf = NULL;

    assert(auth != NULL);

    buf = abcdk_allocator_alloc2(max);
    if(!buf)
        return NULL;

    buf->sizes[0] = abcdk_option_snprintf(buf->pptrs[0],buf->sizes[0],auth,NULL);
    if((ssize_t)buf->sizes[0] <= 0)
        goto final_error;

    return buf;

final_error:

    abcdk_allocator_unref(&buf);

    return NULL;
}

abcdk_tree_t *abcdk_auth_structure(abcdk_allocator_t *plaintext)
{
    abcdk_tree_t *auth = NULL;

    assert(plaintext != NULL);
    
    auth = abcdk_tree_alloc3(1);
    if(!auth)
        return NULL;
    
    abcdk_getargs_text(auth,plaintext->pptrs[0],plaintext->sizes[0],'\n','#',NULL,"--");

    return auth;
}

abcdk_allocator_t *abcdk_auth_encrypt(abcdk_allocator_t *plaintext, uint32_t key)
{
    abcdk_allocator_t *buf = NULL;
    
    assert(plaintext != NULL);

    /*key取反。*/
    key = ~key;

    buf = abcdk_allocator_clone(plaintext);
    if(!buf)
        return NULL;

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 0);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 3, 1);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 1);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 7, 2);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 2);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 2, 1);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 3);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 4, 2);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    return buf;
}

abcdk_allocator_t *abcdk_auth_decrypt(abcdk_allocator_t *ciphertext, uint32_t key)
{
    abcdk_allocator_t *buf = NULL;
    
    assert(ciphertext != NULL);
    
    /*key取反。*/
    key = ~key;

    buf = abcdk_allocator_clone(ciphertext);
    if(!buf)
        return NULL;

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 4, 1);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 3);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 2, 2);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 2);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 7, 1);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 1);

    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    abcdk_cyclic_shift(buf->pptrs[0], buf->sizes[0], 3, 2);

    for (size_t j = 0; j < buf->sizes[0]; j++)
        ABCDK_PTR2U8(buf->pptrs[0], j) ^= ABCDK_PTR2U8(&key, 0);
    
    abcdk_endian_swap(buf->pptrs[0], buf->sizes[0]);

    return buf;
}

int abcdk_auth_save(int fd, const void *auth,size_t len, uint32_t magic)
{
    uint32_t dsize = 0;
    ssize_t chk = 0;

    assert(fd >= 0 && auth != NULL && len > 0);

    /*移动指针到文件末尾。*/
    lseek(fd, 0, SEEK_END);

    /* 转大端字节序存储。*/
    magic = abcdk_endian_h_to_b32(magic);

    chk = abcdk_write(fd, &magic, 4);
    if (chk != 4)
        return -1;

    chk = abcdk_write(fd, auth, len);
    if (chk != len)
        return -1;

    /* 
     * 1: 长度为所有数据的总和。magic[4]+data[N]+size[4]
     * 2: 转大端字节序存储。
     */
    dsize = abcdk_endian_h_to_b32(4 + len + 4);

    chk = abcdk_write(fd, &dsize, 4);
    if (chk != 4)
        return -1;

    return 0;
}

int abcdk_auth_save2(const char *file,const void *auth,size_t len, uint32_t magic)
{
    int fd;
    int chk;

    assert(file != NULL && auth != NULL && len > 0);

    fd = abcdk_open(file, 1, 0, 1);
    if (fd < 0)
        return -1;

    chk = abcdk_auth_save(fd, auth,len,magic);

    abcdk_closep(&fd);

    return chk;
}

abcdk_allocator_t *abcdk_auth_load(int fd, uint32_t magic)
{
    uint32_t magic2 = 0;
    abcdk_allocator_t *auth = NULL;
    uint32_t dsize = 0;
    ssize_t chk = 0;

    assert(fd >= 0);

    /*从末尾开始读。*/
    lseek(fd, -4, SEEK_END);

    chk = abcdk_read(fd, &dsize, 4);
    if (chk != 4)
        goto final_error;

    dsize = abcdk_endian_b_to_h32(dsize);

    /*移动到数据开始的位置。*/
    lseek(fd, -(int32_t)dsize, SEEK_END);

    chk = abcdk_read(fd, &magic2, 4);
    if (chk != 4)
        goto final_error;

    if (abcdk_endian_b_to_h32(magic2) != magic)
        goto final_error;

    auth = abcdk_allocator_alloc2(dsize - 4 - 4);
    if(!auth)
        goto final_error;

    chk = abcdk_read(fd,auth->pptrs[0],auth->sizes[0]);
    if (chk != auth->sizes[0])
        goto final_error;

    return auth;

final_error:

    abcdk_allocator_unref(&auth);

    return NULL;
}

abcdk_allocator_t *abcdk_auth_load2(const char *file,uint32_t magic)
{
    int fd;
    abcdk_allocator_t *auth = NULL;
    
    assert(file != NULL);

    fd = abcdk_open(file, 0, 0, 0);
    if (fd < 0)
        return NULL;

    auth = abcdk_auth_load(fd,magic);

    abcdk_closep(&fd);

    return auth;
}