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
#include <locale.h>
#include "entry.h"

static abcdk_sudp_t *g_ctx = NULL;

static void input_cb(void *opaque,abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    char addrbuf[100] = {0};
    int len;
    int flag;

    abcdk_sockaddr_to_string(addrbuf,remote,0);

    len = (uint16_t)abcdk_bloom_read_number((uint8_t*)data,size,0,16);
    flag =(uint8_t)abcdk_bloom_read_number((uint8_t*)data,size,16,8);

    assert(len +3 == size);

  //  abcdk_trace_output(LOG_DEBUG,"remote(%s),len=%d\n",addrbuf,len);

    if(!flag)
        return;

    abcdk_object_t *rsp_p = abcdk_object_alloc2(len+3);

    abcdk_bloom_write_number(rsp_p->pptrs[0],3,0,16,len);
    abcdk_bloom_write_number(rsp_p->pptrs[0],3,16,8,0);
    memcpy(rsp_p->pptrs[0]+3,ABCDK_PTR2VPTR(data,3),len);

    abcdk_sudp_post_buffer(g_ctx,remote,rsp_p->pptrs[0],rsp_p->sizes[0]);

    abcdk_object_unref(&rsp_p);
}

int abcdk_test_sudp(abcdk_option_t *args)
{
    const char *key_p = abcdk_option_get(args, "--key", 0, NULL);
    const char *listen_p = abcdk_option_get(args, "--listen", 0, "0.0.0.0:1111");
    const char *listen_mreq_p = abcdk_option_get(args, "--listen-mreq", 0, NULL);
    const char *dst_p = abcdk_option_get(args, "--dst", 0, "127.0.0.1:1111");

    abcdk_sockaddr_t remote = {0};
    abcdk_sockaddr_from_string(&remote,dst_p,1);

    abcdk_sudp_config_t cfg = {0};
    
    abcdk_sockaddr_from_string(&cfg.listen_addr,listen_p,0);

    if(listen_mreq_p)
        cfg.mreq_enable = !abcdk_mreqaddr_from_string(&cfg.mreq_addr,listen_mreq_p,"0.0.0.0");

    cfg.input_cb = input_cb;

    g_ctx = abcdk_sudp_create(&cfg);

    if(key_p)
        abcdk_sudp_cipher_reset(g_ctx,(uint8_t*)key_p,strlen(key_p));

    abcdk_object_t *data = abcdk_object_alloc2(64512);


    for(int i = 0;i<100000;i++)
    {
        int k=rand()%64512;
        int len = ABCDK_CLAMP(k,1,64512-3);
        

        abcdk_bloom_write_number(data->pptrs[0],3,0,16,len);
        abcdk_bloom_write_number(data->pptrs[0],3,16,8,1);

#ifndef OPENSSL_VERSION_NUMBER
        RAND_bytes(data->pptrs[0]+3,len);
#else 
        abcdk_rand_bytes(data->pptrs[0]+3,len,2);
#endif //#ifdef OPENSSL_VERSION_NUMBER

        data->sizes[0] = len+3;

        if(remote.family)
            abcdk_sudp_post_buffer(g_ctx,&remote,data->pptrs[0],data->sizes[0]);

        usleep(200);
    }

    while(getchar() != 'q');

    abcdk_object_unref(&data);

    abcdk_sudp_stop(g_ctx);
    abcdk_sudp_destroy(&g_ctx);

    return 0;
}
