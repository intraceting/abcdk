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

    abcdk_sockaddr_to_string(addrbuf,remote,0);

    len = (uint16_t)abcdk_bloom_read_number((uint8_t*)data,size,0,16);

    assert(len <= size-2);

 //   abcdk_trace_output(LOG_DEBUG,"remote(%s),len=%d\n",addrbuf,len);

     
}

int abcdk_test_sudp(abcdk_option_t *args)
{
    const char *listen_p = abcdk_option_get(args, "--listen", 0, "0.0.0.0:1111");
    const char *dst_p = abcdk_option_get(args, "--dst", 0, "127.0.0.1:1111");

    abcdk_sockaddr_t remote = {0};
    abcdk_sockaddr_from_string(&remote,dst_p,1);

    abcdk_sudp_config_t cfg = {0};
    
    cfg.aes_key_file = "";
    abcdk_sockaddr_from_string(&cfg.listen,listen_p,0);

    cfg.input_cb = input_cb;

    g_ctx = abcdk_sudp_start(&cfg);

    abcdk_object_t *data = abcdk_object_alloc2(64512);

    for(int i = 0;i<100000;i++)
    {
        int k=rand()%64512;
        int len = ABCDK_CLAMP(k,1,64512-2);
        abcdk_rand_string(data->pptrs[0]+2,len,0);

        abcdk_bloom_write_number(data->pptrs[0],2,0,16,len);
        data->sizes[0] = len+2;

        if(remote.family)
            abcdk_sudp_post_buffer(g_ctx,&remote,data->pptrs[0],data->sizes[0]);

       // usleep(20*1000);
    }

    while(getchar() != 'q');

    abcdk_object_unref(&data);

    abcdk_sudp_stop(&g_ctx);
}
