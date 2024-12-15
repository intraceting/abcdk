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

static void close_cb(abcdk_sudp_node_t *node)
{
    int *id_p = (int *)abcdk_sudp_get_userdata(node);

    fprintf(stderr, "close(%d)\n", *id_p);
}

static void input_cb(abcdk_sudp_node_t *node, abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    char addrbuf[100] = {0};
    int len;
    int flag;

    abcdk_sockaddr_to_string(addrbuf, remote, 0);

    len = (uint16_t)abcdk_bloom_read_number((uint8_t *)data, size, 0, 16);
    flag = (uint8_t)abcdk_bloom_read_number((uint8_t *)data, size, 16, 8);

    assert(len + 3 == size);

    //  abcdk_trace_output(LOG_DEBUG,"remote(%s),len=%d\n",addrbuf,len);

    if (!flag)
        return;

    abcdk_object_t *rsp_p = abcdk_object_alloc2(len + 3);

    abcdk_bloom_write_number(rsp_p->pptrs[0], 3, 0, 16, len);
    abcdk_bloom_write_number(rsp_p->pptrs[0], 3, 16, 8, 0);
    memcpy(rsp_p->pptrs[0] + 3, ABCDK_PTR2VPTR(data, 3), len);

    abcdk_sudp_post(node, remote, rsp_p->pptrs[0], rsp_p->sizes[0]);

    abcdk_object_unref(&rsp_p);
}

static void free_cb(void *userdata)
{

}

int abcdk_test_sudp(abcdk_option_t *args)
{
    const char *listen_p[2] = {0};
    const char *dst_p[2] = {0};

    int worker = abcdk_option_get_int(args,"--worker",0,1);

    const char *key_p = abcdk_option_get(args, "--key", 0, NULL);

    listen_p[0] = abcdk_option_get(args, "--listen", 0, "0.0.0.0:1111");
    listen_p[1] = abcdk_option_get(args, "--listen", 1, "0.0.0.0:2222");
    dst_p[0] = abcdk_option_get(args, "--dst", 0, "");
    dst_p[1] = abcdk_option_get(args, "--dst", 1, "");

    abcdk_sockaddr_t dst[2] = {0};

    abcdk_sockaddr_from_string(&dst[0], dst_p[0], 1);
    abcdk_sockaddr_from_string(&dst[1], dst_p[1], 1);

    abcdk_sudp_config_t cfg[2] = {0};

    abcdk_sockaddr_from_string(&cfg[0].bind_addr, listen_p[0], 0);
    abcdk_sockaddr_from_string(&cfg[1].bind_addr, listen_p[1], 0);

    cfg[0].close_cb = close_cb;
    cfg[0].input_cb = input_cb;
    cfg[0].ssl_scheme = key_p?ABCDK_SUDP_SSL_SCHEME_AES256GCM:ABCDK_SUDP_SSL_SCHEME_RAW;
    
    cfg[1].close_cb = close_cb;
    cfg[1].input_cb = input_cb;
    cfg[1].ssl_scheme = key_p?ABCDK_SUDP_SSL_SCHEME_AES256GCM:ABCDK_SUDP_SSL_SCHEME_RAW;

    abcdk_sudp_t *ctx = abcdk_sudp_create(worker,5);

#pragma omp parallel for num_threads(2)
    for (int j = 0; j < 2; j++)
    {
        abcdk_sudp_node_t *node = abcdk_sudp_alloc(ctx, sizeof(int), free_cb);

        int *id_p = (int *)abcdk_sudp_get_userdata(node);
        *id_p = j + 1;

        int chk = abcdk_sudp_enroll(node,&cfg[j]);
        assert(chk == 0);

     //   abcdk_sudp_set_timeout(node,10);

        if (key_p)
            abcdk_sudp_cipher_reset(node, (uint8_t *)key_p, strlen(key_p), 0x01 | 0x02);

 //#pragma omp parallel for num_threads(2)
        for (int i = 0; i < 10; i++)
        {
            abcdk_object_t *data = abcdk_object_alloc2(64512);
            //int k = rand() % 64512;
            int k = 1400;
            int len = ABCDK_CLAMP(k, 1, 64512 - 3);

            abcdk_bloom_write_number(data->pptrs[0], 3, 0, 16, len);
            abcdk_bloom_write_number(data->pptrs[0], 3, 16, 8, 1);

#if 0
#ifdef OPENSSL_VERSION_NUMBER
            RAND_bytes(data->pptrs[0] + 3, len);
#else
            abcdk_rand_bytes(data->pptrs[0] + 3, len, 2);
#endif // #ifdef OPENSSL_VERSION_NUMBER
#endif

            data->sizes[0] = len + 3;

            if (dst[j].family)
                abcdk_sudp_post(node, &dst[j], data->pptrs[0], data->sizes[0]);

            abcdk_object_unref(&data);

           // usleep(1);
        }

        abcdk_sudp_unref(&node);
    }

    fprintf(stderr,"Press the q key to exit.\n");

    while (getchar() != 'q')
        ;

    abcdk_sudp_stop(ctx);
    abcdk_sudp_destroy(&ctx);

    return 0;
}
