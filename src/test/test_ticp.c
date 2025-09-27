/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

abcdk_tipc_t * g_ctx = NULL;

static void _abcdk_test_tipc_offline_cb(void *opaque, uint64_t id)
{
    abcdk_trace_printf(LOG_INFO,"id=%llu 离线。",id);
}

static void _abcdk_test_tipc_request_cb(void *opaque, uint64_t id, uint64_t mid, const void *data, size_t size)
{
    abcdk_trace_printf(LOG_INFO,"id=%llu,mid=%llu,size=%zu,data=%s \n",id,mid,size,data);

    if( ABCDK_PTR2I8(data,0) == 'r')
        abcdk_tipc_response(g_ctx,id,mid,data,size);
}

static void _abcdk_test_tipc_subscribe_cb(void *opaque, uint64_t id, uint64_t topic, const void *data, size_t size)
{
    abcdk_trace_printf(LOG_INFO,"id=%llu,topic=%llu,size=%zu \n",id,topic,size);

}

int abcdk_test_tipc(abcdk_option_t *args)
{
    abcdk_tipc_config_t cfg = {0};
    const char *listen_p;
    const char *connect_p;

    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/","test.tipc.log","test.tipc.%d.log",10,10,1,1);

    abcdk_trace_printf_redirect(abcdk_logger_proxy,log_ctx);

    cfg.opaque = NULL;
    cfg.id = abcdk_option_get_llong(args,"--id",0,1);
    cfg.request_cb = _abcdk_test_tipc_request_cb;
    cfg.offline_cb = _abcdk_test_tipc_offline_cb;
    cfg.subscribe_cb = _abcdk_test_tipc_subscribe_cb;
    cfg.ssl_scheme = abcdk_option_get_int(args,"--ssh-scheme",0,0);

    // cfg.ske_key_file = abcdk_option_get(args,"--ske-key-file",0,NULL);
    // cfg.pki_check_cert = abcdk_option_get_int(args,"--pki-check-cert",0,1);
    // cfg.pki_cert_file = abcdk_option_get(args,"--pki-cert-file",0,NULL);
    // cfg.pki_key_file = abcdk_option_get(args,"--pki-key-file",0,NULL);
    cfg.pki_ca_file = abcdk_option_get(args,"--pki-ca-file",0,NULL);
    cfg.pki_ca_path = abcdk_option_get(args,"--pki-ca-path",0,NULL);

    listen_p = abcdk_option_get(args,"--listen",0,"ipv4://127.0.0.1:6666");
    connect_p = abcdk_option_get(args,"--connect",0,NULL);

    uint64_t id2 = abcdk_option_get_llong(args,"--id2",0,2);

    g_ctx = abcdk_tipc_create(&cfg);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_from_string(&addr, listen_p, 0);
    abcdk_tipc_listen(g_ctx,&addr);

    sleep(10);


    abcdk_tipc_subscribe(g_ctx,1,0);
    abcdk_tipc_subscribe(g_ctx,2,0);

    if(connect_p)
    {
        abcdk_tipc_connect(g_ctx,connect_p,id2);

        sleep(1);

        size_t buf_l = 1920*1080*3;
        char *buf_p = (char*)abcdk_heap_alloc(buf_l);
        for(int i = 0;i<100;i++)
        {
            abcdk_object_t *rsp_p = NULL;
            sprintf(buf_p,"%caaaaaa",(i%3==0?'r':'a'));
            //sprintf(buf_p,"raaaaaa");

            size_t buf_l2 = abcdk_rand(2, buf_l);

            abcdk_tipc_request(g_ctx,id2,buf_p,buf_l2,(buf_p[0]=='r'?&rsp_p:NULL));

            if(rsp_p)
                fprintf(stderr,"%s\n",rsp_p->pstrs[0]);

            abcdk_object_unref(&rsp_p);
        }

        abcdk_heap_free(buf_p);
    }


     sleep(1);

    for(int i = 0;i<100;i++)
    {
        abcdk_tipc_publish(g_ctx,1,"bbbbbb",6);
        abcdk_tipc_publish(g_ctx,2,"bbbbbbb",7);

        usleep(1000*100);
    }

     sleep(35);

    for(int i = 0;i<100;i++)
    {
        abcdk_tipc_publish(g_ctx,1,"bbbbbb",6);
        abcdk_tipc_publish(g_ctx,2,"bbbbbbb",7);

        usleep(1000*100);
    }

    abcdk_proc_wait_exit_signal(-1);

    abcdk_tipc_destroy(&g_ctx);

    abcdk_logger_close(&log_ctx);

    return 0;
}