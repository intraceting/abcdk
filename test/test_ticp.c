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

abcdk_tipc_t * g_ctx = NULL;

static void _abcdk_test_tipc_request_cb(void *opaque, uint64_t id, uint64_t mid, const void *data, size_t size)
{
    abcdk_trace_output(LOG_INFO,"id=%llu,mid=%llu,size=%zu \n",id,mid,size);

    if( ABCDK_PTR2I8(data,0) == 'r')
        abcdk_tipc_response(g_ctx,id,mid,data,size);
}

int abcdk_test_tipc(abcdk_option_t *args)
{
    abcdk_tipc_config_t cfg = {0};
    const char *listen_p;
    const char *connect_p;

    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/","test.tipc.log","test.tipc.%d.log",10,10,1,1);

    abcdk_trace_set_log(abcdk_logger_from_trace,log_ctx);

    cfg.opaque = 
    cfg.id = abcdk_option_get_llong(args,"--id",0,1);
    cfg.request_cb = _abcdk_test_tipc_request_cb;

    listen_p = abcdk_option_get(args,"--listen",0,"ipv4://127.0.0.1:6666");
    connect_p = abcdk_option_get(args,"--connect",0,NULL);

    g_ctx = abcdk_tipc_create(&cfg);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_from_string(&addr, listen_p, 0);
    abcdk_tipc_listen(g_ctx,&addr);

    sleep(3);

    if(connect_p)
    {
        abcdk_tipc_connect(g_ctx,connect_p,1);

        for(int i = 0;i<1000000;i++)
        {
            abcdk_object_t *rsp_p = NULL;
            char buf[100] = {0};
            sprintf(buf,"%caaaaaa",7,(i%3==0?'r':'a'));

            abcdk_tipc_request(g_ctx,1,buf,7,(buf[0]=='c'?&rsp_p:NULL));
            abcdk_object_unref(&rsp_p);
        }
    }

    abcdk_proc_wait_exit_signal(-1);

    abcdk_tipc_destroy(&g_ctx);

    abcdk_logger_close(&log_ctx);

    return 0;
}