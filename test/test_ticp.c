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

static void _abcdk_test_tipc_request_cb(void *opaque, uint64_t id, uint64_t mid, const void *data, size_t size)
{
    abcdk_trace_output(LOG_INFO,"id=%llu,mid=%llu,size=%zu \n",id,mid,size);
}

int abcdk_test_tipc(abcdk_option_t *args)
{
    abcdk_tipc_config_t cfg = {0};
    const char *listen_p;
    const char *connect_p;

    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/","test.tipc.log","test.tipc.%d.log",10,10,1,1);

    abcdk_trace_set_log(abcdk_logger_from_trace,log_ctx);


    cfg.id = abcdk_option_get_llong(args,"--id",0,1);
    cfg.request_cb = _abcdk_test_tipc_request_cb;

    listen_p = abcdk_option_get(args,"--listen",0,"ipv4://127.0.0.1:6666");
    connect_p = abcdk_option_get(args,"--connect",0,NULL);

    abcdk_tipc_t * ctx = abcdk_tipc_create(&cfg);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_from_string(&addr, listen_p, 0);
    abcdk_tipc_listen(ctx,&addr);

    sleep(3);

    if(connect_p)
    {
        abcdk_tipc_connect(ctx,connect_p,1);

        for(int i = 0;i<1000000;i++)
            abcdk_tipc_request(ctx,1,"aaaaaaa",7,NULL);
    }

    abcdk_proc_wait_exit_signal(-1);

    abcdk_tipc_destroy(&ctx);

    abcdk_logger_close(&log_ctx);

    return 0;
}