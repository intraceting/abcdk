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

static void  prepare_cb(void *opaque,abcdk_srpc_session_t **session,abcdk_srpc_session_t *listen)
{
    *session = abcdk_srpc_alloc((abcdk_srpc_t *)opaque);
}

static void accept_cb(void *opaque,abcdk_srpc_session_t *session,int *result)
{
    *result = 0;
}

static void ready_cb(void *opaque,abcdk_srpc_session_t *session)
{

}

static void close_cb(void *opaque,abcdk_srpc_session_t *session)
{
    
}

static void request_cb(void *opaque, abcdk_srpc_session_t *session, uint64_t mid, const void *data, size_t size)
{
    
    
    int a = *((int *)data);

    abcdk_trace_output(LOG_INFO,"mid(%llu),size(%zd),a(%d)",mid,size,a);

    if(a)
        abcdk_srpc_response(session,mid,data,size);
}

static void output_cb(void *opaque, abcdk_srpc_session_t *session)
{

}


int abcdk_test_srpc(abcdk_option_t *args)
{
    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/","test.srpc.log","test.srpc.%d.log",10,10,1,1);

    abcdk_trace_set_log(abcdk_logger_from_trace,log_ctx);

    abcdk_srpc_t *srpc_ctx = abcdk_srpc_create(1000,-1);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_from_string(&addr, "ipv4://127.0.0.1:6666", 0);

    abcdk_srpc_config_t server_cfg = {0},client_cfg = {0};

    server_cfg.accept_cb = accept_cb;
    server_cfg.close_cb = close_cb;
    server_cfg.output_cb = output_cb;
    server_cfg.prepare_cb = prepare_cb;
    server_cfg.ready_cb = ready_cb;
    server_cfg.request_cb = request_cb;
    server_cfg.opaque = srpc_ctx;
    server_cfg.ssl_scheme = abcdk_option_get_int(args,"--ssh-scheme",0,0);
    server_cfg.easyssl_key_file = abcdk_option_get(args,"--easyssl-key-file",0,"");
    server_cfg.easyssl_salt_size = 123;
    server_cfg.openssl_no_check_cert = abcdk_option_get_int(args,"--openssl-check-cert",0,1);
    server_cfg.openssl_cert_file = abcdk_option_get(args,"--server-openssl-cert-file",0,NULL);
    server_cfg.openssl_key_file = abcdk_option_get(args,"--server-openssl-key-file",0,NULL);
    server_cfg.openssl_ca_file = abcdk_option_get(args,"--openssl-ca-file",0,NULL);
    server_cfg.openssl_ca_path = abcdk_option_get(args,"--openssl-ca-path",0,NULL);

    client_cfg = server_cfg;

    client_cfg.openssl_cert_file = abcdk_option_get(args,"--client-openssl-cert-file",0,NULL);
    client_cfg.openssl_key_file = abcdk_option_get(args,"--client-openssl-key-file",0,NULL);


    abcdk_srpc_session_t *listen_p = abcdk_srpc_alloc(srpc_ctx);

    abcdk_srpc_listen(listen_p,&addr,&server_cfg);

    int parallel = abcdk_option_get_int(args,"--parallel",0,1);
    int count = abcdk_option_get_int(args,"--count",0,10000);
    int rand_rsp = abcdk_option_get_int(args,"--rand-rsp",0,1);

//#pragma omp parallel for num_threads(parallel)
    for (int i = 0; i < parallel; i++)
    {
        abcdk_srpc_session_t *client_p = abcdk_srpc_alloc(srpc_ctx);

        abcdk_srpc_connect(client_p,&addr,&client_cfg);

        abcdk_trace_output(LOG_INFO,"thread-%d: begin",i);

        for (int j = 0; j < count; j++)
        {
            char buf[1000] = {0};

            int *a = (int*)buf;
            if(j %3 == 0 || rand_rsp)
                *a = 1;

            int b = ((uint64_t)abcdk_rand_number())%995+5;

            abcdk_rand_string(buf+4,b-4,0);

            abcdk_object_t *rsp = NULL;

            int chk = abcdk_srpc_request(client_p,buf,b,*a?(&rsp):NULL);
            if(chk)
                break;

            if(rsp)
                assert(memcmp(rsp->pptrs[0],buf,b)==0);

            abcdk_object_unref(&rsp);
        }

        abcdk_trace_output(LOG_INFO,"thread-%d: end",i);

        abcdk_srpc_unref(&client_p);

    }

    abcdk_srpc_unref(&listen_p);
    abcdk_srpc_destroy(&srpc_ctx);
    abcdk_logger_close(&log_ctx);
}