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


static void free_cb(void *userdata)
{

}

static void prepare_cb(void *opaque, abcdk_srpc_session_t **session, abcdk_srpc_session_t *listen)
{
    *session = abcdk_srpc_alloc((abcdk_srpc_t *)opaque, 0, free_cb);
}

static void accept_cb(void *opaque, abcdk_srpc_session_t *session, int *result)
{
    *result = 0;
}

static void ready_cb(void *opaque, abcdk_srpc_session_t *session)
{
}

static void close_cb(void *opaque, abcdk_srpc_session_t *session)
{
}

static void request_cb(void *opaque, abcdk_srpc_session_t *session, uint64_t mid, const void *data, size_t size)
{

    int a = *((int *)data);

    abcdk_trace_output(LOG_INFO, "mid(%llu),size(%zd),a(%d)", mid, size, a);

    if (a)
    {
        *((int *)data) = 0;
        abcdk_srpc_response(session, mid, data, size);
    }
}

static void output_cb(void *opaque, abcdk_srpc_session_t *session)
{
}


int abcdk_test_srpc(abcdk_option_t *args)
{
    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/", "test.srpc.log", "test.srpc.%d.log", 10, 10, 1, 1);

    abcdk_trace_set_log(abcdk_logger_from_trace, log_ctx);

    abcdk_srpc_t *srpc_ctx = abcdk_srpc_create(2);

    int role = abcdk_option_get_int(args, "--role", 0, 1);

    abcdk_sockaddr_t addr = {0};

    const char *addr_p;

    abcdk_srpc_config_t cfg = {0};

    cfg.opaque = srpc_ctx;
    cfg.prepare_cb = prepare_cb;
    cfg.accept_cb = accept_cb;
    cfg.ready_cb = ready_cb;
    cfg.request_cb = request_cb;
    cfg.output_cb = output_cb;

    cfg.pki_ca_file = abcdk_option_get(args, "--pki-ca-file", 0, NULL);
    cfg.pki_ca_path = abcdk_option_get(args, "--pki-ca-path", 0, NULL);
    cfg.pki_cert_file = abcdk_option_get(args, "--pki-cert-file", 0, NULL);
    cfg.pki_key_file = abcdk_option_get(args, "--pki-key-file", 0, NULL);
    cfg.pki_check_cert = abcdk_option_get_int(args, "--pki-check-cert", 0, 1);

    cfg.sk_key_file = abcdk_option_get(args, "--sk-key-file", 0, NULL);
    cfg.sk_key_cipher = abcdk_option_get_int(args, "--sk-key-cipher", 0, 1);

    cfg.ssl_scheme = abcdk_option_get_int(args, "--ssl-scheme", 0, ABCDK_STCP_SSL_SCHEME_RAW);

    addr_p = abcdk_option_get(args, "--addr", 0, "ipv4://127.0.0.1:1111");
    abcdk_sockaddr_from_string(&addr, addr_p, 0);

    abcdk_srpc_session_t *session_p = abcdk_srpc_alloc(srpc_ctx, 0, free_cb);

    if (role == 1)
    {
        abcdk_srpc_listen(session_p, &addr, &cfg);

        /*等待终止信号。*/
        abcdk_proc_wait_exit_signal(-1);
    }
    else
    {

        int count = abcdk_option_get_int(args, "--count", 0, 10000);
        int rand_rsp = abcdk_option_get_int(args, "--rand-rsp", 0, 1);

        abcdk_srpc_connect(session_p, &addr, &cfg);

        sleep(2);

        uint64_t s = 0;
        abcdk_clock(s,&s);

//#pragma omp parallel for num_threads(4)
        for (int j = 0; j < count; j++)
        {
            char buf[65000] = {0};

            int *a = (int *)buf;
             *a = 1;
            if (j % 3 == 0 && rand_rsp)
           // if(j>0)
                *a = 0;

            int b = ((uint64_t)abcdk_rand_number()) % 64000 + 5;

            abcdk_rand_string(buf + 4, b - 4, 0);

            abcdk_object_t *rsp = NULL;

            int chk = abcdk_srpc_request(session_p, buf, b, *a ? (&rsp) : NULL);
            //int chk = abcdk_srpc_request(session_p, buf, b,NULL);
            assert (chk == 0);

            if (rsp)
                assert(memcmp(rsp->pptrs[0]+4, buf+4, b-4) == 0);

            abcdk_object_unref(&rsp);
        }

        abcdk_trace_output(LOG_INFO,"cast:%0.6f\n",(double)abcdk_clock(s,&s)/1000000.);

        sleep(20);
    }

    abcdk_srpc_unref(&session_p);
    abcdk_srpc_destroy(&srpc_ctx);
    abcdk_logger_close(&log_ctx);
    
    return 0;
}
