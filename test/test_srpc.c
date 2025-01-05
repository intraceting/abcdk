/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
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

 //   abcdk_trace_printf(LOG_INFO, "mid(%llu),size(%zd),a(%d)", mid, size, a);

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

    abcdk_trace_printf_set_callback(abcdk_logger_from_trace, log_ctx);

    abcdk_srpc_t *srpc_ctx = abcdk_srpc_create(4,5);

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

    const char *pki_cert_file = abcdk_option_get(args, "--pki-cert-file", 0, NULL);
    const char *pki_key_file = abcdk_option_get(args, "--pki-key-file", 0, NULL);
    const char *rsa_key_file = abcdk_option_get(args, "--rsa-key-file", 0, NULL);

#ifdef HAVE_OPENSSL

    if(pki_cert_file)
        cfg.pki_use_cert = abcdk_openssl_cert_load(pki_cert_file);

    if(pki_key_file)
        cfg.pki_use_key = abcdk_openssl_evp_pkey_load(pki_key_file,0,NULL);

    if(rsa_key_file)
        cfg.rsa_use_key = abcdk_openssl_rsa_load(rsa_key_file,!role,NULL);

#endif //HAVE_OPENSSL

    cfg.ssl_scheme = abcdk_option_get_int(args, "--ssl-scheme", 0, ABCDK_STCP_SSL_SCHEME_RAW);

    addr_p = abcdk_option_get(args, "--addr", 0, "ipv4://127.0.0.1:1111");
    abcdk_sockaddr_from_string(&addr, addr_p, 0);

    abcdk_srpc_session_t *session_p = abcdk_srpc_alloc(srpc_ctx, 0, free_cb);

    if (role == 1)
    {
        cfg.bind_addr = addr;
        abcdk_srpc_listen(session_p, &cfg);

        /*等待终止信号。*/
        abcdk_proc_wait_exit_signal(-1);
    }
    else
    {

        int count = abcdk_option_get_int(args, "--count", 0, 10000);
        int rand_rsp = abcdk_option_get_int(args, "--rand-rsp", 0, 0);

        abcdk_srpc_connect(session_p, &addr, &cfg);

        sleep(2);

        uint64_t s = 0;
        abcdk_clock(s,&s);

//#pragma omp parallel for num_threads(2)
        for (int j = 0; j < count; j++)
        {
            char buf[65000] = {0};

            int *a = (int *)buf;
             *a = 1;
             if (j % 3 == 0 && rand_rsp)
                 *a = 0;

         //   abcdk_trace_printf(LOG_INFO, "%d,step(%.09f)",__LINE__, (double)abcdk_clock(s,&s)/1000000000.);

            int b = abcdk_rand(5, 64000);
          //  int b = 1000;

        //    abcdk_trace_printf(LOG_INFO, "%d,step(%.09f)",__LINE__, (double)abcdk_clock(s,&s)/1000000000.);

#ifdef HAVE_OPENSSL
        RAND_bytes(buf + 4, b - 4);
#else 
        abcdk_rand_bytes(buf + 4, b - 4, 5);
#endif 

         //   abcdk_trace_printf(LOG_INFO, "%d,step(%.09f)",__LINE__, (double)abcdk_clock(s,&s)/1000000000.);

            abcdk_object_t *rsp = NULL;

           // int chk = abcdk_srpc_request(session_p, buf, b, *a ? (&rsp) : NULL);
            int chk = abcdk_srpc_request(session_p, buf, b,NULL);
            assert (chk == 0);

     //       abcdk_trace_printf(LOG_INFO, "%d,step(%.09f)",__LINE__, (double)abcdk_clock(s,&s)/1000000000.);

            if (rsp)
                assert(memcmp(rsp->pptrs[0]+4, buf+4, b-4) == 0);

       //     abcdk_trace_printf(LOG_INFO, "%d,step(%.09f)",__LINE__, (double)abcdk_clock(s,&s)/1000000000.);

            abcdk_object_unref(&rsp);

            usleep(20);
        }

        abcdk_trace_printf(LOG_INFO,"cast:%0.6f\n",(double)abcdk_clock(s,&s)/1000000.);

        sleep(20);
    }

    abcdk_srpc_unref(&session_p);
    abcdk_srpc_stop(srpc_ctx);
    abcdk_srpc_destroy(&srpc_ctx);
    
#ifdef HAVE_OPENSSL
    abcdk_openssl_rsa_free(&cfg.rsa_use_key);
    abcdk_openssl_x509_free(&cfg.pki_use_cert);
    abcdk_openssl_evp_pkey_free(&cfg.pki_use_key);
#endif //HAVE_OPENSSL

    abcdk_logger_close(&log_ctx);
    
    return 0;
}
