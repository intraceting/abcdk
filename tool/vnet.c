/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"


typedef struct _abcdkvnet
{
    int errcode;
    abcdk_option_t *args;

    /*CA证书。*/
    const char *ca_file;

    /*CA证书目录。*/
    const char *ca_path;

    /*服务器证书。*/
    const char *cert_file;

    /*服务器证书私钥。*/
    const char *key_file;

    /*日志。*/
    abcdk_logger_t *logger;

    abcdk_rpc_t *io_ctx;
    abcdk_rpc_session_t *listen_p;
    abcdk_rpc_session_t *listen_ssl_p;

}abcdkvnet_t;


void _abcdkvnet_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的HTTP服务器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径。默认：/tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒)，1～60之间有效。默认：30\n");
    fprintf(stderr, "\t\t注：此功能不支持supervisor或类似的工具。\n");
}


static void _abcdkvnet_prepare_cb(void *opaque, abcdk_rpc_session_t **session, abcdk_rpc_session_t *listen)
{
    *session = abcdk_rpc_alloc(((abcdkvnet_t *)opaque)->io_ctx);
}

static void _abcdkvnet_accept_cb(void *opaque, abcdk_rpc_session_t *session, int *result)
{
    *result = 0;
}

static void _abcdkvnet_ready_cb(void *opaque, abcdk_rpc_session_t *session)
{
    abcdk_rpc_set_timeout(session, 30);
}

static void _abcdkvnet_close_cb(void *opaque, abcdk_rpc_session_t *session)
{
}

static void _abcdkvnet_request_cb(void *opaque, abcdk_rpc_session_t *session, uint64_t mid, const void *data, size_t size)
{

}

static int _abcdkhttpd_start_listen(abcdkvnet_t *ctx,int ssl)
{
    const char *listen;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_rpc_config_t cfg = {0};
    abcdk_rpc_session_t *listen_p;
    int chk;
    
    if (ssl)
        listen = abcdk_option_get(ctx->args, "--listen-ssl", 0, NULL);
    else
        listen = abcdk_option_get(ctx->args, "--listen", 0, NULL);

    /*未启用。*/
    if(!listen)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'无法识别。", listen);
        return -1;
    }

    cfg.opaque = ctx;
    cfg.prepare_cb = _abcdkvnet_prepare_cb;
    cfg.accept_cb = _abcdkvnet_accept_cb;
    cfg.ready_cb = _abcdkvnet_ready_cb;
    cfg.close_cb = _abcdkvnet_close_cb;
    cfg.request_cb = _abcdkvnet_request_cb;

    if (ssl)
    {
        cfg.ca_file = ctx->ca_file;
        cfg.ca_path = ctx->ca_path;
        cfg.cert_file = ctx->cert_file;
        cfg.key_file = ctx->key_file;
    }

    if (ssl)
        listen_p = ctx->listen_ssl_p = abcdk_rpc_alloc(ctx->io_ctx);
    else 
        listen_p = ctx->listen_p = abcdk_rpc_alloc(ctx->io_ctx);

    if (!listen_p)
    {
        abcdk_trace_output(LOG_ERR, "内部错误。");
        return -2;
    }

    chk = abcdk_rpc_listen(listen_p,&listen_addr,&cfg);
    if(chk == 0)
        return 0;

    return -3;

}


static void _abcdkvnet_process(abcdkvnet_t *ctx)
{
    int max_client = 1000;
    const char *log_path = NULL;
    int chk;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "vnet.log", "vnet.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

#ifdef HEADER_SSL_H
    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);
#endif // HEADER_SSL_H

    ctx->io_ctx = abcdk_rpc_create(256, -1);
    if (!ctx->io_ctx)
    {
        abcdk_trace_output(LOG_WARNING, "内存错误。\n");
        goto final;
    }

    chk = _abcdkhttpd_start_listen(ctx,0);
    if(chk != 0)
        goto final;

    chk = _abcdkhttpd_start_listen(ctx,1);
    if(chk != 0)
        goto final;


final:

    abcdk_rpc_destroy(&ctx->io_ctx);
    abcdk_rpc_unref(&ctx->listen_p);
    abcdk_rpc_unref(&ctx->listen_ssl_p);

    abcdk_trace_output(LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}


static int _abcdkvnet_daemon_process_cb(void *opaque)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;

    _abcdkvnet_process(ctx);

    return 0;
}

static void _abcdkvnet_daemon(abcdkvnet_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path, "vnet-daemon.log", "vent-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, logger);

    abcdk_proc_daemon(interval, _abcdkvnet_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_vnet(abcdk_option_t *args)
{
    abcdkvnet_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkvnet_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdkvnet_daemon(&ctx);
        }
        else
        {
            _abcdkvnet_process(&ctx);
        }
    }

    return 0;
}

