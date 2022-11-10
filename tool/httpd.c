/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

#ifdef HAVE_LIBMAGIC
#include <magic.h>
#endif //

typedef struct _abcdkhttpd
{
    int errcode;
    abcdk_tree_t *args;

    abcdk_comm_t *comm;
    abcdk_comm_node_t *comm_listen;
    SSL_CTX *ssl_ctx;

    int workers;
    const char *server_name;
    const char *root_path;
    const char *listen;
    const char *ca_file;
    const char *ca_path;
    const char *cert_file;
    const char *key_file;

} abcdkhttpd_t;

void _abcdkhttpd_print_usage(abcdk_tree_t *args)
{
}

int _abcdkhttpd_signal_cb(const siginfo_t *info, void *opaque)
{
    if (SI_USER == info->si_code)
        fprintf(stderr, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);
    else
        fprintf(stderr, "signo(%d),errno(%d),code(%d)\n", info->si_signo, info->si_errno, info->si_code);

    if (SIGILL == info->si_signo || SIGTERM == info->si_signo || SIGINT == info->si_signo || SIGQUIT == info->si_signo)
        return -1;
    else
        fprintf(stderr, "如果希望停止服务，按Ctrl+c组合键，或发送SIGTERM(15)信号。例：kill -s 15 %d\n", getpid());

    return 0;
}

void _abcdkhttpd_replay_nobody(abcdk_comm_node_t *node, int status)
{
    abcdk_comm_post_format(node, 100,
                           "HTTP/1.1 %s\r\n"
                           "Connection: Keep-Alive\r\n"
                           "Content-Length: 0\r\n"
                           "\r\n",
                           abcdk_http_status_desc(status));
}

void _abcdkhttpd_replay_file(abcdk_comm_node_t *node,const char *pathfile)
{
    abcdk_object_t *file = NULL;
#ifdef HAVE_LIBMAGIC
    struct magic_set *cookie = NULL;
#endif // HAVE_LIBMAGIC

#ifdef HAVE_LIBMAGIC
    cookie = magic_open(MAGIC_MIME);
    if (!cookie)
    {
        _abcdkhttpd_replay_nobody(node, 500);
        return;
    }
#endif // HAVE_LIBMAGIC

    magic_load(cookie, NULL);

    file = abcdk_mmap2(pathfile, 0, 0, 0);
    if (file)
    {
        abcdk_comm_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Content-Type: %s\r\n"
                               "Content-Length: %lu\r\n"
                               "\r\n",
                               abcdk_http_status_desc(200),
#ifdef HAVE_LIBMAGIC
                               magic_buffer(cookie, file->pptrs[0], file->sizes[0]),
#else  // HAVE_LIBMAGIC
                               abcdk_http_content_type_desc(".*"),
#endif // HAVE_LIBMAGIC
                               file->sizes[0]);

        abcdk_comm_post(node, file);
    }
    else
    {
        _abcdkhttpd_replay_nobody(node, 403);
    }

#ifdef HAVE_LIBMAGIC
    magic_close(cookie);
#endif // HAVE_LIBMAGIC
}

void _abcdkhttpd_request_cb(abcdk_comm_node_t *node, abcdk_http_request_t *req)
{
    abcdkhttpd_t *ctx = NULL;
    const char *p = NULL, *p_next = NULL;
    char pathfile[PATH_MAX] = {0};
    char path[PATH_MAX] = {0};
    size_t path_len = PATH_MAX;
    struct stat attr;
    int chk;

    ctx = (abcdkhttpd_t *)abcdk_comm_get_userdata(node);

    p_next = abcdk_http_request_env(req, 0);
    p = abcdk_strtok(&p_next, " ");
    p = abcdk_strtok(&p_next, " ");

    abcdk_uri_decode(p, p_next - p,path, &path_len);

    abcdk_dirdir(pathfile, ctx->root_path);
    abcdk_dirdir(pathfile, path);

    chk = stat(pathfile, &attr);
    if (chk != 0)
    {
        if (errno == ENOENT)
            _abcdkhttpd_replay_nobody(node, 404);
        else
            _abcdkhttpd_replay_nobody(node, 403);
    }
    else if (S_ISDIR(attr.st_mode))
    {
    }
    else if (S_ISREG(attr.st_mode))
    {
        _abcdkhttpd_replay_file(node,pathfile);
    }
    else
    {
        _abcdkhttpd_replay_nobody(node, 403);
    }
}

void _abcdkhttpd_close_cb(abcdk_comm_node_t *node)
{
}

void _abcdkhttpd_work(abcdkhttpd_t *ctx)
{
    abcdk_signal_t sig;
    abcdk_sockaddr_t addr;
    int chk;

    ctx->workers = abcdk_option_get_int(ctx->args, "--workers", 0, -1);
    ctx->server_name = abcdk_option_get(ctx->args, "--server-name", 0, "abcdk-httpd");
    ctx->root_path = abcdk_option_get(ctx->args, "--root-path", 0, "/tmp/abcdk-httpd/");
    ctx->listen = abcdk_option_get(ctx->args, "--listen", 0, "0.0.0.0:80");
    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);

#ifdef HAVE_OPENSSL

    if (ctx->cert_file && ctx->key_file)
    {
        ctx->ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, ctx->ca_file, ctx->ca_path, 2);
        if (!ctx->ssl_ctx)
        {
            fprintf(stderr, "加载CA证书错误。\n");
            goto final;
        }

        chk = abcdk_openssl_ssl_ctx_load_crt(ctx->ssl_ctx, ctx->cert_file, tx->key_file, NULL);
        if (chk != 0)
        {
            fprintf(stderr, "加载证书或私钥错误。\n");
            goto final;
        }

        SSL_CTX_set_verify(ctx->ssl_ctx, SSL_VERIFY_PEER, NULL);
    }

#endif // HAVE_OPENSSL

    chk = abcdk_sockaddr_from_string(&addr, ctx->listen, 0);
    if (chk != 0)
    {
        fprintf(stderr, "监听地址错误。\n");
        goto final;
    }

    ctx->comm = abcdk_comm_start(ctx->workers, -1);
    if (!ctx->comm)
    {
        fprintf(stderr, "内存错误。\n");
        goto final;
    }

    ctx->comm_listen = abcdk_http_alloc(ctx->comm, 1 << 31 - 1, "/tmp/");
    if (!ctx->comm_listen)
    {
        fprintf(stderr, "内存错误。\n");
        goto final;
    }

    abcdk_comm_set_userdata(ctx->comm_listen, ctx);

    abcdk_http_callback_t cb = {NULL, _abcdkhttpd_request_cb, NULL, _abcdkhttpd_close_cb};
    chk = abcdk_http_listen(ctx->comm_listen, ctx->ssl_ctx, &addr, &cb);
    if (chk != 0)
    {
        fprintf(stderr, "监听错误，也许端口已经被占用。\n");
        goto final;
    }

    /*填充信号及回调函数。*/
    sig.opaque = NULL;
    sig.signal_cb = _abcdkhttpd_signal_cb;
    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    /*等待退出信号。*/
    abcdk_sigwaitinfo(&sig, -1);
    // sleep(30);

final:

    abcdk_comm_unref(&ctx->comm_listen);
    abcdk_comm_stop(&ctx->comm);
#ifdef HAVE_OPENSSL
    abcdk_openssl_ssl_ctx_free(&ctx->ssl_ctx);
#endif // HAVE_OPENSSL
}

int abcdk_tool_httpd(abcdk_tree_t *args)
{
    abcdkhttpd_t ctx;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkhttpd_print_usage(ctx.args);
    }
    else
    {
        _abcdkhttpd_work(&ctx);
    }

    return ctx.errcode;
}
