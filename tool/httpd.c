/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

#ifdef HAVE_LIBMAGIC
#include <magic.h>
#endif // HAVE_LIBMAGIC

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

typedef struct _abcdkhttpd_node
{
    abcdkhttpd_t *ctx;

    const char *timefmt;

#ifdef HAVE_LIBMAGIC
    struct magic_set *magic_handle;
#endif // HAVE_LIBMAGIC

    const char *line0;
    const char *referer;
    const char *user_agent;
    const char *range;

    char method[100];
    char location[PATH_MAX];
    char path[PATH_MAX];
    char params[PATH_MAX];
    char version[100];

    char pathfile[PATH_MAX];
    struct stat attr;

} abcdkhttpd_node_t;

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

void _abcdkhttpd_logprint(abcdk_comm_node_t *node, int status, size_t size)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    fprintf(stderr, "\"%s\" %d %ld \"%s\" \"%s\" \n",
            http_p->line0, status, (ssize_t)size,
            http_p->referer ? http_p->referer : "-",
            http_p->user_agent ? http_p->user_agent : "-");
}

void _abcdkhttpd_replay_option(abcdk_comm_node_t *node, int status)
{
    abcdkhttpd_node_t *http_p;
    struct tm tm;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    abcdk_time_get(&tm, 1);

    abcdk_comm_post_format(node, 300,
                           "HTTP/1.1 %s\r\n"
                           "Server: abcdk\r\n"
                           "Data: %s\r\n"
                           "Connection: Keep-Alive\r\n"
                           "Content-Length: 0\r\n"
                           "Access-Control-Allow-Origin: *\r\n"
                           "Access-Control-Allow-Methods: POST,GET,HEAD,OPTIONS\r\n"
                           "Access-Control-Allow-Headers: *\r\n"
                           "Access-Control-Allow-Age: 3600\r\n"
                           "\r\n",
                           abcdk_http_status_desc(status),
                           abcdk_time_format(http_p->timefmt, &tm));

    _abcdkhttpd_logprint(node, status, 0);
}

void _abcdkhttpd_replay_nobody(abcdk_comm_node_t *node, int status)
{
    abcdkhttpd_node_t *http_p;
    struct tm tm;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    abcdk_time_get(&tm, 1);

    abcdk_comm_post_format(node, 300,
                           "HTTP/1.1 %s\r\n"
                           "Server: abcdk\r\n"
                           "Data: %s\r\n"
                           "Connection: Keep-Alive\r\n"
                           "Content-Length: 0\r\n"
                           "\r\n",
                           abcdk_http_status_desc(status),
                           abcdk_time_format(http_p->timefmt, &tm));

    _abcdkhttpd_logprint(node, status, 0);
}

void _abcdkhttpd_replay_chunked(abcdk_comm_node_t *node, const void *data, size_t size)
{
    abcdk_comm_post_format(node, 20, "%x\r\n", size);
    if (size > 0)
        abcdk_comm_post_buffer(node, data, size);
    abcdk_comm_post_buffer(node, "\r\n", 2);
}

void _abcdkhttpd_replay_dirent(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    abcdk_tree_t *dir;
    char tmp[PATH_MAX], tmp2[PATH_MAX], tmp3[NAME_MAX];
    size_t path_len = PATH_MAX;
    abcdk_object_t *buf;
    struct stat attr;
    struct tm tm;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    dir = abcdk_tree_alloc3(1);
    buf = abcdk_object_alloc2(10000);

    if (!dir || !buf)
    {
        _abcdkhttpd_replay_nobody(node, 500);
        goto final;
    }

    chk = abcdk_dirent_open(dir, http_p->pathfile);
    if (chk != 0)
    {
        _abcdkhttpd_replay_nobody(node, 403);
    }
    else
    {
        abcdk_comm_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: abcdk\r\n"
                               "Data: %s\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Content-Type: %s; charset=utf-8\r\n"
                               "Transfer-Encoding: chunked\r\n"
                               "\r\n",
                               abcdk_http_status_desc(200),
                               abcdk_time_format(http_p->timefmt, &tm),
                               abcdk_http_content_type_desc(".html"));

        chk = snprintf(buf->pstrs[0], buf->sizes[0],
                       "<!DOCTYPE html>\r\n"
                       "<html>\r\n"
                       "<head><title>Index of %s</title></head>\r\n"
                       "<body bgcolor=\"white\">\r\n"
                       "<h1>Index of %s</h1>\r\n"
                       "<hr>\r\n"
                       "<pre>\r\n"
                       "<a href=\"../\">../</a>\r\n",
                       http_p->path,http_p->path);

        _abcdkhttpd_replay_chunked(node, buf->pstrs[0], chk);

        while (1)
        {
            memset(tmp, 0, PATH_MAX);
            chk = abcdk_dirent_read(dir, tmp);
            if (chk != 0)
                break;

            memset(tmp2, 0, PATH_MAX);
            memset(tmp3, 0, NAME_MAX);

            abcdk_basename(tmp3, tmp);

            path_len = PATH_MAX;
            abcdk_uri_encode(tmp3, strlen(tmp3), tmp2, &path_len, 0);

            chk = stat(tmp, &attr);
            if (chk != 0)
                continue;

            abcdk_time_sec2tm(&tm, attr.st_mtim.tv_sec, 1);

            chk = snprintf(buf->pstrs[0], buf->sizes[0], "<a href=\"%s%s\">%s</a>\t\t\t\t\t\t\t\t%s \t%lu\r\n",
                           tmp2 , (S_ISDIR(attr.st_mode) ? "/" : ""),tmp3,
                           abcdk_time_format(http_p->timefmt, &tm), attr.st_size);

            _abcdkhttpd_replay_chunked(node, buf->pstrs[0], chk);
        }

        chk = snprintf(buf->pstrs[0], buf->sizes[0],
                       "</pre>\r\n"
                       "<hr>\r\n"
                       "</body>\r\n"
                       "</html>\r\n");

        _abcdkhttpd_replay_chunked(node, buf->pstrs[0], chk);

        _abcdkhttpd_replay_chunked(node, NULL,0);

        _abcdkhttpd_logprint(node, 200, -1);
    }

final:

    abcdk_tree_free(&dir);
    abcdk_object_unref(&buf);
}

void _abcdkhttpd_replay_file(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    abcdk_object_t *file = NULL;
    const char *content_type;
    const char *p, *p_next;
    char tmp[100] = {0};
    size_t range_s = 0, range_e = -1, file_size = 0;
    struct tm tm;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    file = abcdk_mmap2(http_p->pathfile, 0, 0, 0);
    if (file)
    {

#ifdef HAVE_LIBMAGIC
        content_type = magic_buffer(http_p->magic_handle, file->pptrs[0], file->sizes[0]);
#else  // HAVE_LIBMAGIC
        content_type = abcdk_http_content_type_desc(http_p->pathfile);
#endif // HAVE_LIBMAGIC

        if (http_p->range)
        {
            p_next = http_p->range;
            p = abcdk_strtok(&p_next, "=");
            if (abcdk_strncmp("bytes", p, p_next - p, 0) != 0)
            {
                _abcdkhttpd_replay_nobody(node, 400);
                return;
            }

            p = abcdk_strtok(&p_next, ",");
            strncpy(tmp, p, p_next - p);
            abcdk_strtrim(tmp, isspace, 2);
            sscanf(p, "%*[^0-9]%lu-%lu", &range_s, &range_e);

            if (range_s >= range_e || range_s >= file->sizes[0])
            {
                _abcdkhttpd_replay_nobody(node, 400);
                return;
            }

            /*也许未指定末尾。*/
            range_e = ABCDK_MIN(file->sizes[0] - 1, range_e);

            /*保存文件大小。*/
            file_size = file->sizes[0];

            /*修改地址和长度为请求的数据范围。*/
            file->pptrs[0] += range_s;
            file->sizes[0] = range_e - range_s + 1;

            abcdk_comm_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: abcdk\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Content-Type: %s\r\n"
                                   "Accept-Ranges: bytes\r\n"
                                   "Content-Range: bytes %lu-%lu/%lu\r\n"
                                   "Content-Length: %lu\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(206),
                                   abcdk_time_format(http_p->timefmt, &tm),
                                   content_type,
                                   range_s, range_e, file_size,
                                   file->sizes[0]);
        }
        else
        {
            abcdk_comm_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: abcdk\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Content-Type: %s\r\n"
                                   "Content-Length: %lu\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(200),
                                   abcdk_time_format(http_p->timefmt, &tm),
                                   content_type,
                                   file->sizes[0]);
        }

        if (abcdk_strcmp(http_p->method, "head", 0) != 0)
            abcdk_comm_post(node, file);

        _abcdkhttpd_logprint(node, 200, file->sizes[0]);
    }
    else
    {
        _abcdkhttpd_replay_nobody(node, 403);
    }
}

void _abcdkhttpd_accept_cb(abcdk_comm_node_t *node, int *result)
{
    abcdkhttpd_node_t *http;

    /*设置默认返回值。*/
    *result = 0;

    /*创建用户环境。*/
    http = abcdk_heap_alloc(sizeof(abcdkhttpd_node_t));
    if (!http)
    {
        *result = -1;
        return;
    }

    /*获取服务器环境。*/
    http->ctx = abcdk_comm_get_userdata(node);

    /*重新绑定链路环境。*/
    abcdk_comm_set_userdata(node, http);

    /*设置时间格式串。*/
    http->timefmt = "%a, %d %b %Y %H:%M:%S GMT";

#ifdef HAVE_LIBMAGIC
    http->magic_handle = magic_open(MAGIC_MIME);
    if (!http->magic_handle)
    {
        *result = -1;
        return;
    }

    magic_load(http->magic_handle, NULL);
#endif // HAVE_LIBMAGIC
}

void _abcdkhttpd_request_cb(abcdk_comm_node_t *node, abcdk_http_request_t *req)
{
    abcdkhttpd_node_t *http_p;
    const char *p = NULL, *p_next = NULL;
    size_t path_len = PATH_MAX;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

    http_p->line0 = abcdk_http_request_env(req, 0);
    http_p->referer = abcdk_http_request_getenv(req, "referer");
    http_p->user_agent = abcdk_http_request_getenv(req, "user-agent");
    http_p->range = abcdk_http_request_getenv(req, "range");

    p_next = http_p->line0;

    memset(http_p->method, 0, 100);
    memset(http_p->location, 0, PATH_MAX);
    memset(http_p->path, 0, PATH_MAX);
    memset(http_p->params, 0, PATH_MAX);
    memset(http_p->version, 0, 100);
    memset(http_p->pathfile, 0, PATH_MAX);

    p = abcdk_strtok(&p_next, " ");
    strncpy(http_p->method, p, p_next - p);

    p = abcdk_strtok(&p_next, " ");
    strncpy(http_p->location, p, p_next - p);

    p = abcdk_strtok(&p_next, " ");
    strncpy(http_p->version, p, p_next - p);

    p_next = http_p->location;
    p = abcdk_strtok(&p_next, "?");
    abcdk_uri_decode(p, p_next - p, http_p->path, &path_len);

  //  fprintf(stderr,"%s\n",http_p->path);

    p = abcdk_strtok(&p_next, " ");
    if (p)
        strncpy(http_p->params, p, p_next - p);

    if (abcdk_strcmp(http_p->method, "optipn", 0) == 0)
    {
        _abcdkhttpd_replay_option(node, 200);
        return;
    }

    if (abcdk_strcmp(http_p->method, "POST", 0) != 0 &&
        abcdk_strcmp(http_p->method, "GET", 0) != 0 &&
        abcdk_strcmp(http_p->method, "HEAD", 0) != 0)
    {
        _abcdkhttpd_replay_option(node, 405);
        return;
    }

    abcdk_dirdir(http_p->pathfile, http_p->ctx->root_path);
    abcdk_dirdir(http_p->pathfile, http_p->path);

    chk = stat(http_p->pathfile, &http_p->attr);
    if (chk != 0)
    {
        if (errno == ENOENT)
            _abcdkhttpd_replay_nobody(node, 404);
        else
            _abcdkhttpd_replay_nobody(node, 403);
    }
    else if (S_ISDIR(http_p->attr.st_mode))
    {
        _abcdkhttpd_replay_dirent(node);
    }
    else if (S_ISREG(http_p->attr.st_mode))
    {
        _abcdkhttpd_replay_file(node);
    }
    else
    {
        _abcdkhttpd_replay_nobody(node, 403);
    }
}

void _abcdkhttpd_close_cb(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_userdata(node);

#ifdef HAVE_LIBMAGIC
    if (http_p->magic_handle)
        magic_close(http_p->magic_handle);
    http_p->magic_handle = NULL;
#endif // HAVE_LIBMAGIC

    abcdk_heap_free2((void **)&http_p);
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

    abcdk_http_callback_t cb = {_abcdkhttpd_accept_cb, _abcdkhttpd_request_cb, NULL, _abcdkhttpd_close_cb};
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

    /*英文；UTF-8。*/
    setlocale(LC_ALL, "en_US.UTF-8");

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
