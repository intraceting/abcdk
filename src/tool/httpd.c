/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

#ifdef HAVE_LIBMAGIC
#include <magic.h>
#endif // HAVE_LIBMAGIC


typedef struct _abcdk_httpd
{
    int errcode;
    abcdk_option_t *args;

    /*名称.*/
    const char *name_p;

    /*授权存储路径.*/
    const char *auth_path_p;

    /*WEB根目录.*/
    const char *root_path_p;

    /*CA证书.*/
    const char *ca_file_p;

    /*CA证书目录.*/
    const char *ca_path_p;

    /*服务器证书.*/
    const char *cert_file_p;

    /*服务器证书私钥.*/
    const char *key_file_p;

    /*上行数量包最大长度.*/
    size_t up_max_size;

    /*上行数量包临时缓存目录.*/
    const char *up_tmp_path_p;

    /*是否自动索引目录和文件.*/
    int auto_index;

    /*跨域服务器地址.*/
    const char *acao_p;

    /*是否启用HTTP/2支持.*/
    int enable_h2;

    abcdk_logger_t *logger;

#ifdef _MAGIC_H
    struct magic_set *magic_ctx;
#endif // _MAGIC_H

    /*时间环境*/
    locale_t loc_ctx;

    /**证书.*/
    X509 *pki_cert_ctx;

    /**私钥.*/
    EVP_PKEY *pki_key_ctx;

    abcdk_https_t *io_ctx;
    abcdk_https_session_t *listen_session;
    abcdk_https_session_t *listen_ssl_session;

} abcdk_httpd_t;

typedef struct _abcdk_httpd_stream
{
    abcdk_httpd_t *ctx_p;
    const char *method_p;
    const char *scheme_p;
    const char *host_p;
    const char *script_p;
    const char *range_p;
    abcdk_object_t *script_de;
    abcdk_object_t *pathfile;
    struct stat attr;

    abcdk_object_t *file_ctx;
    abcdk_object_t *range_de;
    abcdk_tree_t *dir_ctx;

} abcdk_httpd_stream_t;

void _abcdk_httpd_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的HTTP服务器.\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息.\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径.默认: /tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒), 1～60之间有效.默认: 30\n");
    fprintf(stderr, "\t\t注: 此功能不支持supervisor或类似的工具.\n");

    fprintf(stderr, "\n\t--name < NAME >\n");
    fprintf(stderr, "\t\t名称.默认: %s\n", "ABCDK");

    fprintf(stderr, "\n\t--auth-path < PATH >\n");
    fprintf(stderr, "\t\t授权存储路径.注: 文件名为账号名, 文件内容为密码.\n");

    fprintf(stderr, "\n\t--access-control-allow-origin < DOMAIN >\n");
    fprintf(stderr, "\t\t访问控制允许源.默认: *\n");

    fprintf(stderr, "\n\t--listen < ADDR >\n");
    fprintf(stderr, "\t\t监听地址.\n");

    fprintf(stderr, "\n\t\tIPv4://IP:PORT\n");
    fprintf(stderr, "\t\tIPv6://[IP]:PORT\n");
    fprintf(stderr, "\t\tIPv6://IP,PORT\n");
#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-ssl < ADDR >\n");
    fprintf(stderr, "\t\tSSL监听地址.\n");
 
    fprintf(stderr, "\n\t--ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件.\n");

    fprintf(stderr, "\n\t\t注: 仅支持PEM格式, 并且要求客户提供证书.\n");

    fprintf(stderr, "\n\t--ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径.\n");

    fprintf(stderr, "\n\t\t注: 仅支持PEM格式, 并且要求客户提供证书, 同时验证吊销列表.\n");

    fprintf(stderr, "\n\t--cert-file < FILE >\n");
    fprintf(stderr, "\t\t证书文件.\n");

    fprintf(stderr, "\n\t\t注: 仅支持PEM格式.\n");

    fprintf(stderr, "\n\t--key-file < FILE >\n");
    fprintf(stderr, "\t\t私钥文件.\n");

    fprintf(stderr, "\n\t\t注: 仅支持PEM格式.\n");

    fprintf(stderr, "\n\t--check-cert < 0|1 >\n");
    fprintf(stderr, "\t\t是否验证对端证书.默认: 0.\n");

    fprintf(stderr, "\n\t\t0: 否\n");
    fprintf(stderr, "\t\t1: 是\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--root-path < PATH >\n");
    fprintf(stderr, "\t\t服务器根据路径.默认: /var/abcdk/\n");

    fprintf(stderr, "\n\t--up-max-size < SIZE >\n");
    fprintf(stderr, "\t\t上行数据最大长度(字节).默认: %d\n", 4 * 1024 * 1024);

    fprintf(stderr, "\n\t--up-tmp-path < PATH >\n");
    fprintf(stderr, "\t\t上行数据临时缓存目录.\n");

    fprintf(stderr, "\n\t--auto-index\n");
    fprintf(stderr, "\t\t启用自动索引.\n");

    fprintf(stderr, "\n\t--enable-h2\n");
    fprintf(stderr, "\t\t启用HTTP2协议.\n");
}

static void _abcdk_httpd_reply_nobody(abcdk_https_stream_t *stream, int status, const char *acam)
{
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);

    abcdk_https_response_header_set(stream,"Status","%d",status);
    abcdk_https_response_header_set(stream,"Access-Control-Allow-Methods","%s",(acam?acam:"*"));
    abcdk_https_response(stream,NULL);
}

static void _abcdk_httpd_reply_dirent(abcdk_https_stream_t *stream)
{
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX], tmp2[PATH_MAX], tmp3[NAME_MAX];
    size_t path_len = PATH_MAX;
    char strsize[20] = {0};
    struct stat attr;
    struct tm tm;
    int chk;

    if (!stream_ctx_p->ctx_p->auto_index)
    {
        _abcdk_httpd_reply_nobody(stream, 403, "");
        return;
    }
    else if (abcdk_strcmp(stream_ctx_p->method_p, "OPTIONS", 0) == 0)
    {
        _abcdk_httpd_reply_nobody(stream, 200, "POST,GET");
        return;
    }
    else if (abcdk_strcmp(stream_ctx_p->method_p, "POST", 0) != 0 &&
             abcdk_strcmp(stream_ctx_p->method_p, "GET", 0) != 0)
    {
        _abcdk_httpd_reply_nobody(stream, 405, "");
        return;
    }

    abcdk_time_sec2tm(&tm, stream_ctx_p->attr.st_mtim.tv_sec, 1);

    chk = abcdk_dirent_open(&stream_ctx_p->dir_ctx, stream_ctx_p->pathfile->pptrs[0]);
    if (chk != 0)
    {
        _abcdk_httpd_reply_nobody(stream, 403, "");
        return;
    }

    abcdk_https_response_header_set(stream, "Content-Type", "text/html; charset=utf-8");

    abcdk_https_response_format(stream, 10000,
                              "<!DOCTYPE html>\r\n"
                              "<html>\r\n"
                              "<head><title>Index of %s</title></head>\r\n"
                              "<body bgcolor=\"white\">\r\n"
                              "<h1>Index of %s</h1>\r\n"
                              "<hr>\r\n"
                              "<pre>\r\n"
                              "<table width=\"100%%\">"
                              "<tr>"
                              "<th width=\"70%%\", align=\"left\">Name</th>"
                              "<th width=\"20%%\", align=\"left\">Time</th>"
                              "<th width=\"10%%\", align=\"right\">Size</th>"
                              "</tr>\r\n"
                              "<tr>\r\n"
                              "<td><a href=\"../\">../</a></td>\r\n"
                              "<td>-</td>\r\n"
                              "<td align=\"right\">-</td>\r\n"
                              "</tr>\r\n",
                              stream_ctx_p->script_de->pptrs[0], stream_ctx_p->script_de->pptrs[0]);
}

static void _abcdk_httpd_reply_dirent_more(abcdk_https_stream_t *stream)
{
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);
    char tmp[PATH_MAX], tmp2[PATH_MAX], tmp3[NAME_MAX];
    size_t path_len = PATH_MAX;
    char strsize[20] = {0};
    struct stat attr;
    struct tm tm;
    int chk;

    if (!stream_ctx_p->dir_ctx)
        return;

    while (1)
    {
        memset(tmp, 0, PATH_MAX);
        chk = abcdk_dirent_read(stream_ctx_p->dir_ctx, NULL, tmp, 1);
        if (chk != 0)
            break;

        memset(tmp2, 0, PATH_MAX);
        memset(tmp3, 0, NAME_MAX);

        abcdk_basename(tmp3, tmp);

        /*以“.”开头的文件表示具有隐藏属性.*/
        if (tmp3[0] == '.')
            continue;

        path_len = PATH_MAX;
        abcdk_url_encode(tmp3, strlen(tmp3), tmp2, &path_len, 1);

        chk = stat(tmp, &attr);
        if (chk != 0)
            continue;

        if (!S_ISDIR(attr.st_mode) && !S_ISREG(attr.st_mode))
            continue;

        abcdk_time_sec2tm(&tm, attr.st_mtim.tv_sec, 1);

        if (S_ISDIR(attr.st_mode))
            snprintf(strsize, 20, "%s", "-");
        else
            snprintf(strsize, 20, "%llu", attr.st_size);

        abcdk_https_response_format(stream, 10000,
                                  "<tr>\r\n"
                                  "<td><a href=\"%s%s\">%s</a></td>"
                                  "<td>%s</td>"
                                  "<td align=\"right\">%s</td>\r\n"
                                  "</tr>\r\n",
                                  tmp2, (S_ISDIR(attr.st_mode) ? "/" : ""), tmp3,
                                  abcdk_time_format(NULL, &tm, stream_ctx_p->ctx_p->loc_ctx), strsize);


        return;
    }

    abcdk_https_response_format(stream, 1000,
                              "</table>"
                              "</pre>\r\n"
                              "<hr>\r\n"
                              "</body>\r\n"
                              "</html>\r\n");

    abcdk_https_response(stream, NULL);
    abcdk_tree_free(&stream_ctx_p->dir_ctx);
}

static void _abcdk_httpd_reply_file(abcdk_https_stream_t *stream)
{
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);
    const char *type = NULL;
    size_t range_s = 0, range_e = -1, file_size = 0;
    struct tm tm;
    int chk;

    if (abcdk_strcmp(stream_ctx_p->method_p, "OPTIONS", 0) == 0)
    {
        _abcdk_httpd_reply_nobody(stream, 200, "POST,GET,HEAD");
        return;
    }
    else if (abcdk_strcmp(stream_ctx_p->method_p, "POST", 0) != 0 &&
             abcdk_strcmp(stream_ctx_p->method_p, "GET", 0) != 0 &&
             abcdk_strcmp(stream_ctx_p->method_p, "HEAD", 0) != 0)
    {
        _abcdk_httpd_reply_nobody(stream, 405, "");
        return;
    }

    abcdk_time_sec2tm(&tm, stream_ctx_p->attr.st_mtim.tv_sec, 1);

    stream_ctx_p->file_ctx = abcdk_mmap_filename(stream_ctx_p->pathfile->pstrs[0], 0, 0, 0, 0);
    if (!stream_ctx_p->file_ctx)
    {
        _abcdk_httpd_reply_nobody(stream, 403, "");
        return;
    }

    /*保存文件大小.*/
    file_size = stream_ctx_p->file_ctx->sizes[0];

#ifdef _MAGIC_H
    if (stream_ctx_p->ctx_p->magic_ctx)
        type = magic_buffer(stream_ctx_p->ctx_p->magic_ctx, stream_ctx_p->file_ctx->pptrs[0], stream_ctx_p->file_ctx->sizes[0]);
#endif // _MAGIC_H

    /*如果无法通过内容判断类型, 尝试通过文件名获取.*/
    if (!type)
        type = abcdk_http_content_type_desc(stream_ctx_p->pathfile->pstrs[0]);

    if (stream_ctx_p->range_p)
    {
        stream_ctx_p->range_de = abcdk_strtok2pair(stream_ctx_p->range_p, "=");

        abcdk_strtrim(stream_ctx_p->range_de->pstrs[0], isspace, 2);
        abcdk_strtrim(stream_ctx_p->range_de->pstrs[1], isspace, 2);

        if (abcdk_strcmp("bytes", stream_ctx_p->range_de->pstrs[0], 0) != 0)
        {
            _abcdk_httpd_reply_nobody(stream, 400, "");
            return;
        }

        sscanf(stream_ctx_p->range_de->pstrs[0], "%zu-%zu", &range_s, &range_e);
        if (range_s >= range_e || range_s >= stream_ctx_p->file_ctx->sizes[0])
        {
            _abcdk_httpd_reply_nobody(stream, 400, "");
            return;
        }

        /*也许未指定末尾.*/
        range_e = ABCDK_MIN(stream_ctx_p->file_ctx->sizes[0] - 1, range_e);

        /*修改地址和长度为请求的数据范围.*/
        stream_ctx_p->file_ctx->pptrs[0] += range_s;
        stream_ctx_p->file_ctx->sizes[0] = range_e - range_s + 1;

        abcdk_https_response_header_set(stream, "Status","%d",206);
        abcdk_https_response_header_set(stream, "Content-Type","%s",type);
        abcdk_https_response_header_set(stream, "Content-Range","bytes %zu-%zu/%zu",range_s, range_e, file_size);
        abcdk_https_response_header_set(stream, "Content-Length","%zu",stream_ctx_p->file_ctx->sizes[0]);
    }
    else
    {
        abcdk_https_response_header_set(stream, "Status","%d",200);
        abcdk_https_response_header_set(stream, "Content-Type","%s",type);
        abcdk_https_response_header_set(stream, "Content-Length","%zu",stream_ctx_p->file_ctx->sizes[0]);
    }

    chk = -1;
    if (abcdk_strcmp(stream_ctx_p->method_p, "HEAD", 0) != 0)
        chk = abcdk_https_response(stream, stream_ctx_p->file_ctx);
    
    abcdk_https_response(stream, NULL);

    /*不需要发送或发送失败时, 需要主动删除.*/
    if (chk != 0)
        abcdk_object_unref(&stream_ctx_p->file_ctx);
    else 
        stream_ctx_p->file_ctx = NULL;
}

static void _abcdk_httpd_session_prepare_cb(void *opaque, abcdk_https_session_t **session, abcdk_https_session_t *listen)
{
    *session = abcdk_https_session_alloc(((abcdk_httpd_t *)opaque)->io_ctx);
}

static void _abcdk_httpd_session_accept_cb(void *opaque, abcdk_https_session_t *session, int *result)
{
    *result = 0;
}

static void _abcdk_httpd_session_ready_cb(void *opaque, abcdk_https_session_t *session)
{
    abcdk_https_session_set_timeout(session, 30);
}

static void _abcdk_httpd_session_close_cb(void *opaque, abcdk_https_session_t *session)
{
}

static void _abcdk_httpd_stream_destructor_cb(void *opaque, abcdk_https_stream_t *stream)
{
    abcdk_httpd_t *ctx_p = (abcdk_httpd_t *)opaque;
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);

    abcdk_tree_free(&stream_ctx_p->dir_ctx);
    abcdk_object_unref(&stream_ctx_p->file_ctx);
    abcdk_object_unref(&stream_ctx_p->pathfile);
    abcdk_object_unref(&stream_ctx_p->range_de);
    abcdk_object_unref(&stream_ctx_p->script_de);

    abcdk_heap_free(stream_ctx_p);
    abcdk_https_set_userdata(stream, NULL);
}

static void _abcdk_httpd_stream_construct_cb(void *opaque, abcdk_https_stream_t *stream)
{
    abcdk_httpd_t *ctx_p = (abcdk_httpd_t *)opaque;
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_heap_alloc(sizeof(abcdk_httpd_stream_t));

    stream_ctx_p->ctx_p = ctx_p;
    abcdk_https_set_userdata(stream, stream_ctx_p);
}

static void _abcdk_httpd_stream_request_cb(void *opaque, abcdk_https_stream_t *stream)
{
    abcdk_httpd_t *ctx_p = (abcdk_httpd_t *)opaque;
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);
    int chk;

    /*删除过时的.*/
    abcdk_tree_free(&stream_ctx_p->dir_ctx);
    abcdk_object_unref(&stream_ctx_p->file_ctx);
    abcdk_object_unref(&stream_ctx_p->pathfile);
    abcdk_object_unref(&stream_ctx_p->range_de);
    abcdk_object_unref(&stream_ctx_p->script_de);

    chk = abcdk_https_check_auth(stream);
    if (chk != 0)
        return;

    stream_ctx_p->method_p = abcdk_https_request_header_get(stream, "Method");
    stream_ctx_p->scheme_p = abcdk_https_request_header_get(stream, "Scheme");
    stream_ctx_p->host_p = abcdk_https_request_header_get(stream, "Host");
    stream_ctx_p->script_p = abcdk_https_request_header_get(stream, "Script");
    stream_ctx_p->range_p = abcdk_https_request_header_get(stream, "Range");

    /*解码路径.*/
    stream_ctx_p->script_de = abcdk_url_decode2(stream_ctx_p->script_p, strlen(stream_ctx_p->script_p), 1);

    /*转换成绝对路径, 以防路径中存在“..”绕过根目录.*/
    abcdk_url_abspath(stream_ctx_p->script_de->pstrs[0], 0);
    stream_ctx_p->script_de->sizes[0] = strlen(stream_ctx_p->script_de->pstrs[0]);

    stream_ctx_p->pathfile = abcdk_object_printf(PATH_MAX, "%s/%s", ctx_p->root_path_p, stream_ctx_p->script_de->pstrs[0]);

    chk = stat(stream_ctx_p->pathfile->pstrs[0], &stream_ctx_p->attr);
    if (chk != 0)
    {
        if (errno == ENOENT)
            _abcdk_httpd_reply_nobody(stream, 404, "");
        else
            _abcdk_httpd_reply_nobody(stream, 403, "");
    }
    else if (S_ISDIR(stream_ctx_p->attr.st_mode))
    {
        _abcdk_httpd_reply_dirent(stream);
    }
    else if (S_ISREG(stream_ctx_p->attr.st_mode))
    {
        _abcdk_httpd_reply_file(stream);
    }
    else
    {
        _abcdk_httpd_reply_nobody(stream, 403, "");
    }

}

static void _abcdk_httpd_stream_output_cb(void *opaque, abcdk_https_stream_t *stream)
{
    abcdk_httpd_t *ctx_p = (abcdk_httpd_t *)opaque;
    abcdk_httpd_stream_t *stream_ctx_p = (abcdk_httpd_stream_t *)abcdk_https_get_userdata(stream);
    int chk;

    if (S_ISDIR(stream_ctx_p->attr.st_mode))
    {
        _abcdk_httpd_reply_dirent_more(stream);
    }
}

static int _abcdk_httpd_start_listen(abcdk_httpd_t *ctx,int ssl)
{
    const char *listen;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_https_config_t cfg = {0};
    abcdk_https_session_t *listen_p;
    int chk;
    
    if (ssl)
        listen = abcdk_option_get(ctx->args, "--listen-ssl", 0, NULL);
    else
        listen = abcdk_option_get(ctx->args, "--listen", 0, NULL);

    /*未启用.*/
    if(!listen)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen, 0);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "监听地址'%s'无法识别.", listen);
        return -1;
    }

    cfg.opaque = ctx;
    cfg.session_prepare_cb = _abcdk_httpd_session_prepare_cb;
    cfg.session_accept_cb = _abcdk_httpd_session_accept_cb;
    cfg.session_ready_cb = _abcdk_httpd_session_ready_cb;
    cfg.session_close_cb = _abcdk_httpd_session_close_cb;
    cfg.stream_destructor_cb = _abcdk_httpd_stream_destructor_cb;
    cfg.stream_construct_cb = _abcdk_httpd_stream_construct_cb;
    cfg.stream_request_cb = _abcdk_httpd_stream_request_cb;
    cfg.stream_output_cb = _abcdk_httpd_stream_output_cb;
    cfg.req_max_size = ctx->up_max_size;
    cfg.req_tmp_path = ctx->up_tmp_path_p;
    cfg.name = ctx->name_p;
    cfg.realm = "httpd";
    cfg.enable_h2 = ctx->enable_h2;
    cfg.auth_path = ctx->auth_path_p;
    cfg.a_c_a_o = ctx->acao_p;

    if (ssl)
    {
        cfg.ssl_scheme = ABCDK_STCP_SSL_SCHEME_PKI;
        cfg.pki_ca_file = ctx->ca_file_p;
        cfg.pki_ca_path = ctx->ca_path_p;
        cfg.pki_chk_crl = ((ctx->ca_file_p||ctx->ca_path_p)?2:0);
        cfg.pki_use_cert = ctx->pki_cert_ctx;
        cfg.pki_use_key = ctx->pki_key_ctx;
    }

    if (ssl)
        listen_p = ctx->listen_ssl_session = abcdk_https_session_alloc(ctx->io_ctx);
    else 
        listen_p = ctx->listen_session = abcdk_https_session_alloc(ctx->io_ctx);

    if (!listen_p)
    {
        abcdk_trace_printf(LOG_ERR, "内部错误.");
        return -2;
    }

    chk = abcdk_https_session_listen(listen_p,&listen_addr,&cfg);
    if(chk == 0)
        return 0;

    return -3;

}

static void _abcdk_httpd_process(abcdk_httpd_t *ctx)
{
    int max_client = 1000;
    const char *log_path = NULL;
    int chk;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志.*/
    ctx->logger = abcdk_logger_open2(log_path, "httpd.log", "httpd.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志.*/
    abcdk_trace_printf_redirect(abcdk_logger_proxy, ctx->logger);

    abcdk_trace_printf(LOG_INFO, "启动……");

    abcdk_openssl_init();

    ctx->name_p = abcdk_option_get(ctx->args, "--name", 0, "ABCDK");
    ctx->acao_p = abcdk_option_get(ctx->args, "--access-control-allow-origin", 0, "*");
#ifdef OPENSSL_VERSION_NUMBER
    ctx->ca_file_p = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path_p = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file_p = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file_p = abcdk_option_get(ctx->args, "--key-file", 0, NULL);
#endif // OPENSSL_VERSION_NUMBER
    ctx->root_path_p = abcdk_option_get(ctx->args, "--root-path", 0, "/var/abcdk/");
    ctx->up_max_size = abcdk_option_get_llong(ctx->args, "--up-max-size", 0, 4 * 1024 * 1024);
    ctx->up_tmp_path_p = abcdk_option_get(ctx->args, "--up-tmp-path", 0, NULL);
    ctx->auto_index = abcdk_option_exist(ctx->args, "--auto-index");
    ctx->auth_path_p = abcdk_option_get(ctx->args, "--auth-path", 0, NULL);
    ctx->enable_h2 = abcdk_option_exist(ctx->args, "--enable-h2");

#ifdef _MAGIC_H
    ctx->magic_ctx = magic_open(MAGIC_MIME | MAGIC_SYMLINK);
    if (ctx->magic_ctx)
        magic_load(ctx->magic_ctx, NULL);
#endif // _MAGIC_H

    ctx->loc_ctx = newlocale(LC_ALL_MASK, "en_US.UTF-8", NULL);

#ifdef OPENSSL_VERSION_NUMBER
    if (ctx->cert_file_p)
    {
        ctx->pki_cert_ctx = abcdk_openssl_cert_load(ctx->cert_file_p);
        if (!ctx->pki_cert_ctx)
        {
            abcdk_trace_printf(LOG_ERR, "加载证书(%s)失败.", ctx->cert_file_p);
            goto ERR;
        }
    }

    if (ctx->key_file_p)
    {
        ctx->pki_key_ctx = abcdk_openssl_evp_pkey_load(ctx->cert_file_p, 0, NULL);
        if (!ctx->pki_key_ctx)
        {
            abcdk_trace_printf(LOG_ERR, "加载密钥(%s)失败.", ctx->key_file_p);
            goto ERR;
        }
    }

#endif //OPENSSL_VERSION_NUMBER

    /*创建可能不存在的路径.*/
    if(ctx->up_tmp_path_p)
        abcdk_mkdir(ctx->up_tmp_path_p, 0600);

    ctx->io_ctx = abcdk_https_create();
    if (!ctx->io_ctx)
    {
        abcdk_trace_printf(LOG_WARNING, "内存错误.\n");
        goto ERR;
    }

    chk = _abcdk_httpd_start_listen(ctx,0);
    if(chk != 0)
        goto ERR;

#ifdef OPENSSL_VERSION_NUMBER
    chk = _abcdk_httpd_start_listen(ctx,1);
    if(chk != 0)
        goto ERR;
#endif //OPENSSL_VERSION_NUMBER

    /*等待终止信号.*/
    abcdk_proc_wait_exit_signal(-1);

ERR:

    abcdk_https_destroy(&ctx->io_ctx);
    abcdk_https_session_unref(&ctx->listen_session);
    abcdk_https_session_unref(&ctx->listen_ssl_session);

#ifdef _MAGIC_H
    if (ctx->magic_ctx)
        magic_close(ctx->magic_ctx);
#endif // _MAGIC_H

    if(ctx->loc_ctx)
        freelocale(ctx->loc_ctx);

#ifdef OPENSSL_VERSION_NUMBER
    abcdk_openssl_x509_free(&ctx->pki_cert_ctx);
    abcdk_openssl_evp_pkey_free(&ctx->pki_key_ctx);
#endif //OPENSSL_VERSION_NUMBER

    abcdk_openssl_cleanup();

    abcdk_trace_printf(LOG_INFO, "停止.");

    /*关闭日志.*/
    abcdk_logger_close(&ctx->logger);
}

static int _abcdk_httpd_daemon_process_cb(void *opaque)
{
    abcdk_httpd_t *ctx = (abcdk_httpd_t *)opaque;

    _abcdk_httpd_process(ctx);

    return 0;
}

static void _abcdk_httpd_daemon(abcdk_httpd_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;
    int chk;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志.*/
    logger = abcdk_logger_open2(log_path, "httpd-daemon.log", "httpd-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志.*/
    abcdk_trace_printf_redirect(abcdk_logger_proxy, logger);

    while(1)
    {
        chk = abcdk_proc_subprocess(_abcdk_httpd_daemon_process_cb, ctx,NULL,NULL);
        if(chk == 0)
            break;

        sleep(interval);
    }

    /*关闭日志.*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_httpd(abcdk_option_t *args)
{
    abcdk_httpd_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_httpd_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式.\n");
            daemon(1, 0);

            _abcdk_httpd_daemon(&ctx);
        }
        else
        {
            _abcdk_httpd_process(&ctx);
        }
    }

    return 0;
}

