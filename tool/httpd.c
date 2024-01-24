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

#define ABCDKHTTPD_LISTEN 0
#define ABCDKHTTPD_LISTEN_SSL 1

typedef struct _abcdkhttpd
{
    int errcode;
    abcdk_option_t *args;

    /*服务器名称。*/
    const char *server_name;

    /*WEB根目录。*/
    const char *root_path;

    /*CA证书。*/
    const char *ca_file;

    /*CA证书目录。*/
    const char *ca_path;

    /*服务器证书。*/
    const char *cert_file;

    /*服务器证书私钥。*/
    const char *key_file;

    /*授权存储路径。*/
    const char *auth_path;

    /*上行数量包最大长度。*/
    size_t up_max_size;

    /*上行数量包临时缓存目录。*/
    const char *up_tmp_path;

    /*是否自动索引目录和文件。*/
    int auto_index;

    /*是否自动索引具有隐藏属性的目录和文件。*/
    int auto_index_hidden_file;

    /*跨域服务器地址。*/
    const char *a_c_a_o;

    abcdk_logger_t *logger;

#ifdef _MAGIC_H
    struct magic_set *magic_ctx;
#endif // _MAGIC_H

    abcdk_httpd_t *io_ctx;

} abcdkhttpd_t;

void _abcdkhttpd_print_usage(abcdk_option_t *args)
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

    fprintf(stderr, "\n\t--max-client < NUMBER >\n");
    fprintf(stderr, "\t\t最大连接数。默认：系统限定的1/2\n");

    fprintf(stderr, "\n\t--server-name < NAME >\n");
    fprintf(stderr, "\t\t服务器名称。默认：%s\n",SOLUTION_NAME);

    fprintf(stderr, "\n\t--auth-path < PATH >\n");
    fprintf(stderr, "\t\t授权存储路径。注：文件名为账号名，文件内容为密码。\n");

    fprintf(stderr, "\n\t--access-control-allow-origin < DOMAIN >\n");
    fprintf(stderr, "\t\t访问控制允许源。默认：*\n");

    fprintf(stderr, "\n\t--listen < ADDR >\n");
    fprintf(stderr, "\t\t监听地址。\n");

    fprintf(stderr, "\n\t\tIPv4://IP:PORT\n");
    fprintf(stderr, "\t\tIPv6://[IP]:PORT\n");
    fprintf(stderr, "\t\tIPv6://IP,PORT\n");
#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-ssl < ADDR >\n");
    fprintf(stderr, "\t\tSSL监听地址。\n");
    
    fprintf(stderr, "\n\t--ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件。注：仅支持PEM格式，并且要求客户提供证书。\n");

    fprintf(stderr, "\n\t--ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径。注：仅支持PEM格式，并且要求客户提供证书，同时验证吊销列表。\n");

    fprintf(stderr, "\n\t--cert-file < FILE >\n");
    fprintf(stderr, "\t\t服务器证书文件。注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--key-file < FILE >\n");
    fprintf(stderr, "\t\t服务器私钥文件。注：仅支持PEM格式。\n");
#endif //HEADER_SSL_H

    fprintf(stderr, "\n\t--root-path < PATH >\n");
    fprintf(stderr, "\t\t服务器根据路径。默认：/var/abcdk/\n");

    fprintf(stderr, "\n\t--up-max-size < SIZE >\n");
    fprintf(stderr, "\t\t上行数据最大长度(字节)。默认：%d\n",4*1024*1024);

    fprintf(stderr, "\n\t--up-tmp-path < PATH >\n");
    fprintf(stderr, "\t\t上行数据临时缓存目录。\n");
    
    fprintf(stderr, "\n\t--auto-index\n");
    fprintf(stderr, "\t\t启用自动索引。\n");

}

static void _abcdkhttpd_reply_nobody(abcdkhttpd_t *ctx, abcdk_object_t *stream, int status,const char *a_c_a_m)
{
    abcdk_httpd_response_nobody(stream,status,a_c_a_m,ctx->a_c_a_o);
}

void _abcdkhttpd_reply_chunked(abcdkhttpd_t *ctx, abcdk_object_t *stream, int max, const char *fmt, ...)
{
    abcdk_object_t *obj = NULL;
    int chk;

    if (max <= 0 || fmt == NULL)
    {
        obj = abcdk_http_chunked_copyfrom(NULL, 0);
    }
    else
    {
        va_list ap;
        va_start(ap, fmt);
        obj = abcdk_http_chunked_vformat(max, fmt, ap);
        va_end(ap);
    }

    chk = abcdk_httpd_response_body(stream, obj);
    if (chk == 0)
        return;

    abcdk_object_unref(&obj);
}

void _abcdkhttpd_reply_dirent(abcdk_asynctcp_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX], tmp2[PATH_MAX], tmp3[NAME_MAX];
    size_t path_len = PATH_MAX;
    char strsize[20] = {0};
    struct stat attr;
    struct tm tm;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_asynctcp_get_userdata(node);

    if(!http_p->ctx->auto_index)
    {
        _abcdkhttpd_reply_nobody(node, 403, "");
        return;
    }
    else if (abcdk_strcmp(http_p->method->pstrs[0], "OPTIONS", 0) == 0)
    {
        _abcdkhttpd_reply_nobody(node, 200, "POST,GET");
        return;
    }
    else if (abcdk_strcmp(http_p->method->pstrs[0], "POST", 0) != 0 &&
             abcdk_strcmp(http_p->method->pstrs[0], "GET", 0) != 0)
    {
        _abcdkhttpd_reply_nobody(node, 405, "");
        return;
    }
            
    chk = _abcdkhttpd_check_auth(node,0);
    if (chk != 0)
        return;

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    chk = abcdk_dirent_open(&dir, http_p->pathfile);
    if (chk != 0)
    {
        _abcdkhttpd_reply_nobody(node, 403,"");
    }
    else
    {
        abcdk_asynctcp_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: %s\r\n"
                               "Data: %s\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Access-Control-Allow-Origin: %s\r\n"
                               "Content-Type: %s; charset=utf-8\r\n"
                               "Transfer-Encoding: chunked\r\n"
                               "Cache-Control: no-cache\r\n"
                               "Expires: 0\r\n"
                               "\r\n",
                               abcdk_http_status_desc(200),
                               http_p->ctx->server_name,
                               abcdk_time_format(http_p->timefmt, &tm,http_p->ctx->loc),
                               http_p->ctx->a_c_a_o,
                               abcdk_http_content_type_desc(".html"));

        _abcdkhttpd_reply_chunked(node, 10000,
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
                                        http_p->path->pstrs[0], http_p->path->pstrs[0]);

        while (1)
        {
            memset(tmp, 0, PATH_MAX);
            chk = abcdk_dirent_read(dir,NULL, tmp,1);
            if (chk != 0)
                break;

            memset(tmp2, 0, PATH_MAX);
            memset(tmp3, 0, NAME_MAX);

            abcdk_basename(tmp3, tmp);

            if (!http_p->ctx->auto_index_hidden_file)
            {
                /*以“.”开头的文件表示具有隐藏属性。*/
                if (tmp3[0] == '.')
                    continue;
            }

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

            _abcdkhttpd_reply_chunked(node, 10000,
                                            "<tr>\r\n"
                                            "<td><a href=\"%s%s\">%s</a></td>"
                                            "<td>%s</td>"
                                            "<td align=\"right\">%s</td>\r\n"
                                            "</tr>\r\n",
                                            tmp2, (S_ISDIR(attr.st_mode) ? "/" : ""), tmp3,
                                            abcdk_time_format(http_p->timefmt_lc, &tm,http_p->ctx->loc), strsize);
        }

        _abcdkhttpd_reply_chunked(node, 1000,
                                        "</table>"
                                        "</pre>\r\n"
                                        "<hr>\r\n"
                                        "</body>\r\n"
                                        "</html>\r\n");

        _abcdkhttpd_reply_chunked(node, 0,NULL);

        _abcdkhttpd_logprint(node, 200, -1);
    }

final:

    abcdk_tree_free(&dir);
}

void _abcdkhttpd_reply_file(abcdkhttpd_t *ctx, abcdk_object_t *stream,)
{
    abcdkhttpd_node_t *http_p;
    abcdk_object_t *file = NULL;
    const char *content_type = NULL;
    const char *p, *p_next;
    char tmp[100] = {0};
    size_t range_s = 0, range_e = -1, file_size = 0;
    int status;
    struct tm tm;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_asynctcp_get_userdata(node);

    if (abcdk_strcmp(http_p->method->pstrs[0], "OPTIONS", 0) == 0)
    {
        _abcdkhttpd_reply_nobody(node, 200, "POST,GET,HEAD");
        return;
    }
    else if (abcdk_strcmp(http_p->method->pstrs[0], "POST", 0) != 0 &&
             abcdk_strcmp(http_p->method->pstrs[0], "GET", 0) != 0 &&
             abcdk_strcmp(http_p->method->pstrs[0], "HEAD", 0) != 0)
    {
        _abcdkhttpd_reply_nobody(node, 405, "");
        return;
    }
        
    chk = _abcdkhttpd_check_auth(node,0);
    if (chk != 0)
        return;

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    file = abcdk_mmap_filename(http_p->pathfile, 0, 0, 0,0);
    if (file)
    {
        /*保存文件大小。*/
        file_size = file->sizes[0];

#ifdef _MAGIC_H
        if (http_p->ctx->magic_handle)
            content_type = magic_buffer(http_p->ctx->magic_handle, file->pptrs[0], file->sizes[0]);
#endif // _MAGIC_H

        /*如果无法通过内容判断类型，尝试通过文件名获取。*/
        if (!content_type)
            content_type = abcdk_http_content_type_desc(http_p->pathfile);

        if (http_p->range)
        {
            p_next = http_p->range;
            p = abcdk_strtok(&p_next, "=");
            if (abcdk_strncmp("bytes", p, p_next - p, 0) != 0)
            {
                _abcdkhttpd_reply_nobody(node, 400, "");
                return;
            }

            p = abcdk_strtok(&p_next, "=");
            strncpy(tmp, p, p_next - p);
            abcdk_strtrim(tmp, isspace, 2);
            sscanf(p, "%zu-%zu", &range_s, &range_e);

            if (range_s >= range_e || range_s >= file->sizes[0])
            {
                _abcdkhttpd_reply_nobody(node, 400, "");
                return;
            }

            /*也许未指定末尾。*/
            range_e = ABCDK_MIN(file->sizes[0] - 1, range_e);

            /*修改地址和长度为请求的数据范围。*/
            file->pptrs[0] += range_s;
            file->sizes[0] = range_e - range_s + 1;

            abcdk_asynctcp_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: %s\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Access-Control-Allow-Origin: %s\r\n"
                                   "Content-Type: %s\r\n"
                                   "Accept-Ranges: bytes\r\n"
                                   "Content-Range: bytes %zu-%zu/%zu\r\n"
                                   "Content-Length: %llu\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Expires: 0\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(status = 206),
                                   http_p->ctx->server_name,
                                   abcdk_time_format(http_p->timefmt, &tm,http_p->ctx->loc),
                                   http_p->ctx->a_c_a_o,
                                   content_type,
                                   range_s, range_e, file_size,
                                   file->sizes[0]);
        }
        else
        {
            abcdk_asynctcp_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: %s\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Access-Control-Allow-Origin: %s\r\n"
                                   "Content-Type: %s\r\n"
                                   "Content-Length: %llu\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Expires: 0\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(status = 200),
                                   http_p->ctx->server_name,
                                   abcdk_time_format(http_p->timefmt, &tm,http_p->ctx->loc),
                                   http_p->ctx->a_c_a_o,
                                   content_type,
                                   file->sizes[0]);
        }

        chk = -1;
        if (abcdk_strcmp(http_p->method->pstrs[0], "HEAD", 0) != 0)
            chk = abcdk_asynctcp_post(node, file);

        /*不需要发送或发送失败时，需要主动删除。*/
        if (chk != 0)
            abcdk_object_unref(&file);

        _abcdkhttpd_logprint(node, status, file_size);
    }
    else
    {
        _abcdkhttpd_reply_nobody(node, 403,"");
    }
}

static void _abcdkhttpd_session_prepare_cb(void *opaque,abcdk_httpd_session_t **session,abcdk_httpd_session_t *listen)
{
    *session = abcdk_httpd_session_alloc((abcdkhttpd_t*)opaque);
}

static void _abcdkhttpd_session_accept_cb(void *opaque,abcdk_httpd_session_t *session,int *result)
{
    *result = 0;
}

static void _abcdkhttpd_session_ready_cb(void *opaque,abcdk_httpd_session_t *session)
{
    abcdk_httpd_session_set_timeout(session,10);
}

static void _abcdkhttpd_ssession_close_cb(void *opaque,abcdk_httpd_session_t *session)
{

}

static void _abcdkhttpd_request_cb(void *opaque, abcdk_object_t *stream)
{
    abcdkhttpd_t *ctx_p = (abcdkhttpd_t*)opaque;

    const char *method_p = abcdk_httpd_request_header_get(stream,"Method");
    const char *scheme_p = abcdk_httpd_request_header_get(stream,"Scheme");
    const char *host_p = abcdk_httpd_request_header_get(stream,"Host");
    const char *script_p = abcdk_httpd_request_header_get(stream,"Script");

    /*解码路径。*/
    abcdk_object_t *script_de = abcdk_url_decode2(script_p,strlen(script_p),1);

    /*转换成绝对路径，以防路径中存在“..”绕过根目录。*/
    abcdk_url_abspath(script_de->pstrs[0],0);
    script_de->sizes[0] = strlen(script_de->pstrs[0]);

    abcdk_object_t *pathfile = abcdk_object_printf(PATH_MAX,"%s/%s",ctx_p->root_path,script_de->pstrs[0]);

    struct stat attr;

    chk = stat(pathfile->pstrs[0], &attr);
    if (chk != 0)
    {
        if (errno == ENOENT)
            _abcdkhttpd_reply_nobody(ctx, stream, 404, "");
        else
            _abcdkhttpd_reply_nobody(ctx, stream, 403, "");
    }
    else if (S_ISDIR(http_p->attr.st_mode))
    {
        _abcdkhttpd_reply_dirent(ctx, stream,pathfile->pstrs[0]);
    }
    else if (S_ISREG(http_p->attr.st_mode))
    {
        _abcdkhttpd_reply_file(ctx, stream,pathfile->pstrs[0]);
    }
    else
    {
        _abcdkhttpd_reply_nobody(ctx, stream, 403, "");
    }

    abcdk_object_unref(&pathfile);
    abcdk_object_unref(&script_de);
}

static void _abcdkhttpd_process(abcdkhttpd_t *ctx)
{
    int max_client = 1000;
    const char *log_path = NULL;
    int chk;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path,"httpd.log", "httpd.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace,ctx->logger);

    abcdk_trace_output( LOG_INFO, "启动……");

    max_client = abcdk_option_get_int(ctx->args, "--max-client", 0, -1);
    ctx->server_name = abcdk_option_get(ctx->args, "--server-name", 0, SOLUTION_NAME);
    ctx->a_c_a_o = abcdk_option_get(ctx->args, "--access-control-allow-origin",0,"*");
    ctx->listen[ABCDKHTTPD_LISTEN] = abcdk_option_get(ctx->args, "--listen", 0, NULL);
#ifdef HEADER_SSL_H
    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);
#endif //HEADER_SSL_H
    ctx->root_path = abcdk_option_get(ctx->args, "--root-path", 0, "/var/abcdk/");
    ctx->up_max_size = abcdk_option_get_llong(ctx->args, "--up-max-size", 0, 4 * 1024 * 1024);
    ctx->up_tmp_path = abcdk_option_get(ctx->args, "--up-tmp-path", 0, NULL);
    ctx->auto_index = abcdk_option_exist(ctx->args, "--auto-index");
    ctx->auth_path = abcdk_option_get(ctx->args, "--auth-path", 0, NULL);

#ifdef _MAGIC_H
    ctx->magic_ctx = magic_open(MAGIC_MIME | MAGIC_SYMLINK);
    if (ctx->magic_ctx)
        magic_load(ctx->magic_ctx, NULL);
#endif // _MAGIC_H

    /**/
    abcdk_mkdir(ctx->up_tmp_path,0600);
    

#ifdef HEADER_SSL_H

#endif // HEADER_SSL_H

    ctx->io_ctx = abcdk_httpd_create(max_client, -1);
    if (!ctx->io_ctx)
    {
        abcdk_trace_output( LOG_WARNING, "内存错误。\n");
        goto final;
    }

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

final:

    abcdk_httpd_destroy(&ctx->io_ctx);

#ifdef _MAGIC_H
    if (ctx->magic_ctx)
        magic_close(ctx->magic_ctx);
#endif // _MAGIC_H


    abcdk_trace_output( LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}

static int _abcdkhttpd_daemon_process_cb(void *opaque)
{
    abcdkhttpd_t *ctx = (abcdkhttpd_t*)opaque;

    _abcdkhttpd_process(ctx);

    return 0;
}

static void _abcdkhttpd_daemon(abcdkhttpd_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval,1,60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path,"httpd-daemon.log", "httpd-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace,logger);
            
    abcdk_proc_daemon(interval, _abcdkhttpd_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_httpd(abcdk_option_t *args)
{
    abcdkhttpd_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkhttpd_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args,"--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);
            
            _abcdkhttpd_daemon(&ctx);
        }
        else
        {
            _abcdkhttpd_process(&ctx);
        }
    }

    return 0;
}
