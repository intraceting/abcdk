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
#define ABCDKHTTPD_PASSWD_MAX 100

typedef struct _abcdkhttpd
{
    int errcode;
    abcdk_option_t *args;

    uint64_t realm;

    locale_t loc;

#ifdef HAVE_LIBMAGIC
    struct magic_set *magic_handle;
#endif // HAVE_LIBMAGIC
    abcdk_mutex_t magic_mutex;

    abcdk_comm_t *comm;
    abcdk_comm_node_t *comm_listen[2];
    SSL_CTX *ssl_ctx;

    int max_client;
    const char *server_name;
    const char *root_path;
    const char *listen[2];
    const char *ca_file;
    const char *ca_path;
    const char *cert_file;
    const char *key_file;
    const char *passwd[ABCDKHTTPD_PASSWD_MAX];
    size_t up_max_size;
    const char *up_tmp_path;
    int exclude_hidden_file;
    const char *a_c_a_o;

} abcdkhttpd_t;

typedef struct _abcdkhttpd_node
{
    abcdkhttpd_t *ctx;

    abcdk_http_request_t *req;

    char remote[100];

    const char *timefmt;
    const char *timefmt_lc;

    abcdk_md5_t *md5;

    /**
     * 0: unknown
     * 1: http/1.0 or http/1.1 or http/0.9 or rtsp/1.0
     * 2: http/2
     */
    int protocol;

    const char *line0;
    const char *referer;
    const char *user_agent;
    const char *range;
    const char *auth;

    char method[100];
    char location[PATH_MAX];
    char path[PATH_MAX];
    char params[PATH_MAX];
    char version[100];
    char referer_de[PATH_MAX];
    char a_c_a_o[PATH_MAX];

    char pathfile[PATH_MAX];
    struct stat attr;

} abcdkhttpd_node_t;

void _abcdkhttpd_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的HTTP服务器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--max-client < NUMBER >\n");
    fprintf(stderr, "\t\t最大连接数。默认：系统限定的1/2\n");

    fprintf(stderr, "\n\t--server-name < NAME >\n");
    fprintf(stderr, "\t\t服务器名称。默认：%s\n",SOLUTION_NAME);

    fprintf(stderr, "\n\t--root-path < PATH >\n");
    fprintf(stderr, "\t\t服务器根据路径。默认：/var/abcdk/\n");

    fprintf(stderr, "\n\t--access-control-allow-origin < DOMAIN >\n");
    fprintf(stderr, "\t\t访问控制允许源。\n");

    fprintf(stderr, "\n\t--listen < ADDR [ ADDR ...] >\n");
    fprintf(stderr, "\t\t监听地址。\n");

    fprintf(stderr, "\n\t--listen-ssl < ADDR [ ADDR ...] >\n");
    fprintf(stderr, "\t\tSSL监听地址。\n");

    fprintf(stderr, "\n\t\tIPv4：IP:PORT\n");
    fprintf(stderr, "\t\tIPv4：DOMAIN:PORT\n");
    fprintf(stderr, "\t\tIPv6：IP,PORT\n");
    fprintf(stderr, "\t\tIPv6：[IP]:PORT\n");
    fprintf(stderr, "\t\tIPv6：DOMAIN,PORT\n");

    fprintf(stderr, "\n\t--ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件。注：仅支持PEM格式，并且要求客户提供证书。\n");

    fprintf(stderr, "\n\t--ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径。注：仅支持PEM格式，并且要求客户提供证书，同时验证吊销列表。\n");

    fprintf(stderr, "\n\t--cert-file < FILE >\n");
    fprintf(stderr, "\t\t服务器证书文件。注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--key-file < FILE >\n");
    fprintf(stderr, "\t\t服务器私钥文件。注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--exclude-hidden-file\n");
    fprintf(stderr, "\t\t排除隐藏属性的文件和目录。\n");
}

uint64_t _abcdkhttpd_clock()
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 0);
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

void _abcdkhttpd_node_destroy_cb(abcdk_object_t *obj, void *opaque)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)obj->pptrs[0];
    if (!http_p)
        return;

    abcdk_http_request_unref(&http_p->req);
    abcdk_md5_destroy(&http_p->md5);
}

void _abcdkhttpd_prepare_cb(abcdk_comm_node_t **node, abcdk_comm_node_t *listen);
void _abcdkhttpd_accept_cb(abcdk_comm_node_t *node, int *result);
void _abcdkhttpd_connect_cb(abcdk_comm_node_t *node);
void _abcdkhttpd_close_cb(abcdk_comm_node_t *node);

void _abcdkhttpd_event_cb(abcdk_comm_node_t *node, uint32_t event, int *result)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);
    if (!http_p)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }

    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdkhttpd_accept_cb(node, result);
        break;
    case ABCDK_COMM_EVENT_CONNECT:
        _abcdkhttpd_connect_cb(node);
        break;
    case ABCDK_COMM_EVENT_INPUT:
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    case ABCDK_COMM_EVENT_INTERRUPT:
    default:
        _abcdkhttpd_close_cb(node);
        break;
    }
}

void _abcdkhttpd_logprint(abcdk_comm_node_t *node, int status, size_t size)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    abcdk_log_printf(LOG_INFO, "\"%s\" \"%s\" %d %ld \"%s\" \"%s\" \n",
                     http_p->remote,
                     http_p->line0, status, (ssize_t)size,
                     http_p->referer ? http_p->referer : "-",
                     http_p->user_agent ? http_p->user_agent : "-");
}

int _abcdkhttpd_check_auth(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    char auth_method[100] = {0};
    char basic_req[100] = {0};
    abcdk_object_t *digest_req = NULL;
    uint8_t digest_md5_buf[16];
    char digest_ha1[33] = {0}, digest_ha2[33] = {0}, digest_rsp[33] = {0};
    const char *key, *value, *p, *p_next;
    size_t klen, vlen;
    uint64_t nonce;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    if (!http_p->ctx->passwd[0])
        return 0;

    if (!http_p->auth)
        goto final_error;

    p_next = http_p->auth;

    p = abcdk_strtok(&p_next, " ");
    strncpy(auth_method, p, p_next - p);

    if (abcdk_strcmp(auth_method, "Basic", 0) == 0)
    {
        if (http_p->md5)
            goto final_error;

        p = abcdk_strtok2(&p_next, "\r", 1);

        abcdk_basecode_t bc;
        abcdk_basecode_init(&bc, 64);
        abcdk_basecode_decode(&bc, p, p_next - p, (uint8_t *)basic_req, 100);

        for (int i = 0; http_p->ctx->passwd[i]; i++)
        {
            if (abcdk_strcmp(basic_req, http_p->ctx->passwd[i], 1) == 0)
                return 0;
        }
    }
    else if (abcdk_strcmp(auth_method, "Digest", 0) == 0)
    {
        if (!http_p->md5)
            goto final_error;

        size_t sizes[] = {100, 100, 4096, 100, 100};
        digest_req = abcdk_object_alloc(sizes, 5, 0);

        for (; p_next;)
        {
            p = abcdk_strtok2(&p_next, ",", 1);
            if (!p)
                break;

            p_next = p;

            key = abcdk_strtok2(&p_next, "=", 1);
            if (!key)
                break;
            klen = p_next - key;

            value = abcdk_strtok2(&p_next, ",", 1);
            if (!value)
                break;
            vlen = p_next - value;

            if (abcdk_strncmp(key, "username", klen, 0) == 0)
            {
                strncpy(digest_req->pstrs[0], value, ABCDK_MIN(vlen, digest_req->sizes[0]));
                abcdk_strtrim2(digest_req->pstrs[0], isspace, "=\"\'", 2);
            }
            else if (abcdk_strncmp(key, "nonce", klen, 0) == 0)
            {
                strncpy(digest_req->pstrs[1], value, ABCDK_MIN(vlen, digest_req->sizes[1]));
                abcdk_strtrim2(digest_req->pstrs[1], isspace, "=\"\'", 2);
            }
            else if (abcdk_strncmp(key, "uri", klen, 0) == 0)
            {
                strncpy(digest_req->pstrs[2], value, ABCDK_MIN(vlen, digest_req->sizes[2]));
                abcdk_strtrim2(digest_req->pstrs[2], isspace, "=\"\'", 2);
            }
            else if (abcdk_strncmp(key, "response", klen, 0) == 0)
            {
                strncpy(digest_req->pstrs[3], value, ABCDK_MIN(vlen, digest_req->sizes[3]));
                abcdk_strtrim2(digest_req->pstrs[3], isspace, "=\"\'", 2);
            }
            else if (abcdk_strncmp(key, "realm", klen, 0) == 0)
            {
                strncpy(digest_req->pstrs[4], value, ABCDK_MIN(vlen, digest_req->sizes[4]));
                abcdk_strtrim2(digest_req->pstrs[4], isspace, "=\"\'", 2);
            }
        }

        /*如果服务重启过，通知客户端重新输入验证信息。*/
        if (strtoull(digest_req->pstrs[4], NULL, 0) != http_p->ctx->realm)
            goto final_error;

        chk = -1;
        for (int i = 0; http_p->ctx->passwd[i]; i++)
        {
            p_next = http_p->ctx->passwd[i];
            if (!p_next)
                break;

            p = abcdk_strtok(&p_next, ":");
            if (!p)
                continue;

            /*用户名称长度必须相同。*/
            if (p_next - p != strlen(digest_req->pstrs[0]))
                continue;

            /*比较用户名。*/
            if (abcdk_strncmp(p, digest_req->pstrs[0], p_next - p, 0) != 0)
                continue;

            /*找到密码。*/
            p = abcdk_strtok(&p_next, ":");
            if (!p)
                continue;

            if (p_next - p <= 0)
                continue;

            abcdk_md5_reset(http_p->md5);
            abcdk_md5_update(http_p->md5, digest_req->pstrs[0], strlen(digest_req->pstrs[0]));
            abcdk_md5_update(http_p->md5, ":", 1);
            abcdk_md5_update(http_p->md5, digest_req->pstrs[4], strlen(digest_req->pstrs[4]));
            abcdk_md5_update(http_p->md5, ":", 1);
            abcdk_md5_update(http_p->md5, p, p_next - p);
            abcdk_md5_final(http_p->md5, digest_md5_buf);
            abcdk_bin2hex(digest_ha1, digest_md5_buf, 16, 0);

            abcdk_md5_reset(http_p->md5);
            abcdk_md5_update(http_p->md5, http_p->method, strlen(http_p->method));
            abcdk_md5_update(http_p->md5, ":", 1);
            abcdk_md5_update(http_p->md5, digest_req->pstrs[2], strlen(digest_req->pstrs[2]));
            abcdk_md5_final(http_p->md5, digest_md5_buf);
            abcdk_bin2hex(digest_ha2, digest_md5_buf, 16, 0);

            abcdk_md5_reset(http_p->md5);
            abcdk_md5_update(http_p->md5, digest_ha1, 32);
            abcdk_md5_update(http_p->md5, ":", 1);
            abcdk_md5_update(http_p->md5, digest_req->pstrs[1], strlen(digest_req->pstrs[1]));
            abcdk_md5_update(http_p->md5, ":", 1);
            abcdk_md5_update(http_p->md5, digest_ha2, 32);
            abcdk_md5_final(http_p->md5, digest_md5_buf);
            abcdk_bin2hex(digest_rsp, digest_md5_buf, 16, 0);

            if (abcdk_strcmp(digest_rsp, digest_req->pstrs[3], 0) == 0)
            {
                chk = 0;
                break;
            }
        }

        abcdk_object_unref(&digest_req);
        if (chk == 0)
            return 0;
    }

final_error:

    nonce = rand();
    nonce = nonce << 32 | rand();

    abcdk_comm_post_format(node, 1000,
                           "HTTP/1.1 %s\r\n"
                           "Server: %s\r\n"
                           "Data: %s\r\n"
                           "Connection: Keep-Alive\r\n"
                           "Access-Control-Allow-Origin: %s\r\n"
                           "WWW-Authenticate: %s realm=\"%lu\", charset=utf-8, nonce=\"%lu\"\r\n"
                           "Content-Length: 0\r\n"
                           "\r\n",
                           abcdk_http_status_desc(401),
                           http_p->ctx->server_name,
                           abcdk_time_format(http_p->timefmt, NULL, http_p->ctx->loc),
                           http_p->a_c_a_o,
                           (http_p->md5 ? "Digest" : "Basic"),
                           http_p->ctx->realm,
                           nonce);

    _abcdkhttpd_logprint(node, 401, 0);

    return -1;
}

void _abcdkhttpd_reply_nobody(abcdk_comm_node_t *node, int status,const char *a_c_a_m)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    abcdk_comm_post_format(node, 300,
                           "HTTP/1.1 %s\r\n"
                           "Server: %s\r\n"
                           "Data: %s\r\n"
                           "Connection: Keep-Alive\r\n"
                           "Access-Control-Allow-Origin: %s\r\n"
                           "Access-Control-Allow-Methods: %s\r\n"
                           "Content-Length: 0\r\n"
                           "\r\n",
                           abcdk_http_status_desc(status),
                           http_p->ctx->server_name,
                           abcdk_time_format(http_p->timefmt, NULL,http_p->ctx->loc),
                           http_p->a_c_a_o,
                           a_c_a_m);

    _abcdkhttpd_logprint(node, status, 0);
}

void _abcdkhttpd_reply_dirent(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX], tmp2[PATH_MAX], tmp3[NAME_MAX];
    size_t path_len = PATH_MAX;
    char strsize[20] = {0};
    struct stat attr;
    struct tm tm;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    if (abcdk_strcmp(http_p->method, "OPTIONS", 0) == 0)
    {
        _abcdkhttpd_reply_nobody(node, 200,"POST,GET");
        return;
    }
    else if (abcdk_strcmp(http_p->method, "POST", 0) != 0 &&
        abcdk_strcmp(http_p->method, "GET", 0) != 0)
    {
        _abcdkhttpd_reply_nobody(node, 405,"");
        return;
    }

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    chk = abcdk_dirent_open(&dir, http_p->pathfile);
    if (chk != 0)
    {
        _abcdkhttpd_reply_nobody(node, 403,"");
    }
    else
    {
        abcdk_comm_post_format(node, 1000,
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
                               http_p->a_c_a_o,
                               abcdk_http_content_type_desc(".html"));

        abcdk_http_reply_chunked_format(node, 10000,
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
                                        http_p->path, http_p->path);

        while (1)
        {
            memset(tmp, 0, PATH_MAX);
            chk = abcdk_dirent_read(dir,NULL, tmp);
            if (chk != 0)
                break;

            memset(tmp2, 0, PATH_MAX);
            memset(tmp3, 0, NAME_MAX);

            abcdk_basename(tmp3, tmp);

            if (http_p->ctx->exclude_hidden_file)
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
                snprintf(strsize, 20, "%lu", attr.st_size);

            abcdk_http_reply_chunked_format(node, 10000,
                                            "<tr>\r\n"
                                            "<td><a href=\"%s%s\">%s</a></td>"
                                            "<td>%s</td>"
                                            "<td align=\"right\">%s</td>\r\n"
                                            "</tr>\r\n",
                                            tmp2, (S_ISDIR(attr.st_mode) ? "/" : ""), tmp3,
                                            abcdk_time_format(http_p->timefmt_lc, &tm,http_p->ctx->loc), strsize);
        }

        abcdk_http_reply_chunked_format(node, 1000,
                                        "</table>"
                                        "</pre>\r\n"
                                        "<hr>\r\n"
                                        "</body>\r\n"
                                        "</html>\r\n");

        abcdk_http_reply_chunked_buffer(node, NULL, 0);

        _abcdkhttpd_logprint(node, 200, -1);
    }

final:

    abcdk_tree_free(&dir);
}

void _abcdkhttpd_reply_file(abcdk_comm_node_t *node)
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

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);
    
    if (abcdk_strcmp(http_p->method, "OPTIONS", 0) == 0)
    {
        _abcdkhttpd_reply_nobody(node, 200, "POST,GET,HEAD");
        return;
    }
    else if (abcdk_strcmp(http_p->method, "POST", 0) != 0 &&
        abcdk_strcmp(http_p->method, "GET", 0) != 0 &&
        abcdk_strcmp(http_p->method, "HEAD", 0) != 0)
    {
        _abcdkhttpd_reply_nobody(node, 405,"");
        return;
    }

    abcdk_time_sec2tm(&tm, http_p->attr.st_mtim.tv_sec, 1);

    file = abcdk_mmap2(http_p->pathfile, 0, 0, 0);
    if (file)
    {
        /*保存文件大小。*/
        file_size = file->sizes[0];

#ifdef HAVE_LIBMAGIC
        if (http_p->ctx->magic_handle)
        {
            abcdk_mutex_lock(&http_p->ctx->magic_mutex, 1);
            content_type = magic_buffer(http_p->ctx->magic_handle, file->pptrs[0], file->sizes[0]);
            abcdk_mutex_unlock(&http_p->ctx->magic_mutex);
        }
#endif // HAVE_LIBMAGIC

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
            sscanf(p, "%lu-%lu", &range_s, &range_e);

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

            abcdk_comm_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: %s\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Access-Control-Allow-Origin: %s\r\n"
                                   "Content-Type: %s\r\n"
                                   "Accept-Ranges: bytes\r\n"
                                   "Content-Range: bytes %lu-%lu/%lu\r\n"
                                   "Content-Length: %lu\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Expires: 0\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(status = 206),
                                   http_p->ctx->server_name,
                                   abcdk_time_format(http_p->timefmt, &tm,http_p->ctx->loc),
                                   http_p->a_c_a_o,
                                   content_type,
                                   range_s, range_e, file_size,
                                   file->sizes[0]);
        }
        else
        {
            abcdk_comm_post_format(node, 1000,
                                   "HTTP/1.1 %s\r\n"
                                   "Server: %s\r\n"
                                   "Data: %s\r\n"
                                   "Connection: Keep-Alive\r\n"
                                   "Access-Control-Allow-Origin: %s\r\n"
                                   "Content-Type: %s\r\n"
                                   "Content-Length: %lu\r\n"
                                   "Cache-Control: no-cache\r\n"
                                   "Expires: 0\r\n"
                                   "\r\n",
                                   abcdk_http_status_desc(status = 200),
                                   http_p->ctx->server_name,
                                   abcdk_time_format(http_p->timefmt, &tm,http_p->ctx->loc),
                                   http_p->a_c_a_o,
                                   content_type,
                                   file->sizes[0]);
        }

        chk = -1;
        if (abcdk_strcmp(http_p->method, "HEAD", 0) != 0)
            chk = abcdk_comm_post(node, file);

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

void _abcdkhttpd_process(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    chk = _abcdkhttpd_check_auth(node);
    if (chk != 0)
        return;

    /*去掉路径中的“..”和“.”，以防客户端构造特殊路径绕过WEB根目录。*/
    abcdk_abspath(http_p->path);

    abcdk_dirdir(http_p->pathfile, http_p->ctx->root_path);
    abcdk_dirdir(http_p->pathfile, http_p->path);

    chk = stat(http_p->pathfile, &http_p->attr);
    if (chk != 0)
    {
        if (errno == ENOENT)
            _abcdkhttpd_reply_nobody(node, 404,"");
        else
            _abcdkhttpd_reply_nobody(node, 403,"");
    }
    else if (S_ISDIR(http_p->attr.st_mode))
    {
        _abcdkhttpd_reply_dirent(node);
    }
    else if (S_ISREG(http_p->attr.st_mode))
    {
        _abcdkhttpd_reply_file(node);
    }
    else
    {
        _abcdkhttpd_reply_nobody(node, 403,"");
    }
}

void _abcdkhttpd_parse_request(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    const char *p = NULL, *p_next = NULL;
    size_t path_len = PATH_MAX;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    http_p->line0 = abcdk_http_request_env(http_p->req, 0);
    http_p->referer = abcdk_http_request_getenv(http_p->req, "Referer");
    http_p->user_agent = abcdk_http_request_getenv(http_p->req, "User-Agent");
    http_p->range = abcdk_http_request_getenv(http_p->req, "Range");
    http_p->auth = abcdk_http_request_getenv(http_p->req, "Authorization");
    if (!http_p->auth)
        http_p->auth = abcdk_http_request_getenv(http_p->req, "Proxy-Authorization");

    if (!http_p->line0)
        goto final_error;

    p_next = http_p->line0;

    memset(http_p->method, 0, 100);
    memset(http_p->location, 0, PATH_MAX);
    memset(http_p->path, 0, PATH_MAX);
    memset(http_p->params, 0, PATH_MAX);
    memset(http_p->version, 0, 100);
    memset(http_p->pathfile, 0, PATH_MAX);

    p = abcdk_strtok(&p_next, " ");
    if(!p)
        goto final_error;

    strncpy(http_p->method, p, p_next - p);

    p = abcdk_strtok(&p_next, " ");
    if (!p)
        goto final_error;

    strncpy(http_p->location, p, p_next - p);

    p = abcdk_strtok(&p_next, " ");
    if (!p)
        goto final_error;

    strncpy(http_p->version, p, p_next - p);

    p_next = http_p->location;
    p = abcdk_strtok(&p_next, "?");
    if (!p)
        goto final_error;

    abcdk_url_decode(p, p_next - p, http_p->path, &path_len,0);

    if (p_next)
    {
        if (*p_next == '?')
            p_next += 1;
    }

    p = abcdk_strtok(&p_next, "#");
    if (p)
        strncpy(http_p->params, p, p_next - p);

    if (!http_p->ctx->a_c_a_o)
    {
        if (http_p->referer)
        {
            path_len = PATH_MAX;
            abcdk_url_decode(http_p->referer, strlen(http_p->referer), http_p->referer_de, &path_len, 1);
            if (abcdk_strncmp(http_p->referer_de, "http://", 7, 0) == 0)
            {
                strncpy(http_p->a_c_a_o, "http://", PATH_MAX);
                sscanf(http_p->referer_de + 7, "%[^/]", http_p->a_c_a_o + 7);
            }
            else if (abcdk_strncmp(http_p->referer_de, "https://", 8, 0) == 0)
            {
                strncpy(http_p->a_c_a_o, "https://", PATH_MAX);
                sscanf(http_p->referer_de + 8, "%[^/]", http_p->a_c_a_o + 8);
            }
            else
            {
                strncpy(http_p->a_c_a_o, "*", PATH_MAX);
            }
        }
        else
        {
            strncpy(http_p->a_c_a_o, "*", PATH_MAX);
        }
    }
    else
    {
        strncpy(http_p->a_c_a_o, http_p->ctx->a_c_a_o, PATH_MAX);
    }

    _abcdkhttpd_process(node);
    return;

final_error:

    abcdk_comm_set_timeout(node, 1);
    return;
}

void _abcdkhttpd_request_v1(abcdk_comm_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdkhttpd_node_t *http_p;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    if (!http_p->req)
        http_p->req = abcdk_http_request_alloc(http_p->ctx->up_max_size, http_p->ctx->up_tmp_path);

    if (!http_p->req)
    {
        *remain = 0;
        abcdk_comm_set_timeout(node, 1);
        return;
    }

    chk = abcdk_http_request_append(http_p->req, data, size, remain);
    if (chk < 0)
    {
        *remain = 0;
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        return;
    }
    else if (chk > 0)
    {
        _abcdkhttpd_parse_request(node);
        abcdk_http_request_unref(&http_p->req);
    }
}

void _abcdkhttpd_request_cb(abcdk_comm_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdkhttpd_node_t *http_p;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    /*默认处理全部数据。*/
    *remain = 0;

    if (http_p->protocol == 1)
        _abcdkhttpd_request_v1(node, data, size, remain);
    else
        abcdk_comm_set_timeout(node, 1);
}

void _abcdkhttpd_prepare_cb(abcdk_comm_node_t **node, abcdk_comm_node_t *listen)
{
    abcdk_comm_node_t *node_p;
    abcdkhttpd_node_t *http_listen_p;
    abcdkhttpd_node_t *http_p;
    abcdk_object_t *extend_p = NULL;

    http_listen_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(listen);

    node_p = abcdk_comm_alloc(http_listen_p->ctx->comm, sizeof(abcdkhttpd_node_t), 0);
    if (!node_p)
        return;

    extend_p = abcdk_comm_extend(node_p);
    http_p = (abcdkhttpd_node_t *)extend_p->pptrs[0];

    /*绑定扩展数据析构函数。*/
    abcdk_object_atfree(extend_p, _abcdkhttpd_node_destroy_cb, NULL);
    abcdk_object_unref(&extend_p);

    /*复制上下文环境指针。*/
    http_p->ctx = http_listen_p->ctx;

    /*准备完毕，返回。*/
    *node = node_p;
}

void _abcdkhttpd_accept_cb(abcdk_comm_node_t *node, int *result)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    /*设置默认返回值。*/
    *result = 0;

    /*获取远程地址。*/
    abcdk_comm_get_sockaddr_str(node, NULL, http_p->remote);

    /*设置时间格式串。*/
    http_p->timefmt = "%a, %d %b %Y %H:%M:%S GMT";
    http_p->timefmt_lc = "%Y-%m-%d %H:%M:%S";

    http_p->md5 = abcdk_md5_create();

    abcdk_log_printf(LOG_INFO, "Accept: %s", http_p->remote);
}

void _abcdkhttpd_connect_cb(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;
    SSL *ssl_p;
    void *ver_p;
    int ver_l;
    int chk;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    /*默认支持1.1 or 1.0 or 0.9。*/
    http_p->protocol = 1;

#ifdef HEADER_SSL_H
    /*如果SSL开启，检查SSL验证结果。*/
    ssl_p = abcdk_comm_ssl(node);
    if (ssl_p)
    {
        chk = SSL_get_verify_result(ssl_p);
        if (chk != X509_V_OK)
        {
            /*修改超时，使用超时检测器关闭。*/
            abcdk_comm_set_timeout(node, 1);
            return;
        }

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
        SSL_get0_alpn_selected(ssl_p, (const uint8_t **)&ver_p, &ver_l);
        if (ver_p != NULL && ver_l > 0)
            http_p->protocol = ((abcdk_strncmp("h2", ver_p, ABCDK_MIN(ver_l, 2), 0) == 0) ? 2 : 1);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
    }
#endif // HEADER_SSL_H

    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
    // abcdk_comm_send_watch(node);
}

void _abcdkhttpd_close_cb(abcdk_comm_node_t *node)
{
    abcdkhttpd_node_t *http_p;

    http_p = (abcdkhttpd_node_t *)abcdk_comm_get_extend(node);

    abcdk_log_printf(LOG_INFO, "Close: %s", http_p->remote);
}

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
int _abcdkhttpd_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                               const unsigned char *in, unsigned int inlen, void *arg)
{
    unsigned int srvlen;

    /*协议选择时，仅做指针的复制，因此这里要么用静态的变量，要么创建一个全局有效的。*/
    static unsigned char srv[] = {"\x08http/1.1\x08http/1.0\x08http/0.9"};

    /*精确的长度。*/
    srvlen = sizeof(srv) - 1;

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
    if (SSL_select_next_proto((unsigned char **)out, outlen, in, inlen, srv, srvlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}

#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H


void _abcdkhttpd_work(abcdkhttpd_t *ctx)
{
    abcdk_signal_t sig;
    abcdk_sockaddr_t addr;
    abcdk_object_t *extend_p = NULL;
    abcdkhttpd_node_t *http_p = NULL;
    const char *p, *p_next, p2;
    size_t plen, p2len;
    int chk;

    ctx->max_client = abcdk_option_get_int(ctx->args, "--max-client", 0, -1);
    ctx->server_name = abcdk_option_get(ctx->args, "--server-name", 0, SOLUTION_NAME);
    ctx->root_path = abcdk_option_get(ctx->args, "--root-path", 0, "/var/abcdk/");
    ctx->a_c_a_o = abcdk_option_get(ctx->args, "--access-control-allow-origin",0,NULL);
    ctx->listen[ABCDKHTTPD_LISTEN] = abcdk_option_get(ctx->args, "--listen", 0, NULL);
    ctx->listen[ABCDKHTTPD_LISTEN_SSL] = abcdk_option_get(ctx->args, "--listen-ssl", 0, NULL);
    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);
    ctx->up_max_size = abcdk_option_get_llong(ctx->args, "--up-max-size", 0, 4 * 1024 * 1024);
    ctx->up_tmp_path = abcdk_option_get(ctx->args, "--up-tmp-path", 0, NULL);
    ctx->exclude_hidden_file = abcdk_option_exist(ctx->args, "--exclude-hidden-file");

    for (int i = 0; i < ABCDKHTTPD_PASSWD_MAX; i++)
        ctx->passwd[i] = abcdk_option_get(ctx->args, "--passwd", i, NULL);

    ctx->realm = rand();
    ctx->realm = ctx->realm << 32 | rand();

    ctx->loc = newlocale(LC_ALL_MASK,"en_US.UTF-8",NULL);

    abcdk_mutex_init2(&ctx->magic_mutex, 0);
    ctx->comm_listen[ABCDKHTTPD_LISTEN] = NULL;
    ctx->comm_listen[ABCDKHTTPD_LISTEN_SSL] = NULL;

    if (access(ctx->root_path, R_OK) != 0)
    {
        fprintf(stderr, "'%s'目录不存在或无法访问。\n", ctx->root_path);
        goto final;
    }

    if (!ctx->listen[ABCDKHTTPD_LISTEN] && !ctx->listen[ABCDKHTTPD_LISTEN_SSL])
    {
        fprintf(stderr, "至少需要监听一个地址。\n");
        goto final;
    }

    if (ctx->up_tmp_path && access(ctx->up_tmp_path, W_OK) != 0)
    {
        fprintf(stderr, "'%s'缓存目录不存在或无法访问，忽略。\n", ctx->up_tmp_path);
        ctx->up_tmp_path = NULL;
    }

#ifdef HAVE_OPENSSL
    if (ctx->listen[ABCDKHTTPD_LISTEN_SSL] && ctx->cert_file && ctx->key_file)
    {
        ctx->ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, ctx->ca_file, ctx->ca_path, (ctx->ca_path ? 2 : 0));
        if (!ctx->ssl_ctx)
        {
            fprintf(stderr, "加载CA证书错误。\n");
            goto final;
        }

        chk = abcdk_openssl_ssl_ctx_load_crt(ctx->ssl_ctx, ctx->cert_file, ctx->key_file, NULL);
        if (chk != 0)
        {
            fprintf(stderr, "加载证书或私钥错误。\n");
            goto final;
        }

        if (ctx->ca_file || ctx->ca_path)
            SSL_CTX_set_verify(ctx->ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
        SSL_CTX_set_alpn_select_cb(ctx->ssl_ctx, _abcdkhttpd_alpn_select_cb, NULL);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
    }
#endif // HAVE_OPENSSL

#ifdef HAVE_LIBMAGIC
    ctx->magic_handle = magic_open(MAGIC_MIME | MAGIC_SYMLINK);
    if (ctx->magic_handle)
        magic_load(ctx->magic_handle, NULL);
#endif // HAVE_LIBMAGIC

    ctx->comm = abcdk_comm_start(ctx->max_client, -1);
    if (!ctx->comm)
    {
        fprintf(stderr, "内存错误。\n");
        goto final;
    }

    for (int i = 0; i < 2; i++)
    {
        if (!ctx->listen[i])
            break;

        chk = abcdk_sockaddr_from_string(&addr, ctx->listen[i], 0);
        if (chk != 0)
        {
            fprintf(stderr, "监听地址错误。\n");
            goto final;
        }

        ctx->comm_listen[i] = abcdk_comm_alloc(ctx->comm, sizeof(abcdkhttpd_node_t), 1);
        if (!ctx->comm_listen[i])
        {
            fprintf(stderr, "内存错误。\n");
            goto final;
        }

        extend_p = abcdk_comm_extend(ctx->comm_listen[i]);
        http_p = (abcdkhttpd_node_t *)extend_p->pptrs[0];

        /*绑定扩展数据析构函数。*/
        abcdk_object_atfree(extend_p, _abcdkhttpd_node_destroy_cb, NULL);
        abcdk_object_unref(&extend_p);

        /*绑定上下文环境指针。*/
        http_p->ctx = ctx;

        abcdk_comm_callback_t cb = {_abcdkhttpd_prepare_cb, _abcdkhttpd_event_cb, _abcdkhttpd_request_cb};
        chk = abcdk_comm_listen(ctx->comm_listen[i], (i == ABCDKHTTPD_LISTEN_SSL ? ctx->ssl_ctx : NULL), &addr, &cb);
        if (chk != 0)
        {
            fprintf(stderr, "监听错误，无权限或端口被占用。\n");
            goto final;
        }
    }

    /*填充信号及回调函数。*/
    sig.opaque = NULL;
    sig.signal_cb = _abcdkhttpd_signal_cb;
    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGTRAP);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    /*等待退出信号。*/
    abcdk_sigwaitinfo(&sig, -1);
    // sleep(30);

final:

    abcdk_comm_stop(&ctx->comm);

    for (int i = 0; i < 2; i++)
        abcdk_comm_unref(&ctx->comm_listen[i]);

#ifdef HAVE_OPENSSL
    abcdk_openssl_ssl_ctx_free(&ctx->ssl_ctx);
#endif // HAVE_OPENSSL

#ifdef HAVE_LIBMAGIC
    if (ctx->magic_handle)
        magic_close(ctx->magic_handle);
    ctx->magic_handle = NULL;
#endif // HAVE_LIBMAGIC

    abcdk_mutex_destroy(&ctx->magic_mutex);

    freelocale(ctx->loc);
}

int abcdk_tool_httpd(abcdk_option_t *args)
{
    abcdkhttpd_t ctx = {0};

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
