/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

/*简单的代理。*/
typedef struct _abcdk_proxy
{
    int errcode;
    abcdk_option_t *args;

    /*日志。*/
    abcdk_logger_t *logger;

    /*服务器名称。*/
    const char *server_name;

    /*服务器领域。*/
    const char *server_realm;

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

    /*上级服务器地址。*/
    const char *up_link;

    /*空闲超时(秒)。*/
    time_t stimeout;

    /*IO环境。*/
    abcdk_asynctcp_t *io_ctx;

    /*监听对象。*/
    abcdk_asynctcp_node_t *listen_p;

    /*SSL监听对象。*/
    abcdk_asynctcp_node_t *listen_ssl_p;

} abcdk_proxy_t;

/*代理节点。*/
typedef struct _abcdk_proxy_node
{
    /*父级。*/
    abcdk_proxy_t *father;

    /*追踪ID。*/
    uint64_t tid;

    /*标志。0：监听，1：服务端，2：客户端。*/
    int flag;

    /*协议。0：无效，1：代理，2：隧道。*/
    int protocol;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*SSL连接标志。*/
    int ssl_ok;

    /*时间环境*/
    locale_t loc_ctx;

    /*SSL环境。*/
    SSL_CTX *ssl_ctx;

    /*隧道CA证书。*/
    const char *tunnel_cafile;

    /*隧道CA路径。*/
    const char *tunnel_capath;

    /*隧道证书。*/
    const char *tunnel_cert;

    /*隧道私钥。*/
    const char *tunnel_key;

    /*隧道加密(上行)。*/
    abcdk_enigma_t *tunnel_encrypt;

    /*隧道解密(下行)。*/
    abcdk_enigma_t *tunnel_decrypt;

    /*隧道。*/
    abcdk_asynctcp_node_t *tunnel;

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*方法。*/
    abcdk_object_t *method;

    /*脚本。*/
    abcdk_object_t *script;

    /*版本。*/
    abcdk_object_t *version;

    /*上级地址。*/
    abcdk_object_t *up_link;

    
    
} abcdk_proxy_node_t;

static void _abcdk_proxy_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的代理服务器。\n");

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
    fprintf(stderr, "\t\t服务器名称。默认：%s\n", SOLUTION_NAME);

    fprintf(stderr, "\n\t--auth-path < PATH >\n");
    fprintf(stderr, "\t\t授权存储路径。注：文件名为账号名，文件内容为密码。\n");

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
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--up-link < URL >\n");
    fprintf(stderr, "\t\t上行地址。\n");

    fprintf(stderr, "\n\t\thttp://DOMAIN[:PORT]\n");
    fprintf(stderr, "\t\thttps://DOMAIN[:PORT]\n");
}

static void _abcdk_proxy_node_destroy_cb(void *userdata)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)userdata;
    if (!node_ctx_p)
        return;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_ctx_free(&node_ctx_p->ssl_ctx);
#endif // HEADER_SSL_H

    if(node_ctx_p->loc_ctx)
        freelocale(node_ctx_p->loc_ctx);

    abcdk_asynctcp_unref(&node_ctx_p->tunnel);

    abcdk_receiver_unref(&node_ctx_p->req_data);
    abcdk_object_unref(&node_ctx_p->method);
    abcdk_object_unref(&node_ctx_p->script);
    abcdk_object_unref(&node_ctx_p->version);
    abcdk_object_unref(&node_ctx_p->up_link);
    abcdk_enigma_free(&node_ctx_p->tunnel_decrypt);
    abcdk_enigma_free(&node_ctx_p->tunnel_encrypt);
}

static abcdk_asynctcp_node_t *_abcdk_proxy_node_alloc(abcdk_proxy_t *ctx)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_proxy_node_t *node_ctx_p;

    node_p = abcdk_asynctcp_alloc(ctx->io_ctx, sizeof(abcdk_proxy_node_t), _abcdk_proxy_node_destroy_cb);
    if(!node_p)
        return NULL;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->tid = abcdk_sequence_num();
    node_ctx_p->flag = 0;
    node_ctx_p->protocol = 0;
    node_ctx_p->loc_ctx = newlocale(LC_ALL_MASK,"en_US.UTF-8",NULL);

    return node_p;
}

static void _abcdk_proxy_trace_output(abcdk_asynctcp_node_t *node,int type, const char* fmt,...)
{
    abcdk_proxy_node_t *node_ctx_p;
    char new_tname[18] = {0}, old_tname[18] = {0};

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    snprintf(new_tname, 16, "%x", node_ctx_p->tid);

    pthread_getname_np(pthread_self(), old_tname, 18);
    pthread_setname_np(pthread_self(), new_tname);

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_voutput(type, fmt, vp);
    va_end(vp);

    pthread_setname_np(pthread_self(), old_tname);
}

static void _abcdk_proxy_prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_proxy_node_t *node_ctx_p;
    abcdk_proxy_node_t *listen_ctx_p;

    listen_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(listen);

    node_p = _abcdk_proxy_node_alloc(listen_ctx_p->father);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->flag = 1;

    *node = node_p;
}

static void _abcdk_httpd_event_connect(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (node_ctx_p->flag == 2)
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    ssl_p = abcdk_asynctcp_ssl(node);
    if (!ssl_p)
        goto END;

    /*检查SSL验证结果。*/
#ifdef HEADER_SSL_H
    chk = SSL_get_verify_result(ssl_p);
    if (chk != X509_V_OK)
    {
        _abcdk_proxy_trace_output(node, LOG_INFO, "验证('%s')的证书失败，证书已过期或未生效。", node_ctx_p->remote_addr);

        /*修改超时，使用超时检测器关闭。*/
        abcdk_asynctcp_set_timeout(node, 1);
        return;
    }

    /*标记SSL连接OK。*/
    node_ctx_p->ssl_ok = 1;
#endif // HEADER_SSL_H

END:

    _abcdk_proxy_trace_output(node, LOG_INFO, "本机与'%s'建立%s连接。", node_ctx_p->remote_addr,(node_ctx_p->ssl_ok ? "安全" : "普通"));

    /*设置超时。*/
    abcdk_asynctcp_set_timeout(node, node_ctx_p->father->stimeout * 1000);

    if (node_ctx_p->flag == 1)
    {
        /*设置默认协议。*/
        node_ctx_p->protocol = 1;

        /*如果存在上级，则转换为隧道协议。*/
        if (node_ctx_p->father->up_link)
            node_ctx_p->protocol = 2;
    }
    else if (node_ctx_p->flag == 2)
    {   
        /*连接上级，设置为隧道协议。*/
        node_ctx_p->protocol = 2;
    }
    
    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _abcdk_proxy_event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (event == ABCDK_ASYNCTCP_EVENT_ACCEPT)
    {
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

        *result = 0;
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CONNECT)
    {
        _abcdk_httpd_event_connect(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
    {
        // /*转发送数据完成，监听隧道另外一端的数据。*/
        // if (node_ctx_p->tunnel)
        //     abcdk_asynctcp_recv_watch(node_ctx_p->tunnel);

        // /*转发送数据完成，监听隧道应答数据。*/
        // abcdk_asynctcp_recv_watch(node);

    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CLOSE || event == ABCDK_ASYNCTCP_EVENT_INTERRUPT)
    {
        if (!node_ctx_p->remote_addr[0])
            abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

        /*一定要在这里关闭另一端的隧道，否则因引用计数未减少，从而造成内存泄漏。*/
        if (node_ctx_p->tunnel)
        {
            abcdk_asynctcp_set_timeout(node_ctx_p->tunnel, 1);
            abcdk_asynctcp_unref(&node_ctx_p->tunnel);
        }

        _abcdk_proxy_trace_output(node, LOG_INFO, "本机与'%s'连接已经断开。", node_ctx_p->remote_addr);
    }
}


static int _abcdk_proxy_load_auth(void *opaque,const char *user,char pawd[160])
{
    abcdk_proxy_node_t *node_ctx_p;
    char tmp[PATH_MAX] = {0};
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)opaque;

    abcdk_dirdir(tmp,node_ctx_p->father->auth_path);
    abcdk_dirdir(tmp,user);

    chk = abcdk_load(tmp,pawd,160,0);
    if(chk >0)
        return 0;
    else if(chk == 0)
        return -2;
    
    return -1;
}

static int _abcdk_proxy_check_auth(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    abcdk_option_t *auth_opt = NULL;
    const char *auth_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (!node_ctx_p->father->auth_path)
        return 0;

    auth_p = abcdk_receiver_header_line_getenv(node_ctx_p->req_data, "Proxy-Authorization",':');

    /*如果客户端没携带授权，则提示客户端提交授权。*/
    if (!auth_p)
        goto ERR;

    abcdk_http_parse_auth(&auth_opt, auth_p);
    abcdk_option_set(auth_opt, "http-method", node_ctx_p->method->pstrs[0]);

    chk = abcdk_http_check_auth(auth_opt, _abcdk_proxy_load_auth, node_ctx_p);
    abcdk_option_free(&auth_opt);

    if (chk == 0)
        return 0;

ERR:

    abcdk_asynctcp_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: %s\r\n"
                               "Data: %s\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Proxy-Authenticate: Digest realm=\"%s\", charset=utf-8, nonce=\"%llu\"\r\n"
                               "Content-Length: 0\r\n"
                               "\r\n",
                               abcdk_http_status_desc(407),
                               node_ctx_p->father->server_name,
                               abcdk_time_format_gmt(NULL, node_ctx_p->loc_ctx),
                               node_ctx_p->father->server_realm,
                               (uint64_t)abcdk_rand_q());

    _abcdk_proxy_trace_output(node, LOG_INFO, "Status: %s\n", abcdk_http_status_desc(407));

    return -1;
}

static void _abcdk_proxy_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain);

static void _abcdk_proxy_reply_nobody(abcdk_asynctcp_node_t *node, int status, const char *a_c_a_m)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    abcdk_asynctcp_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: %s\r\n"
                               "Data: %s\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Access-Control-Allow-Methods: %s\r\n"
                               "Content-Length: 0\r\n"
                               "\r\n",
                               abcdk_http_status_desc(status),
                               node_ctx_p->father->server_name,
                               abcdk_time_format_gmt(NULL, node_ctx_p->loc_ctx),
                               (a_c_a_m && *a_c_a_m ? a_c_a_m : "*"));

    _abcdk_proxy_trace_output(node, LOG_INFO, "Status: %s\n", abcdk_http_status_desc(status));
}

static void _abcdk_proxy_process_forward(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    abcdk_proxy_node_t *node_uplink_ctx_p;
    abcdk_sockaddr_t uplink_addr = {0};
    abcdk_asynctcp_callback_t cb = {0};

    size_t body_l;
    void *body_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    /*仅支持向上级创建隧道。*/
    if(node_ctx_p->flag != 1)
        return;

    /*上级隧道，创建一次即可。*/
    if (node_ctx_p->tunnel)
        return;

    if (node_ctx_p->protocol == 1)
    {
        if (abcdk_strcmp(node_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
            node_ctx_p->up_link = abcdk_url_create(1000,"connect://%s",node_ctx_p->script->pstrs[0]);
        else 
            node_ctx_p->up_link = abcdk_url_split(node_ctx_p->script->pstrs[0]);
    }
    else if (node_ctx_p->protocol == 2)
    {
        node_ctx_p->tunnel_cafile = node_ctx_p->father->ca_file;
        node_ctx_p->tunnel_capath = node_ctx_p->father->ca_path;
        node_ctx_p->tunnel_cert = node_ctx_p->father->cert_file;
        node_ctx_p->tunnel_key = node_ctx_p->father->key_file;

        node_ctx_p->method = abcdk_object_copyfrom("UPLINK",6);

        node_ctx_p->up_link = abcdk_url_split(node_ctx_p->father->up_link);
    }

    /* 解析上级地址，仅支持IPV4。*/
    uplink_addr.family = AF_INET;
    chk = abcdk_sockaddr_from_string(&uplink_addr, node_ctx_p->up_link->pstrs[ABCDK_URL_HOST], 1);
    if (chk != 0)
    {
        _abcdk_proxy_trace_output(node, LOG_WARNING, "上级地址'%s'无法识别。", node_ctx_p->up_link->pstrs[ABCDK_URL_HOST]);

        _abcdk_proxy_reply_nobody(node, 404, "CONNECT");
        goto ERR;
    }

    /*如果未指定端口，则按协议指定默认端口。*/
    if (!uplink_addr.addr4.sin_port)
    {
        uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(80);
        if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "https", 0) == 0 ||
            abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "wss", 0) == 0)
            uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(443);
    }

    if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "https", 0) == 0 ||
        abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "wss", 0) == 0)
    {
#ifdef HEADER_SSL_H
        node_ctx_p->ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(0, node_ctx_p->tunnel_cafile, node_ctx_p->tunnel_capath, node_ctx_p->tunnel_cert, node_ctx_p->tunnel_key, NULL);
#endif // HEADER_SSL_H
        if (!node_ctx_p->ssl_ctx)
        {
            _abcdk_proxy_reply_nobody(node, 500, "CONNECT");
            goto ERR;
        }
    }

    node_ctx_p->tunnel = _abcdk_proxy_node_alloc(node_ctx_p->father);
    if (!node_ctx_p->tunnel)
    {
        _abcdk_proxy_reply_nobody(node, 500, "CONNECT");
        goto ERR;
    }

    node_uplink_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_ctx_p->tunnel);

    node_uplink_ctx_p->father = node_ctx_p->father;
    node_uplink_ctx_p->flag = 2;
    node_uplink_ctx_p->protocol = 2;
    node_uplink_ctx_p->tunnel = abcdk_asynctcp_refer(node);

    cb.prepare_cb = _abcdk_proxy_prepare_cb;
    cb.event_cb = _abcdk_proxy_event_cb;
    cb.request_cb = _abcdk_proxy_request_cb;

    chk = abcdk_asynctcp_connect(node_ctx_p->tunnel, node_ctx_p->ssl_ctx, &uplink_addr, &cb);
    if (chk != 0)
    {
        _abcdk_proxy_trace_output(node, LOG_WARNING, "连接上级'%s'失败，网络不可达或服务未启动。", node_ctx_p->father->up_link);

        _abcdk_proxy_reply_nobody(node, 404, "CONNECT");
        goto ERR;
    }

    /*转换为代理隧道。*/
    if (node_ctx_p->protocol == 1)
        node_ctx_p->protocol = 2;

    if(abcdk_strcmp(node_ctx_p->method->pstrs[0], "UPLINK", 0) == 0)
    {
        ;/*由上级应答。*/
    }
    else if(abcdk_strcmp(node_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
    {
        _abcdk_proxy_reply_nobody(node,200,"CONNECT");
    }
    else
    {
        /*转发请求头。*/
        abcdk_asynctcp_post_format(node_ctx_p->tunnel, 256*1024, "%s %s %s\r\n", 
                                    node_ctx_p->method->pstrs[0], node_ctx_p->up_link->pstrs[ABCDK_URL_SCRIPT], node_ctx_p->version->pstrs[0]);

        for (int i = 1; i < 100; i++)
        {
            const char *p = abcdk_receiver_header_line(node_ctx_p->req_data, i);
            if (!p)
                break;

            abcdk_asynctcp_post_format(node_ctx_p->tunnel, 256*1024, "%s\r\n", p);
        }

        abcdk_asynctcp_post_buffer(node_ctx_p->tunnel, "\r\n", 2);

        /*转发请求实体。*/
        body_l = abcdk_receiver_body_length(node_ctx_p->req_data);
        if (body_l > 0)
        {
            body_p = (void*)abcdk_receiver_body(node_ctx_p->req_data, 0);
            abcdk_asynctcp_post_buffer(node_ctx_p->tunnel, body_p, body_l);
        }
    }

    return;

ERR:

    abcdk_asynctcp_set_timeout(node,1);
}

static void _abcdk_proxy_process_request(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    const char *req_line;
    size_t body_l;
    void *body_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (node_ctx_p->protocol == 1)
    {
        _abcdk_proxy_trace_output(node, LOG_INFO, "Remote: %s\n%.*s\n",
                                  node_ctx_p->remote_addr,
                                  (int)abcdk_receiver_header_length(node_ctx_p->req_data),
                                  abcdk_receiver_data(node_ctx_p->req_data, 0));

        req_line = abcdk_receiver_header_line(node_ctx_p->req_data, 0);
        if (!req_line)
            goto ERR;

        abcdk_http_parse_request_header0(req_line, &node_ctx_p->method, &node_ctx_p->script, &node_ctx_p->version);

        chk = _abcdk_proxy_check_auth(node);
        if (chk != 0)
            return;

        _abcdk_proxy_process_forward(node);

    }
    else if (node_ctx_p->protocol == 2)
    {
        _abcdk_proxy_process_forward(node);

        /*获取缓存指针和数据长度。*/
        body_l = abcdk_receiver_body_length(node_ctx_p->req_data);
        body_p = (void *)abcdk_receiver_body(node_ctx_p->req_data, 0);

        /*转发到另外一端。*/
        if(node_ctx_p->tunnel)
            abcdk_asynctcp_post_buffer(node_ctx_p->tunnel, body_p, body_l);
    }

    /*继续监听客户端数据。*/
    //abcdk_asynctcp_recv_watch(node);

    return;

ERR:

    abcdk_asynctcp_set_timeout(node,1);
}

static void _abcdk_proxy_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_proxy_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (!node_ctx_p->req_data)
    {
        if (node_ctx_p->protocol == 1)
            node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, 256 * 1024, NULL);
        else if (node_ctx_p->protocol == 2)
            node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, 256 * 1024, NULL);
        else
            goto ERR;
    }

    if (!node_ctx_p->req_data)
        goto ERR;

    chk = abcdk_receiver_append(node_ctx_p->req_data, data, size, remain);
    if (chk < 0)
        goto ERR;
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_proxy_process_request(node);



    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);
    abcdk_object_unref(&node_ctx_p->method);
    abcdk_object_unref(&node_ctx_p->script);
    abcdk_object_unref(&node_ctx_p->version);
    abcdk_object_unref(&node_ctx_p->up_link);


    /*No Error.*/
    return;

ERR:

    abcdk_asynctcp_set_timeout(node, 1);
}

static int _abcdk_proxy_start_listen(abcdk_proxy_t *ctx, int ssl)
{
    const char *listen;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_asynctcp_node_t *listen_p;
    abcdk_proxy_node_t *listen_ctx_p;
    abcdk_asynctcp_callback_t cb = {0};
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

    if (ssl)
        listen_p = ctx->listen_ssl_p = _abcdk_proxy_node_alloc(ctx);
    else
        listen_p = ctx->listen_p = _abcdk_proxy_node_alloc(ctx);

    if (!listen_p)
        return -2;

    listen_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(listen_p);

    if (ssl)
    {
#ifdef HEADER_SSL_H
        if (ctx->cert_file && ctx->key_file)
            listen_ctx_p->ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(1, ctx->ca_file, ctx->ca_path, ctx->cert_file, ctx->key_file, NULL);
#endif // HEADER_SSL_H
    }

    cb.prepare_cb = _abcdk_proxy_prepare_cb;
    cb.event_cb = _abcdk_proxy_event_cb;
    cb.request_cb = _abcdk_proxy_request_cb;

    chk = abcdk_asynctcp_listen(listen_p, listen_ctx_p->ssl_ctx, &listen_addr, &cb);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'失败，无权限或被占用。", listen);
        return -3;
    }

    return 0;
}

static void _abcdk_proxy_process(abcdk_proxy_t *ctx)
{
    const char *log_path;
    int max_client;
    int chk;

    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);
    ctx->server_name = abcdk_option_get(ctx->args, "--server-name", 0, "abcdk");
    ctx->server_realm = abcdk_option_get(ctx->args, "--server-realm", 0, "proxy");
    ctx->auth_path = abcdk_option_get(ctx->args, "--auth-path", 0, NULL);
    ctx->up_link = abcdk_option_get(ctx->args, "--up-link", 0, NULL);
    ctx->stimeout = abcdk_option_get_int(ctx->args, "--stiemout", 0, 180);
    max_client = abcdk_option_get_int(ctx->args, "--max-client", 0, 1000);
    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "proxy.log", "proxy.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

    ctx->io_ctx = abcdk_asynctcp_start(1000, -1);
    if (!ctx->io_ctx)
        goto END;

    chk = _abcdk_proxy_start_listen(ctx, 0);
    if (chk != 0)
        goto END;

#ifdef HEADER_SSL_H
    chk = _abcdk_proxy_start_listen(ctx, 1);
    if (chk != 0)
        goto END;
#endif // HEADER_SSL_H

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

END:

    abcdk_asynctcp_stop(&ctx->io_ctx);
    abcdk_asynctcp_unref(&ctx->listen_p);
    abcdk_asynctcp_unref(&ctx->listen_ssl_p);

    abcdk_trace_output(LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}

static int _abcdk_proxy_daemon_process_cb(void *opaque)
{
    abcdk_proxy_t *ctx = (abcdk_proxy_t *)opaque;

    _abcdk_proxy_process(ctx);

    return 0;
}

static void _abcdk_proxy_daemon(abcdk_proxy_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path, "proxy-daemon.log", "proxy-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, logger);

    abcdk_proc_daemon(interval, _abcdk_proxy_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_proxy(abcdk_option_t *args)
{
    abcdk_proxy_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_proxy_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdk_proxy_daemon(&ctx);
        }
        else
        {
            _abcdk_proxy_process(&ctx);
        }
    }

    return 0;
}
