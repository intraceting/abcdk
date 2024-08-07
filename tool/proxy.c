/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

/*简单的代理。*/
typedef struct _abcdkproxy
{
    int errcode;
    abcdk_option_t *args;

    /*日志。*/
    abcdk_logger_t *logger;

    /*授权路径。*/
    const char *auth_path;

    /*名字。*/
    const char *name;

    /*领域。*/
    const char *realm;

    /*监听地址。*/
    const char *listen_raw;

    /*PKI监听地址。*/
    const char *listen_pki;

    /*ENIGMA监听地址。*/
    const char *listen_enigma;

    /*PKIonENIGMA监听地址。*/
    const char *listen_pki_enigma;

    /*CA证书。*/
    const char *pki_ca_file;

    /*CA路径。*/
    const char *pki_ca_path;

    /*证书。*/
    const char *pki_cert_file;

    /*私钥。*/
    const char *pki_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int pki_check_cert;

    /*共享密钥。*/
    const char *enigma_key_file;

    /*盐的长度。*/
    int enigma_salt_size;

    /*上级地址。*/
    const char *uplink;

    /*IO环境。*/
    abcdk_asio_t *io_ctx;

    /*原始监听对象。*/
    abcdk_asio_node_t *listen_raw_p;

    /*PKI监听对象。*/
    abcdk_asio_node_t *listen_pki_p;

    /*ENIGMA监听对象。*/
    abcdk_asio_node_t *listen_enigma_p;

    /*PKIonENIGMA监听对象。*/
    abcdk_asio_node_t *listen_pki_enigma_p;


} abcdkproxy_t;

/*代理节点。*/
typedef struct _abcdkproxy_node
{
    /*父级。*/
    abcdkproxy_t *father;

    /*追踪ID。*/
    uint64_t tid;

    /*标志。0：监听，1：服务端，2：客户端。*/
    int flag;

    /*协议。0：无效，1：代理，2：隧道。*/
    int protocol;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*本机地址。*/
    char local_addr[NAME_MAX];

    /*时间环境*/
    locale_t loc_ctx;

    /*隧道。*/
    abcdk_asio_node_t *tunnel;

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
    
    
} abcdkproxy_node_t;

static void _abcdkproxy_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的代理服务器。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径。默认：/tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒)。默认：30\n");

    fprintf(stderr, "\n\t\t注1：1～60之间有效。\n");
    fprintf(stderr, "\t\t注2：此功能不支持supervisor或类似的工具。\n");

    fprintf(stderr, "\n\t--name < NAME >\n");
    fprintf(stderr, "\t\t名字。默认：%s\n", SOLUTION_NAME);

    fprintf(stderr, "\n\t--realm < NAME >\n");
    fprintf(stderr, "\t\t领域。默认：proxy\n");

    fprintf(stderr, "\n\t--auth-path < PATH >\n");
    fprintf(stderr, "\t\t授权存储路径。注：文件名为账号名，文件内容为密码。\n");

    fprintf(stderr, "\n\t--listen-raw < ADDR >\n");
    fprintf(stderr, "\t\t监听地址。\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-pki < ADDR >\n");
    fprintf(stderr, "\t\tPKI监听地址。\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--listen-enigma < ADDR >\n");
    fprintf(stderr, "\t\tENIGMA监听地址。\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-pki-enigma < ADDR >\n");
    fprintf(stderr, "\t\tPKIonENIGMA监听地址。\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t\t例：ipv4://IP:PORT\n");

#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--pki-ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书。\n");

    fprintf(stderr, "\n\t--pki-ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书，同时验证吊销列表。\n");

    fprintf(stderr, "\n\t--pki-cert-file < FILE >\n");
    fprintf(stderr, "\t\t证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--pki-key-file < FILE >\n");
    fprintf(stderr, "\t\t私钥文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--pki-check-cert < 0|1 >\n");
    fprintf(stderr, "\t\t是否验证对端证书。默认：1。\n");

    fprintf(stderr, "\n\t\t0：否\n");
    fprintf(stderr, "\t\t1：是\n");
#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--enigma-key-file < FILE >\n");
    fprintf(stderr, "\t\t共享密钥文件。\n");

    fprintf(stderr, "\n\t--enigma-salt-size < SIZE >\n");
    fprintf(stderr, "\t\t监的长度。默认：123。\n");

    fprintf(stderr, "\n\t--uplink < URL >\n");
    fprintf(stderr, "\t\t上行地址。\n");

    fprintf(stderr, "\n\t\t例：http://DOMAIN[:PORT]\n");
    fprintf(stderr, "\t\t例：https://DOMAIN[:PORT],默认端口443.\n");
    fprintf(stderr, "\t\t例：enigma://DOMAIN[:PORT],默认端口4891.\n");
    fprintf(stderr, "\t\t例：pki-enigma://DOMAIN[:PORT],默认端口4892.\n");
}

static void _abcdkproxy_node_destroy_cb(void *userdata)
{
    abcdkproxy_node_t *node_ctx_p;

    node_ctx_p = (abcdkproxy_node_t *)userdata;
    if (!node_ctx_p)
        return;

    if(node_ctx_p->loc_ctx)
        freelocale(node_ctx_p->loc_ctx);

    abcdk_asio_unref(&node_ctx_p->tunnel);

    abcdk_receiver_unref(&node_ctx_p->req_data);
    abcdk_object_unref(&node_ctx_p->method);
    abcdk_object_unref(&node_ctx_p->script);
    abcdk_object_unref(&node_ctx_p->version);
    abcdk_object_unref(&node_ctx_p->up_link);
}

static abcdk_asio_node_t *_abcdkproxy_node_alloc(abcdkproxy_t *ctx)
{
    abcdk_asio_node_t *node_p;
    abcdkproxy_node_t *node_ctx_p;

    node_p = abcdk_asio_alloc(ctx->io_ctx, sizeof(abcdkproxy_node_t), _abcdkproxy_node_destroy_cb);
    if(!node_p)
        return NULL;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->tid = abcdk_sequence_num();
    node_ctx_p->flag = 0;
    node_ctx_p->protocol = 0;
    node_ctx_p->loc_ctx = newlocale(LC_ALL_MASK,"en_US.UTF-8",NULL);

    return node_p;
}

static void _abcdkproxy_trace_output(abcdk_asio_node_t *node,int type, const char* fmt,...)
{
    abcdkproxy_node_t *node_ctx_p;
    char new_tname[18] = {0}, old_tname[18] = {0};

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    snprintf(new_tname, 16, "%x", abcdk_asio_get_index(node));

#ifdef __USE_GNU
    pthread_getname_np(pthread_self(), old_tname, 18);
    pthread_setname_np(pthread_self(), new_tname);
#endif //__USE_GNU

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_voutput(type, fmt, vp);
    va_end(vp);

#ifdef __USE_GNU
    pthread_setname_np(pthread_self(), old_tname);
#endif //__USE_GNU

}

static void _abcdkproxy_prepare_cb(abcdk_asio_node_t **node, abcdk_asio_node_t *listen)
{
    abcdk_asio_node_t *node_p;
    abcdkproxy_node_t *node_ctx_p;
    abcdkproxy_node_t *listen_ctx_p;
    int chk;

    listen_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(listen);

    node_p = _abcdkproxy_node_alloc(listen_ctx_p->father);
    if (!node_p)
        return;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node_p);

    node_ctx_p->flag = 1;

    *node = node_p;
}

static void _abcdk_httpd_event_connect(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    /*设置协议。*/
    if (node_ctx_p->flag == 1)
    {
        /*设置默认协议。*/
        node_ctx_p->protocol = 1;

        /*如果存在上级，则转换为隧道协议。*/
        if (node_ctx_p->father->uplink)
            node_ctx_p->protocol = 2;
    }
    else if (node_ctx_p->flag == 2)
    {   
        /*连接上级，设置为隧道协议。*/
        node_ctx_p->protocol = 2;
    }

    /*设置超时。*/
    abcdk_asio_set_timeout(node, 180 * 1000);

    _abcdkproxy_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接已建立。",node_ctx_p->local_addr,node_ctx_p->remote_addr);

    /*已连接到远端，注册读写事件。*/
    abcdk_asio_recv_watch(node);
    abcdk_asio_send_watch(node);
}

static void _abcdk_httpd_event_output(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;

    // /*转发送数据完成，监听隧道另外一端的数据。*/
    // if (node_ctx_p->tunnel)
    //     abcdk_asio_recv_watch(node_ctx_p->tunnel);

    // /*转发送数据完成，监听隧道应答数据。*/
    // abcdk_asio_recv_watch(node);
}

static void _abcdk_httpd_event_close(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;
    const char *errmsg_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    if (node_ctx_p->flag == 0)
    {
        _abcdkproxy_trace_output(node,LOG_INFO, "监听关闭，忽略。");
        return;
    }

    /*一定要在这里关闭另一端的隧道，否则因引用计数未减少，从而造成内存泄漏。*/
    if (node_ctx_p->tunnel)
    {
        abcdk_asio_set_timeout(node_ctx_p->tunnel, 1);
        abcdk_asio_unref(&node_ctx_p->tunnel);
    }

    _abcdkproxy_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接已断开。", node_ctx_p->local_addr, node_ctx_p->remote_addr);
}

static void _abcdkproxy_event_cb(abcdk_asio_node_t *node, uint32_t event, int *result)
{
    abcdkproxy_node_t *node_ctx_p;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    if (!node_ctx_p->remote_addr[0])
        abcdk_asio_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    if (!node_ctx_p->local_addr[0])
        abcdk_asio_get_sockaddr_str(node, node_ctx_p->local_addr, NULL);

    if (event == ABCDK_ASIO_EVENT_ACCEPT)
    {
        *result = 0;
    }
    else if (event == ABCDK_ASIO_EVENT_CONNECT)
    {
        _abcdk_httpd_event_connect(node);
    }
    else if (event == ABCDK_ASIO_EVENT_OUTPUT)
    {
        _abcdk_httpd_event_output(node);
    }
    else if (event == ABCDK_ASIO_EVENT_CLOSE || event == ABCDK_ASIO_EVENT_INTERRUPT)
    {
        _abcdk_httpd_event_close(node);
    }
}


static int _abcdkproxy_load_auth(void *opaque,const char *user,char pawd[160])
{
    abcdkproxy_node_t *node_ctx_p;
    char tmp[PATH_MAX] = {0};
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)opaque;

    abcdk_dirdir(tmp,node_ctx_p->father->auth_path);
    abcdk_dirdir(tmp,user);

    chk = abcdk_load(tmp,pawd,160,0);
    if(chk >0)
        return 0;
    else if(chk == 0)
        return -2;
    
    return -1;
}

static int _abcdkproxy_check_auth(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;
    abcdk_option_t *auth_opt = NULL;
    const char *auth_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    if (!node_ctx_p->father->auth_path)
        return 0;

    auth_p = abcdk_receiver_header_line_getenv(node_ctx_p->req_data, "Proxy-Authorization",':');

    /*如果客户端没携带授权，则提示客户端提交授权。*/
    if (!auth_p)
        goto ERR;

    abcdk_http_parse_auth(&auth_opt, auth_p);
    abcdk_option_set(auth_opt, "http-method", node_ctx_p->method->pstrs[0]);

    chk = abcdk_http_check_auth(auth_opt, _abcdkproxy_load_auth, node_ctx_p);
    abcdk_option_free(&auth_opt);

    if (chk == 0)
        return 0;

ERR:

    abcdk_asio_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: %s\r\n"
                               "Data: %s\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Proxy-Authenticate: Digest realm=\"%s\", charset=utf-8, nonce=\"%llu\"\r\n"
                               "Content-Length: 0\r\n"
                               "\r\n",
                               abcdk_http_status_desc(407),
                               node_ctx_p->father->name,
                               abcdk_time_format_gmt(NULL, node_ctx_p->loc_ctx),
                               node_ctx_p->father->realm,
                               (uint64_t)abcdk_rand_q());

    _abcdkproxy_trace_output(node, LOG_INFO, "Status: %s\n", abcdk_http_status_desc(407));

    return -1;
}

static void _abcdkproxy_request_cb(abcdk_asio_node_t *node, const void *data, size_t size, size_t *remain);

static void _abcdkproxy_reply_nobody(abcdk_asio_node_t *node, int status)
{
    abcdkproxy_node_t *node_ctx_p;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    abcdk_asio_post_format(node, 1000,
                               "HTTP/1.1 %s\r\n"
                               "Server: %s\r\n"
                               "Data: %s\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Access-Control-Allow-Methods: *\r\n"
                               "Content-Length: 0\r\n"
                               "\r\n",
                               abcdk_http_status_desc(status),
                               node_ctx_p->father->name,
                               abcdk_time_format_gmt(NULL, node_ctx_p->loc_ctx));

    _abcdkproxy_trace_output(node, LOG_INFO, "Status: %s\n", abcdk_http_status_desc(status));
}

static void _abcdkproxy_process_forward(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;
    abcdkproxy_node_t *node_uplink_ctx_p;
    abcdk_sockaddr_t uplink_addr = {0};
    abcdk_asio_config_t asio_cfg = {0};

    size_t body_l;
    void *body_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    /*仅支持向上级创建隧道。*/
    if(node_ctx_p->flag != 1)
        return;

    /*上级隧道，创建一次即可。*/
    if (node_ctx_p->tunnel)
        return;

    node_ctx_p->tunnel = _abcdkproxy_node_alloc(node_ctx_p->father);
    if (!node_ctx_p->tunnel)
    {
        _abcdkproxy_reply_nobody(node, 500);
        goto ERR;
    }
    
    node_uplink_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node_ctx_p->tunnel);

    node_uplink_ctx_p->father = node_ctx_p->father;
    node_uplink_ctx_p->flag = 2;
    node_uplink_ctx_p->protocol = 2;
    node_uplink_ctx_p->tunnel = abcdk_asio_refer(node);//上级关联到下级。

    if (node_ctx_p->protocol == 1)
    {
        /*设置默认安全方案。*/
        asio_cfg.ssl_scheme = ABCDK_ASIO_SSL_SCHEME_RAW;

        /*不检查对端证书。因为代理服务器可能未及时更新，无法验证所有证书的有效性。*/
        asio_cfg.pki_check_cert = 0;
        
        if (abcdk_strcmp(node_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
            node_ctx_p->up_link = abcdk_url_create(1000,"connect://%s",node_ctx_p->script->pstrs[0]);
        else 
            node_ctx_p->up_link = abcdk_url_split(node_ctx_p->script->pstrs[0]);
        
    }
    else if (node_ctx_p->protocol == 2)
    {
        asio_cfg.ssl_scheme = ABCDK_ASIO_SSL_SCHEME_RAW;
        
        asio_cfg.pki_ca_file = node_ctx_p->father->pki_ca_file;
        asio_cfg.pki_ca_path = node_ctx_p->father->pki_ca_path;
        asio_cfg.pki_cert_file = node_ctx_p->father->pki_cert_file;
        asio_cfg.pki_key_file = node_ctx_p->father->pki_key_file;
        asio_cfg.pki_check_cert = node_ctx_p->father->pki_check_cert;
        asio_cfg.enigma_key_file = node_ctx_p->father->enigma_key_file;
        asio_cfg.enigma_salt_size = node_ctx_p->father->enigma_salt_size;

        node_ctx_p->method = abcdk_object_copyfrom("UPLINK",6);

        node_ctx_p->up_link = abcdk_url_split(node_ctx_p->father->uplink);
    }

    /*可能发错了。*/
    if(!node_ctx_p->up_link->pstrs[ABCDK_URL_HOST])
    {
        _abcdkproxy_trace_output(node, LOG_WARNING, "上级地址不存在。");

        _abcdkproxy_reply_nobody(node, 404);
        goto ERR;
    }

    /* 解析上级地址，仅支持IPV4。*/
    uplink_addr.family = AF_INET;
    chk = abcdk_sockaddr_from_string(&uplink_addr, node_ctx_p->up_link->pstrs[ABCDK_URL_HOST], 1);
    if (chk != 0)
    {
        _abcdkproxy_trace_output(node, LOG_WARNING, "上级地址(%s)无法识别。", node_ctx_p->up_link->pstrs[ABCDK_URL_HOST]);

        _abcdkproxy_reply_nobody(node, 404);
        goto ERR;
    }

    /*如果未指定端口，则按协议指定默认端口。*/
    if (!uplink_addr.addr4.sin_port)
    {
        uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(80);
        if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "https", 0) == 0 ||
            abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "wss", 0) == 0)
        {
            uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(443);
        }
        else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "enigma", 0) == 0)
        {
            uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(4891);
        }
        else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "pki-enigma", 0) == 0)
        {
            uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(4892);
        }
    }

    /*配置安全方案。*/
    if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "https", 0) == 0 ||
        abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "wss", 0) == 0)
    {
        asio_cfg.ssl_scheme = ABCDK_ASIO_SSL_SCHEME_PKI;
    }
    else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "enigma", 0) == 0)
    {
        asio_cfg.ssl_scheme = ABCDK_ASIO_SSL_SCHEME_ENIGMA;
    }
    else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "pki-enigma", 0) == 0)
    {
        asio_cfg.ssl_scheme = ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA;
    }

    asio_cfg.prepare_cb = _abcdkproxy_prepare_cb;
    asio_cfg.event_cb = _abcdkproxy_event_cb;
    asio_cfg.request_cb = _abcdkproxy_request_cb;

    chk = abcdk_asio_connect(node_ctx_p->tunnel, &uplink_addr, &asio_cfg);
    if (chk != 0)
    {
        _abcdkproxy_trace_output(node, LOG_WARNING, "连接上级(%s)失败，网络不可达或服务未启动。", node_uplink_ctx_p->up_link->pstrs[ABCDK_URL_HOST]);

        _abcdkproxy_reply_nobody(node, 404);
        goto ERR;
    }

    /*连接到上级后，转换代理为隧道。*/
    if (node_ctx_p->protocol == 1)
        node_ctx_p->protocol = 2;

    if(abcdk_strcmp(node_ctx_p->method->pstrs[0], "UPLINK", 0) == 0)
    {
        ;/*由上级应答。*/
    }
    else if(abcdk_strcmp(node_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
    {
        _abcdkproxy_reply_nobody(node,200);
    }
    else
    {
        /*转发请求头。*/
        abcdk_asio_post_format(node_ctx_p->tunnel, 256*1024, "%s %s %s\r\n", 
                                    node_ctx_p->method->pstrs[0], node_ctx_p->up_link->pstrs[ABCDK_URL_SCRIPT], node_ctx_p->version->pstrs[0]);

        /*最多支持100行的头部转发。*/
        for (int i = 1; i < 100; i++)
        {
            const char *p = abcdk_receiver_header_line(node_ctx_p->req_data, i);
            if (!p)
                break;

            abcdk_asio_post_format(node_ctx_p->tunnel, 256*1024, "%s\r\n", p);
        }

        abcdk_asio_post_buffer(node_ctx_p->tunnel, "\r\n", 2);

        /*转发请求实体。*/
        body_l = abcdk_receiver_body_length(node_ctx_p->req_data);
        if (body_l > 0)
        {
            body_p = (void*)abcdk_receiver_body(node_ctx_p->req_data, 0);
            abcdk_asio_post_buffer(node_ctx_p->tunnel, body_p, body_l);
        }
    }

    return;

ERR:

    abcdk_asio_set_timeout(node,1);
}

static void _abcdkproxy_process_request(abcdk_asio_node_t *node)
{
    abcdkproxy_node_t *node_ctx_p;
    const char *req_line;
    size_t body_l;
    void *body_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

    if (node_ctx_p->protocol == 1)
    {
        _abcdkproxy_trace_output(node, LOG_INFO, "Remote: %s\n%.*s\n",
                                  node_ctx_p->remote_addr,
                                  (int)abcdk_receiver_header_length(node_ctx_p->req_data),
                                  abcdk_receiver_data(node_ctx_p->req_data, 0));

        req_line = abcdk_receiver_header_line(node_ctx_p->req_data, 0);
        if (!req_line)
            goto ERR;

        abcdk_http_parse_request_header0(req_line, &node_ctx_p->method, &node_ctx_p->script, &node_ctx_p->version);

        chk = _abcdkproxy_check_auth(node);
        if (chk != 0)
            return;

        _abcdkproxy_process_forward(node);

    }
    else if (node_ctx_p->protocol == 2)
    {
        _abcdkproxy_process_forward(node);

        /*获取缓存指针和数据长度。*/
        body_l = abcdk_receiver_body_length(node_ctx_p->req_data);
        body_p = (void *)abcdk_receiver_body(node_ctx_p->req_data, 0);

        /*转发到另外一端。*/
        if(node_ctx_p->tunnel)
            abcdk_asio_post_buffer(node_ctx_p->tunnel, body_p, body_l);
    }

    /*继续监听客户端数据。*/
    //abcdk_asio_recv_watch(node);

    return;

ERR:

    abcdk_asio_set_timeout(node,1);
}

static void _abcdkproxy_request_cb(abcdk_asio_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdkproxy_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node);

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

    _abcdkproxy_process_request(node);



    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);
    abcdk_object_unref(&node_ctx_p->method);
    abcdk_object_unref(&node_ctx_p->script);
    abcdk_object_unref(&node_ctx_p->version);
    abcdk_object_unref(&node_ctx_p->up_link);


    /*No Error.*/
    return;

ERR:

    abcdk_asio_set_timeout(node, 1);
}

static int _abcdkproxy_start_listen(abcdkproxy_t *ctx, int ssl_scheme)
{
    const char *listen_p = NULL;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_asio_node_t *node_p = NULL;
    abcdkproxy_node_t *node_ctx_p = NULL;
    abcdk_asio_config_t asio_cfg = {0};
    int chk;

    if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
        listen_p = ctx->listen_raw;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        listen_p = ctx->listen_pki;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        listen_p = ctx->listen_enigma;
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        listen_p = ctx->listen_pki_enigma;

    /*未启用。*/
    if(!listen_p)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen_p, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'无法识别。", listen_p);
        return -1;
    }

    if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
        node_p = ctx->listen_raw_p = _abcdkproxy_node_alloc(ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
        node_p = ctx->listen_pki_p = _abcdkproxy_node_alloc(ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
        node_p = ctx->listen_enigma_p = _abcdkproxy_node_alloc(ctx);
    else if (ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
        node_p = ctx->listen_pki_enigma_p = _abcdkproxy_node_alloc(ctx);

    if (!node_p)
        return -2;

    node_ctx_p = (abcdkproxy_node_t *)abcdk_asio_get_userdata(node_p);

    asio_cfg.ssl_scheme = ssl_scheme;
    asio_cfg.pki_ca_file = ctx->pki_ca_file;
    asio_cfg.pki_ca_path = ctx->pki_ca_path;
    asio_cfg.pki_cert_file = ctx->pki_cert_file;
    asio_cfg.pki_key_file = ctx->pki_key_file;
    asio_cfg.pki_check_cert = ctx->pki_check_cert;
    asio_cfg.enigma_key_file = ctx->enigma_key_file;
    asio_cfg.enigma_salt_size = ctx->enigma_salt_size;

    asio_cfg.prepare_cb = _abcdkproxy_prepare_cb;
    asio_cfg.event_cb = _abcdkproxy_event_cb;
    asio_cfg.request_cb = _abcdkproxy_request_cb;

    chk = abcdk_asio_listen(node_p, &listen_addr, &asio_cfg);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'失败，无权限或被占用。", listen_p);
        return -3;
    }

    return 0;
}

static void _abcdkproxy_process(abcdkproxy_t *ctx)
{
    const char *log_path;
    int chk;

    ctx->name = abcdk_option_get(ctx->args, "--name", 0, SOLUTION_NAME);
    ctx->realm = abcdk_option_get(ctx->args, "--realm", 0, "proxy");
    ctx->auth_path = abcdk_option_get(ctx->args, "--auth-path", 0, NULL);

    ctx->listen_raw = abcdk_option_get(ctx->args, "--listen-raw", 0, NULL);
    ctx->listen_pki = abcdk_option_get(ctx->args, "--listen-pki", 0, NULL);
    ctx->listen_enigma = abcdk_option_get(ctx->args, "--listen-enigma", 0, NULL);
    ctx->listen_pki_enigma = abcdk_option_get(ctx->args, "--listen-pki-enigma", 0, NULL);

    ctx->pki_ca_file = abcdk_option_get(ctx->args, "--pki-ca-file", 0, NULL);
    ctx->pki_ca_path = abcdk_option_get(ctx->args, "--pki-ca-path", 0, NULL);
    ctx->pki_cert_file = abcdk_option_get(ctx->args, "--pki-cert-file", 0, NULL);
    ctx->pki_key_file = abcdk_option_get(ctx->args, "--pki-key-file", 0, NULL);
    ctx->pki_check_cert = abcdk_option_get_int(ctx->args, "--pki-check-cert", 0, 1);

    ctx->enigma_key_file = abcdk_option_get(ctx->args, "--enigma-key-file", 0, NULL);
    ctx->enigma_salt_size = abcdk_option_get_int(ctx->args, "--enigma-salt-size", 0, 123);

    /*修复不支持的配置。*/
    ctx->enigma_key_file = (ctx->enigma_key_file?ctx->enigma_key_file:"");
    ctx->enigma_salt_size = ABCDK_CLAMP(ctx->enigma_salt_size, 0, 256);
    
    ctx->uplink = abcdk_option_get(ctx->args, "--uplink", 0, NULL);
    
    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "proxy.log", "proxy.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

    ctx->io_ctx = abcdk_asio_start(1000, -1);
    if (!ctx->io_ctx)
        goto END;

    chk = _abcdkproxy_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_RAW);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_PKI);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_ENIGMA);
    if (chk != 0)
        goto END;

    chk = _abcdkproxy_start_listen(ctx, ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA);
    if (chk != 0)
        goto END;

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

END:

    abcdk_asio_stop(&ctx->io_ctx);
    abcdk_asio_unref(&ctx->listen_raw_p);
    abcdk_asio_unref(&ctx->listen_pki_p);
    abcdk_asio_unref(&ctx->listen_enigma_p);
    abcdk_asio_unref(&ctx->listen_pki_enigma_p);

    abcdk_trace_output(LOG_INFO, "停止。");

    /*关闭日志。*/
    abcdk_logger_close(&ctx->logger);
}

static int _abcdkproxy_daemon_process_cb(void *opaque)
{
    abcdkproxy_t *ctx = (abcdkproxy_t *)opaque;

    _abcdkproxy_process(ctx);

    return 0;
}

static void _abcdkproxy_daemon(abcdkproxy_t *ctx)
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

    abcdk_proc_daemon(interval, _abcdkproxy_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_proxy(abcdk_option_t *args)
{
    abcdkproxy_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkproxy_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdkproxy_daemon(&ctx);
        }
        else
        {
            _abcdkproxy_process(&ctx);
        }
    }

    return 0;
}
