/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

#define ABCDK_PROXY_SSL_SCHEME_RAW       0
#define ABCDK_PROXY_SSL_SCHEME_OPENSSL   1
#define ABCDK_PROXY_SSL_SCHEME_EASYSSL   2

/*简单的代理。*/
typedef struct _abcdk_proxy
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

    /*OPENSSL监听地址。*/
    const char *listen_openssl;

    /*EASYSSL监听地址。*/
    const char *listen_easyssl;

    /*CA证书。*/
    const char *openssl_ca_file;

    /*CA路径。*/
    const char *openssl_ca_path;

    /*证书。*/
    const char *openssl_cert_file;

    /*私钥。*/
    const char *openssl_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int openssl_check_cert;

    /*共享密钥。*/
    const char *easyssl_key_file;

    /*盐的长度。*/
    int easyssl_salt_size;

    /*上级地址。*/
    const char *uplink;

    /*IO环境。*/
    abcdk_asynctcp_t *io_ctx;

    /*服务器监听对象。*/
    abcdk_asynctcp_node_t *listen_raw_p;

    /*服务器OPENSSL监听对象。*/
    abcdk_asynctcp_node_t *listen_openssl_p;

    /*服务器EASYSSL监听对象。*/
    abcdk_asynctcp_node_t *listen_easyssl_p;


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

    /*本机地址。*/
    char local_addr[NAME_MAX];

    /*时间环境*/
    locale_t loc_ctx;

    /*安全方案。*/
    int ssl_scheme;

    /*OPENSSL环境。*/
    SSL_CTX *openssl_ctx;

    /*EASYSSL环境。*/
    abcdk_easyssl_t *easyssl_ctx;

    /*CA证书。*/
    const char *openssl_ca_file;

    /*CA路径。*/
    const char *openssl_ca_path;

    /*证书。*/
    const char *openssl_cert_file;

    /*私钥。*/
    const char *openssl_key_file;

    /*是否验证对端证书。0 否，!0 是。*/
    int openssl_check_cert;

    /*共享密钥。*/
    const char *easyssl_key_file;

    /*盐长度。*/
    int easyssl_salt_size;

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

    fprintf(stderr, "\n\t\t例：ipv4://IP:PORT\n");
    fprintf(stderr, "\t\t例：ipv6://[IP]:PORT\n");
    fprintf(stderr, "\t\t例：ipv6://IP,PORT\n");
#ifdef HEADER_SSL_H
    fprintf(stderr, "\n\t--listen-openssl < ADDR >\n");
    fprintf(stderr, "\t\tOPENSSL监听地址。\n");

    fprintf(stderr, "\n\t--openssl-ca-file < FILE >\n");
    fprintf(stderr, "\t\tCA证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书。\n");

    fprintf(stderr, "\n\t--openssl-ca-path < PATH >\n");
    fprintf(stderr, "\t\tCA证书路径。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式，并且要求客户提供证书，同时验证吊销列表。\n");

    fprintf(stderr, "\n\t--openssl-cert-file < FILE >\n");
    fprintf(stderr, "\t\t证书文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--openssl-key-file < FILE >\n");
    fprintf(stderr, "\t\t私钥文件。\n");

    fprintf(stderr, "\n\t\t注：仅支持PEM格式。\n");

    fprintf(stderr, "\n\t--openssl-check-cert < 0|1 >\n");
    fprintf(stderr, "\t\t是否验证对端证书。默认：1。\n");

    fprintf(stderr, "\n\t\t0：否\n");
    fprintf(stderr, "\t\t1：是\n");

#endif // HEADER_SSL_H

    fprintf(stderr, "\n\t--easyssl-key-file < FILE >\n");
    fprintf(stderr, "\t\t共享密钥文件。\n");

    fprintf(stderr, "\n\t--easyssl-salt-size < SIZE >\n");
    fprintf(stderr, "\t\t监的长度。默认：123。\n");

    fprintf(stderr, "\n\t--tunnel-uplink < URL >\n");
    fprintf(stderr, "\t\t隧道上行地址。\n");

    fprintf(stderr, "\n\t\t例：http://DOMAIN[:PORT]\n");
    fprintf(stderr, "\t\t例：https://DOMAIN[:PORT]\n");
    fprintf(stderr, "\t\t例：easys://DOMAIN[:PORT]\n");
}

static void _abcdk_proxy_node_destroy_cb(void *userdata)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)userdata;
    if (!node_ctx_p)
        return;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_ctx_free(&node_ctx_p->openssl_ctx);
#endif // HEADER_SSL_H

    abcdk_easyssl_destroy(&node_ctx_p->easyssl_ctx);

    if(node_ctx_p->loc_ctx)
        freelocale(node_ctx_p->loc_ctx);

    abcdk_asynctcp_unref(&node_ctx_p->tunnel);

    abcdk_receiver_unref(&node_ctx_p->req_data);
    abcdk_object_unref(&node_ctx_p->method);
    abcdk_object_unref(&node_ctx_p->script);
    abcdk_object_unref(&node_ctx_p->version);
    abcdk_object_unref(&node_ctx_p->up_link);
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
    int chk;

    listen_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(listen);

    node_p = _abcdk_proxy_node_alloc(listen_ctx_p->father);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->flag = 1;
    node_ctx_p->ssl_scheme = listen_ctx_p->ssl_scheme;

    if(node_ctx_p->ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
    {
        chk = abcdk_asynctcp_upgrade2openssl(node_p,listen_ctx_p->openssl_ctx,listen_ctx_p->father->openssl_check_cert);
        if(chk != 0)
            abcdk_asynctcp_unref(&node_p);
    }
    else if(node_ctx_p->ssl_scheme == ABCDK_PROXY_SSL_SCHEME_EASYSSL)
    {
        node_ctx_p->easyssl_ctx = abcdk_easyssl_create_from_file(node_ctx_p->father->easyssl_key_file, ABCDK_EASYSSL_SCHEME_ENIGMA,
                                                                 ABCDK_CLAMP(node_ctx_p->father->easyssl_salt_size, 0, 256));
        if (!node_ctx_p->easyssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载共享密钥失败，无法创建SSL环境。");
            abcdk_asynctcp_unref(&node_p);
        }
        else 
        {
            chk = abcdk_asynctcp_upgrade2easyssl(node_p,node_ctx_p->easyssl_ctx);
            if(chk != 0)
                abcdk_asynctcp_unref(&node_p);
        }
    }

    *node = node_p;
}

static void _abcdk_httpd_event_connect(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if(node_ctx_p->ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
    {
#ifdef HEADER_SSL_H
        ssl_p = abcdk_asynctcp_openssl_ctx(node);

        X509 *cert = SSL_get_peer_certificate(ssl_p);
        if(cert)
        {
            abcdk_object_t *info = abcdk_openssl_dump_crt(cert);
            if(info)
            {
                _abcdk_proxy_trace_output(node,LOG_INFO,"远端(%s)的证书信息：\n%s",node_ctx_p->remote_addr,info->pstrs[0]);
                abcdk_object_unref(&info);
            }

            X509_free(cert);
        }
#endif // HEADER_SSL_H
    }

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
    abcdk_asynctcp_set_timeout(node, 180 * 1000);

    _abcdk_proxy_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接已建立(SSL-scheme=%d)。",node_ctx_p->local_addr,node_ctx_p->remote_addr,node_ctx_p->ssl_scheme);

    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _abcdk_httpd_event_output(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;

    // /*转发送数据完成，监听隧道另外一端的数据。*/
    // if (node_ctx_p->tunnel)
    //     abcdk_asynctcp_recv_watch(node_ctx_p->tunnel);

    // /*转发送数据完成，监听隧道应答数据。*/
    // abcdk_asynctcp_recv_watch(node);
}

static void _abcdk_httpd_event_close(abcdk_asynctcp_node_t *node)
{
    abcdk_proxy_node_t *node_ctx_p;
    const char *errmsg_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (node_ctx_p->flag == 0)
    {
        _abcdk_proxy_trace_output(node,LOG_INFO, "监听关闭，忽略。");
        return;
    }

    if(node_ctx_p->ssl_scheme == ABCDK_TIPC_SSL_SCHEME_OPENSSL)
    {
#ifdef HEADER_SSL_H
        ssl_p = abcdk_asynctcp_openssl_ctx(node);
        if(ssl_p && node_ctx_p->openssl_check_cert)
        {
            /*获取验证结果。*/
            chk = SSL_get_verify_result(ssl_p);
            if (chk != X509_V_OK)
                _abcdk_proxy_trace_output(node,LOG_INFO, "验证远端(%s)的证书失败(openssl_errno=%d)。", node_ctx_p->remote_addr,chk);
        }
#endif // HEADER_SSL_H
    }

    /*一定要在这里关闭另一端的隧道，否则因引用计数未减少，从而造成内存泄漏。*/
    if (node_ctx_p->tunnel)
    {
        abcdk_asynctcp_set_timeout(node_ctx_p->tunnel, 1);
        abcdk_asynctcp_unref(&node_ctx_p->tunnel);
    }

    _abcdk_proxy_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接已断开。", node_ctx_p->local_addr, node_ctx_p->remote_addr);
}

static void _abcdk_proxy_event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    if (!node_ctx_p->remote_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    if (!node_ctx_p->local_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node, node_ctx_p->local_addr, NULL);

    if (event == ABCDK_ASYNCTCP_EVENT_ACCEPT)
    {
        *result = 0;
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CONNECT)
    {
        _abcdk_httpd_event_connect(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
    {
        _abcdk_httpd_event_output(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CLOSE || event == ABCDK_ASYNCTCP_EVENT_INTERRUPT)
    {
        _abcdk_httpd_event_close(node);
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
                               node_ctx_p->father->name,
                               abcdk_time_format_gmt(NULL, node_ctx_p->loc_ctx),
                               node_ctx_p->father->realm,
                               (uint64_t)abcdk_rand_q());

    _abcdk_proxy_trace_output(node, LOG_INFO, "Status: %s\n", abcdk_http_status_desc(407));

    return -1;
}

static void _abcdk_proxy_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain);

static void _abcdk_proxy_reply_nobody(abcdk_asynctcp_node_t *node, int status)
{
    abcdk_proxy_node_t *node_ctx_p;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node);

    abcdk_asynctcp_post_format(node, 1000,
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

    node_ctx_p->tunnel = _abcdk_proxy_node_alloc(node_ctx_p->father);
    if (!node_ctx_p->tunnel)
    {
        _abcdk_proxy_reply_nobody(node, 500);
        goto ERR;
    }
    
    node_uplink_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_ctx_p->tunnel);

    node_uplink_ctx_p->father = node_ctx_p->father;
    node_uplink_ctx_p->flag = 2;
    node_uplink_ctx_p->protocol = 2;
    node_uplink_ctx_p->tunnel = abcdk_asynctcp_refer(node);//上级关联到下级。

    if (node_ctx_p->protocol == 1)
    {
        /*设置默认安全方案。*/
        node_uplink_ctx_p->ssl_scheme = ABCDK_PROXY_SSL_SCHEME_RAW;

        /*不检查对端证书。因为代理服务器可能未及时更新，无法验证所有证书的有效性。*/
        node_uplink_ctx_p->openssl_check_cert = 0;
        
        if (abcdk_strcmp(node_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
            node_ctx_p->up_link = abcdk_url_create(1000,"connect://%s",node_ctx_p->script->pstrs[0]);
        else 
            node_ctx_p->up_link = abcdk_url_split(node_ctx_p->script->pstrs[0]);
        
    }
    else if (node_ctx_p->protocol == 2)
    {
        node_uplink_ctx_p->ssl_scheme = ABCDK_PROXY_SSL_SCHEME_RAW;
        
        if(node_ctx_p->father->openssl_check_cert)
        {
            node_uplink_ctx_p->openssl_ca_file = node_ctx_p->father->openssl_ca_file;
            node_uplink_ctx_p->openssl_ca_path = node_ctx_p->father->openssl_ca_path;
        }

        node_uplink_ctx_p->openssl_cert_file = node_ctx_p->father->openssl_cert_file;
        node_uplink_ctx_p->openssl_key_file = node_ctx_p->father->openssl_key_file;
        node_uplink_ctx_p->openssl_check_cert = node_ctx_p->father->openssl_check_cert;
        node_uplink_ctx_p->easyssl_key_file = node_ctx_p->father->easyssl_key_file;
        node_uplink_ctx_p->easyssl_salt_size = node_ctx_p->father->easyssl_salt_size;

        node_ctx_p->method = abcdk_object_copyfrom("UPLINK",6);

        node_ctx_p->up_link = abcdk_url_split(node_ctx_p->father->uplink);
    }

    /*可能发错了。*/
    if(!node_ctx_p->up_link->pstrs[ABCDK_URL_HOST])
    {
        _abcdk_proxy_trace_output(node, LOG_WARNING, "上级地址不存在。");

        _abcdk_proxy_reply_nobody(node, 404);
        goto ERR;
    }

    /* 解析上级地址，仅支持IPV4。*/
    uplink_addr.family = AF_INET;
    chk = abcdk_sockaddr_from_string(&uplink_addr, node_ctx_p->up_link->pstrs[ABCDK_URL_HOST], 1);
    if (chk != 0)
    {
        _abcdk_proxy_trace_output(node, LOG_WARNING, "上级地址(%s)无法识别。", node_ctx_p->up_link->pstrs[ABCDK_URL_HOST]);

        _abcdk_proxy_reply_nobody(node, 404);
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
        else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "easys", 0) == 0)
        {
            uplink_addr.addr4.sin_port = abcdk_endian_h_to_b16(12345);
        }
    }

    /*配置安全方案。*/
    if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "https", 0) == 0 ||
        abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "wss", 0) == 0)
    {
        node_uplink_ctx_p->ssl_scheme = ABCDK_PROXY_SSL_SCHEME_OPENSSL;
    }
    else if (abcdk_strcmp(node_ctx_p->up_link->pstrs[ABCDK_URL_SCHEME], "easys", 0) == 0)
    {
        node_uplink_ctx_p->ssl_scheme = ABCDK_PROXY_SSL_SCHEME_EASYSSL;
    }

    /*根据需要建立安全环境。*/
    if (node_uplink_ctx_p->ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
    {
#ifdef HEADER_SSL_H
        node_uplink_ctx_p->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(0, node_uplink_ctx_p->openssl_ca_file, node_uplink_ctx_p->openssl_ca_path,
                                                                          node_uplink_ctx_p->openssl_cert_file, node_uplink_ctx_p->openssl_key_file, NULL);
#endif // HEADER_SSL_H
        if (!node_uplink_ctx_p->openssl_ctx)
        {
            _abcdk_proxy_reply_nobody(node, 500);
            goto ERR;
        }
     
        chk = abcdk_asynctcp_upgrade2openssl(node_ctx_p->tunnel,node_uplink_ctx_p->openssl_ctx,node_uplink_ctx_p->openssl_check_cert);
        if(chk != 0)
        {
            _abcdk_proxy_reply_nobody(node, 500);
            goto ERR;
        }
    }
    else if (node_uplink_ctx_p->ssl_scheme == ABCDK_PROXY_SSL_SCHEME_EASYSSL)
    {
        node_uplink_ctx_p->easyssl_ctx = abcdk_easyssl_create_from_file(node_uplink_ctx_p->easyssl_key_file, ABCDK_EASYSSL_SCHEME_ENIGMA,
                                                                        node_uplink_ctx_p->easyssl_salt_size);
        if (!node_uplink_ctx_p->easyssl_ctx)
        {
            _abcdk_proxy_reply_nobody(node, 500);
            goto ERR;
        }

        chk = abcdk_asynctcp_upgrade2easyssl(node_ctx_p->tunnel,node_uplink_ctx_p->easyssl_ctx);
        if(chk != 0)
        {
            _abcdk_proxy_reply_nobody(node, 500);
            goto ERR;
        }
    }

    cb.prepare_cb = _abcdk_proxy_prepare_cb;
    cb.event_cb = _abcdk_proxy_event_cb;
    cb.request_cb = _abcdk_proxy_request_cb;

    chk = abcdk_asynctcp_connect(node_ctx_p->tunnel, &uplink_addr, &cb);
    if (chk != 0)
    {
        _abcdk_proxy_trace_output(node, LOG_WARNING, "连接上级'%s'失败，网络不可达或服务未启动。", node_uplink_ctx_p->up_link->pstrs[ABCDK_URL_HOST]);

        _abcdk_proxy_reply_nobody(node, 404);
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
        _abcdk_proxy_reply_nobody(node,200);
    }
    else
    {
        /*转发请求头。*/
        abcdk_asynctcp_post_format(node_ctx_p->tunnel, 256*1024, "%s %s %s\r\n", 
                                    node_ctx_p->method->pstrs[0], node_ctx_p->up_link->pstrs[ABCDK_URL_SCRIPT], node_ctx_p->version->pstrs[0]);

        /*最多支持100行的头部转发。*/
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

static int _abcdk_proxy_start_listen(abcdk_proxy_t *ctx, int ssl_scheme)
{
    const char *listen_p = NULL;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_asynctcp_node_t *node_p = NULL;
    abcdk_proxy_node_t *node_ctx_p = NULL;
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_RAW)
        listen_p = ctx->listen_raw;
    else if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
        listen_p = ctx->listen_openssl;
    else if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_EASYSSL)
        listen_p = ctx->listen_easyssl;

    /*未启用。*/
    if(!listen_p)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen_p, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'无法识别。", listen_p);
        return -1;
    }

    if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_RAW)
        node_p = ctx->listen_raw_p = _abcdk_proxy_node_alloc(ctx);
    else if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
        node_p = ctx->listen_openssl_p = _abcdk_proxy_node_alloc(ctx);
    else if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_EASYSSL)
        node_p = ctx->listen_easyssl_p = _abcdk_proxy_node_alloc(ctx);

    if (!node_p)
        return -2;

    node_ctx_p = (abcdk_proxy_node_t *)abcdk_asynctcp_get_userdata(node_p);

    /*绑定安全模式。*/
    node_ctx_p->ssl_scheme = ssl_scheme;

    if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_OPENSSL)
    {
#ifdef HEADER_SSL_H
        node_ctx_p->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(1, (node_ctx_p->father->openssl_check_cert ? node_ctx_p->father->openssl_ca_file : NULL),
                                                                   (node_ctx_p->father->openssl_check_cert ? node_ctx_p->father->openssl_ca_path : NULL),
                                                                   node_ctx_p->father->openssl_cert_file, node_ctx_p->father->openssl_key_file, NULL);
#endif // HEADER_SSL_H
        if (!node_ctx_p->openssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载证书或私钥失败，无法创建SSL安全环境。");
            return -2;
        }
        
    }
    else if (ssl_scheme == ABCDK_PROXY_SSL_SCHEME_EASYSSL)
    {
        /*仅用于验证密钥是否可以加载。*/
        node_ctx_p->easyssl_ctx = abcdk_easyssl_create_from_file(node_ctx_p->father->easyssl_key_file, ABCDK_EASYSSL_SCHEME_ENIGMA,
                                                                 node_ctx_p->father->easyssl_salt_size);
        if (!node_ctx_p->easyssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载共享密钥失败，无法创建SSL环境。");
            return -2;
        }

        abcdk_easyssl_destroy(&node_ctx_p->easyssl_ctx);
    }

    cb.prepare_cb = _abcdk_proxy_prepare_cb;
    cb.event_cb = _abcdk_proxy_event_cb;
    cb.request_cb = _abcdk_proxy_request_cb;

    chk = abcdk_asynctcp_listen(node_p, &listen_addr, &cb);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'失败，无权限或被占用。", listen_p);
        return -3;
    }

    return 0;
}

static void _abcdk_proxy_process(abcdk_proxy_t *ctx)
{
    const char *log_path;
    int chk;

    ctx->name = abcdk_option_get(ctx->args, "--name", 0, SOLUTION_NAME);
    ctx->realm = abcdk_option_get(ctx->args, "--realm", 0, "proxy");
    ctx->auth_path = abcdk_option_get(ctx->args, "--auth-path", 0, NULL);

    ctx->listen_raw = abcdk_option_get(ctx->args, "--listen-raw", 0, NULL);
    ctx->listen_openssl = abcdk_option_get(ctx->args, "--listen-openssl", 0, NULL);
    ctx->listen_easyssl = abcdk_option_get(ctx->args, "--listen-easyssl", 0, NULL);

    ctx->openssl_ca_file = abcdk_option_get(ctx->args, "--openssl-ca-file", 0, NULL);
    ctx->openssl_ca_path = abcdk_option_get(ctx->args, "--openssl-ca-path", 0, NULL);
    ctx->openssl_cert_file = abcdk_option_get(ctx->args, "--openssl-cert-file", 0, NULL);
    ctx->openssl_key_file = abcdk_option_get(ctx->args, "--openssl-key-file", 0, NULL);
    ctx->openssl_check_cert = abcdk_option_get_int(ctx->args, "--openssl-check-cert", 0, 1);

    ctx->easyssl_key_file = abcdk_option_get(ctx->args, "--easyssl-key-file", 0, NULL);
    ctx->easyssl_salt_size = abcdk_option_get_int(ctx->args, "--easyssl-salt-size", 0, 123);

    /*修复不支持的配置。*/
    ctx->easyssl_key_file = (ctx->easyssl_key_file?ctx->easyssl_key_file:"");
    ctx->easyssl_salt_size = ABCDK_CLAMP(ctx->easyssl_salt_size, 0, 256);
    
    ctx->uplink = abcdk_option_get(ctx->args, "--uplink", 0, NULL);
    
    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    ctx->logger = abcdk_logger_open2(log_path, "proxy.log", "proxy.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, ctx->logger);

    abcdk_trace_output(LOG_INFO, "启动……");

    ctx->io_ctx = abcdk_asynctcp_start(1000, -1);
    if (!ctx->io_ctx)
        goto END;

    chk = _abcdk_proxy_start_listen(ctx, ABCDK_PROXY_SSL_SCHEME_RAW);
    if (chk != 0)
        goto END;

#ifdef HEADER_SSL_H
    chk = _abcdk_proxy_start_listen(ctx, ABCDK_PROXY_SSL_SCHEME_OPENSSL);
    if (chk != 0)
        goto END;
#endif // HEADER_SSL_H

    chk = _abcdk_proxy_start_listen(ctx, ABCDK_PROXY_SSL_SCHEME_EASYSSL);
    if (chk != 0)
        goto END;

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

END:

    abcdk_asynctcp_stop(&ctx->io_ctx);
    abcdk_asynctcp_unref(&ctx->listen_raw_p);
    abcdk_asynctcp_unref(&ctx->listen_openssl_p);
    abcdk_asynctcp_unref(&ctx->listen_easyssl_p);

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
