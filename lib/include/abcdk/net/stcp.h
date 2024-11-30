/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_NET_STCP_H
#define ABCDK_NET_STCP_H

#include "abcdk/util/general.h"
#include "abcdk/util/getargs.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/asioex.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/map.h"
#include "abcdk/util/time.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/util/worker.h"
#include "abcdk/util/wred.h"
#include "abcdk/openssl/bio.h"

__BEGIN_DECLS

/**/
#ifndef HEADER_SSL_H
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct bio_st BIO;
#define SSL_read(f,b,s) 0
#define SSL_write(f,b,s) 0
#define BIO_read(f,b,s) 0
#define BIO_write(f,b,s) 0
#endif //HEADER_SSL_H

/**简单的TCP环境。 */
typedef struct _abcdk_stcp abcdk_stcp_t;

/**TCP节点。 */
typedef struct _abcdk_stcp_node abcdk_stcp_node_t;

/**安全方案。*/
typedef enum _abcdk_stcp_ssl_scheme
{
    /**RAW.*/
    ABCDK_STCP_SSL_SCHEME_RAW = 0,
#define ABCDK_STCP_SSL_SCHEME_RAW   ABCDK_STCP_SSL_SCHEME_RAW

    /**PKI(Public Key Infrastructure).*/
    ABCDK_STCP_SSL_SCHEME_PKI = 1,
#define ABCDK_STCP_SSL_SCHEME_PKI   ABCDK_STCP_SSL_SCHEME_PKI

    /**SKE(Shared key encryption).*/
    ABCDK_STCP_SSL_SCHEME_SKE = 2,
#define ABCDK_STCP_SSL_SCHEME_SKE   ABCDK_STCP_SSL_SCHEME_SKE

    /**PKI is based on SKE.*/
    ABCDK_STCP_SSL_SCHEME_PKIS = 3
#define ABCDK_STCP_SSL_SCHEME_PKIS   ABCDK_STCP_SSL_SCHEME_PKIS
}abcdk_stcp_ssl_scheme_t;

/**通知事件。*/
typedef enum _abcdk_stcp_event
{
    /**
     * 新连接。
     * 
     * @return 0 允许连接，-1 禁止连接。
    */
    ABCDK_STCP_EVENT_ACCEPT = 1,
#define ABCDK_STCP_EVENT_ACCEPT ABCDK_STCP_EVENT_ACCEPT

    /**
     * 已连接。
     * 
     * @return 忽略。
    */
    ABCDK_STCP_EVENT_CONNECT = 2,
#define ABCDK_STCP_EVENT_CONNECT ABCDK_STCP_EVENT_CONNECT

    /**
     * 有数据到达。
     * 
     * @return 忽略。
    */
    ABCDK_STCP_EVENT_INPUT = 3,
#define ABCDK_STCP_EVENT_INPUT ABCDK_STCP_EVENT_INPUT

    /**
     * 链路空闲，可以发送。
     * 
     * @return 忽略。
    */
    ABCDK_STCP_EVENT_OUTPUT = 4,
#define ABCDK_STCP_EVENT_OUTPUT ABCDK_STCP_EVENT_OUTPUT

    /**
     * 已断开。
     * 
     * @warning 此事件不需要释放连接。
     * 
     * @return 忽略。
    */
    ABCDK_STCP_EVENT_CLOSE = 5,
#define ABCDK_STCP_EVENT_CLOSE ABCDK_STCP_EVENT_CLOSE

    /**
     * 中断(资源不足，或禁止连接)。
     * 
     * @warning 此事件不需要释放连接。
     * 
     * @return 忽略。
    */
    ABCDK_STCP_EVENT_INTERRUPT = 6
#define ABCDK_STCP_EVENT_INTERRUPT ABCDK_STCP_EVENT_INTERRUPT

}abcdk_stcp_event_t;

/** 
 * 配置。
*/
typedef struct _abcdk_stcp_config
{
    /**安全方案*/
    int ssl_scheme;

    /**CA证书。*/
    const char *pki_ca_file;

    /**CA路径。*/
    const char *pki_ca_path;

    /**证书。*/
    const char *pki_cert_file;

    /**私钥。*/
    const char *pki_key_file;
    
    /**密钥。*/
    const char *pki_key_passwd;

    /**是否验证对端证书。0 否，!0 是。*/
    int pki_check_cert;
    
    /** 
     * 下层协议。
     * 
     * 例1："\x08http/1.1"
     * 例2："\x02h2\x08http/1.1"
     *
    */
    const uint8_t *pki_next_proto;

    /**算法列表。*/
    const char *pki_cipher_list;

    /**共享密钥。*/
    const char *ske_key_file;

    /**绑定地址。*/
    abcdk_sockaddr_t bind_addr;

    /**
     * 绑定设备。
     * 
     * @note 需要root权限支持，否则忽略。
    */
    const char *bind_ifname;

    /**
     * 输出队列丢包最小阈值。 
     * 
     * @note 有效范围：200~600，默认：200
    */
    int out_hook_min_th;

    /**
     * 输出队列丢包最大阈值。 
     * 
     * @note 有效范围：400~800，默认：400
    */
    int out_hook_max_th;

    /**
     * 输出队列丢包权重因子。
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int out_hook_weight;

    /**
     * 输出队列丢包概率因子。 
     * 
     * @note 有效范围：1~99，默认：2 
    */
    int out_hook_prob;

    /**
     * 为新连接做准备工作的通知回调函数。
     * 
     * @note 仅监听有效，必须为有效地址。
     * 
     * @param [out] node 新的节点，返回时填写。
     */
    void (*prepare_cb)(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen);

    /**
     * 事件通知回调函数。
     *
     * @note 除ABCDK_STCP_EVENT_ACCEPT事件外，其余事件均忽略返回值。
     */
    void (*event_cb)(abcdk_stcp_node_t *node, uint32_t event, int *result);

    /**
     * 输入数据到达通知回调函数。
     *
     * @note 如果未指定，则通知ABCDK_STCP_EVENT_INPUT事件，否则将被拦截。
     *
     * @param [out] remain 剩余的数据长度，返回时填写。
     */
    void (*input_cb)(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain);

} abcdk_stcp_config_t;

/**
 * 释放。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_stcp_unref(abcdk_stcp_node_t **node);

/**
 * 引用。
*/
abcdk_stcp_node_t *abcdk_stcp_refer(abcdk_stcp_node_t *src);

/**
 * 申请。
 *
 * @param [in] userdata 用户数据长度。
 * @param [in] free_cb 用户数据销毁函数。
 *
 * @return !NULL(0) 成功(指针)，NULL(0) 失败。
 */
abcdk_stcp_node_t *abcdk_stcp_alloc(abcdk_stcp_t *ctx, size_t userdata, void (*free_cb)(void *userdata));

/**
 * 获取索引。
 * 
 * @note 进程内唯一。
 * 
*/
uint64_t abcdk_stcp_get_index(abcdk_stcp_node_t *node);

/**
 * 获取SSL链路句柄。
 * 
 * @warning 应用层不能释放链路句柄。
*/
SSL *abcdk_stcp_ssl_get_handle(abcdk_stcp_node_t *node);

/**
 * 获取SSL应用层协议名称。
 */
char *abcdk_stcp_ssl_get_alpn_selected(abcdk_stcp_node_t *node, char proto[255+1]);

/**
 * 获取用户环境指针。
*/
void *abcdk_stcp_get_userdata(abcdk_stcp_node_t *node);

/**
 * 设置超时。
 * 
 * @param timeout 时长(秒)。!0 有效， 0 禁用。默认：0。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_stcp_set_timeout(abcdk_stcp_node_t *node, time_t timeout);

/**
 * 获取地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_stcp_get_sockaddr(abcdk_stcp_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote);

/**
 * 获取地址(转换成字符串)。
 * 
 * @note unix/IPv4/IPv6有效。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_stcp_get_sockaddr_str(abcdk_stcp_node_t *node, char local[NAME_MAX],char remote[NAME_MAX]);

/**
 * 读。
 * 
 * @return > 0 已读取数据的长度，0 无数据。
*/
ssize_t abcdk_stcp_recv(abcdk_stcp_node_t *node, void *buf, size_t size);

/**
 * 监听输入事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_stcp_recv_watch(abcdk_stcp_node_t *node);

/**
 * 写。
 * 
 * @warning 在SSL环境中，重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已写入数据的长度，0 链路忙。
*/
ssize_t abcdk_stcp_send(abcdk_stcp_node_t *node, void *buf, size_t size);

/**
 * 监听输出事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_stcp_send_watch(abcdk_stcp_node_t *node);

/** 销毁。*/
void abcdk_stcp_destroy(abcdk_stcp_t **ctx);

/**
 * 创建。
 * 
 * @param [in] worker 工人(线程)数量。
*/
abcdk_stcp_t *abcdk_stcp_create(int worker);

/** 停止。*/
void abcdk_stcp_stop(abcdk_stcp_t *ctx);

/**
 * 启动监听。
 * 
 * @note 在对象关闭前，配置信息必须保持有效且不能更改。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_stcp_listen(abcdk_stcp_node_t *node, abcdk_stcp_config_t *cfg);

/**
 * 连接远程。
 * 
 * @note 在对象关闭前，配置信息必须保持有效且不能更改。
 * @note 仅发出连接指令，连接是否成功以消息通知。
 * 
 * @param [in] addr 远程地址。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_stcp_connect(abcdk_stcp_node_t *node, abcdk_sockaddr_t *addr,abcdk_stcp_config_t *cfg);

/**
 * 投递数据。
 * 
 * @note 投递的数据对象将被托管，应用层不可以继续访问数据对象。
 * @warning 当网络不理想时，通讯可能会有延时，非关键数据可能会被丢弃。
 * 
 * @param [in] data 数据对象，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * @param [in] key 关键数据。 !0 是，0 否。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_stcp_post(abcdk_stcp_node_t *node, abcdk_object_t *data, int key);

/**
 * 投递数据。
 * 
 * @note 仅支持关键数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
 */
int abcdk_stcp_post_buffer(abcdk_stcp_node_t *node, const void *data,size_t size);

/** 
 * 投递数据。
 * 
 * @note 仅支持关键数据。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_stcp_post_vformat(abcdk_stcp_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 投递数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_stcp_post_format(abcdk_stcp_node_t *node, int max, const char *fmt, ...);

__END_DECLS

#endif //ABCDK_NET_STCP_H