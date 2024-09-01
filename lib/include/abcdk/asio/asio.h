/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_ASIO_ASIO_H
#define ABCDK_ASIO_ASIO_H

#include "abcdk/util/general.h"
#include "abcdk/util/getargs.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/epollex.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/map.h"
#include "abcdk/util/time.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/enigma/bio.h"

__BEGIN_DECLS

/**/
#ifndef HEADER_SSL_H
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct bio_st BIO;
#define SSL_read(f,b,s) 0
#define SSL_write(f,b,s) 0
#endif //HEADER_SSL_H

/**异步IO环境。 */
typedef struct _abcdk_asio abcdk_asio_t;
/**异步IO节点。 */
typedef struct _abcdk_asio_node abcdk_asio_node_t;

/**安全方案。*/
typedef enum _abcdk_asio_ssl_scheme
{
    /**原始。*/
    ABCDK_ASIO_SSL_SCHEME_RAW = 0,
#define ABCDK_ASIO_SSL_SCHEME_RAW   ABCDK_ASIO_SSL_SCHEME_RAW

    /**PKI.*/
    ABCDK_ASIO_SSL_SCHEME_PKI = 1,
#define ABCDK_ASIO_SSL_SCHEME_PKI   ABCDK_ASIO_SSL_SCHEME_PKI

    /**ENIGMA.*/
    ABCDK_ASIO_SSL_SCHEME_ENIGMA = 2,
#define ABCDK_ASIO_SSL_SCHEME_ENIGMA   ABCDK_ASIO_SSL_SCHEME_ENIGMA

    /*PKI is based on ENIGMA.*/
    ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA = 3
#define ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA   ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA
}abcdk_asio_ssl_scheme_t;

/**通知事件。*/
typedef enum _abcdk_asio_event
{
    /**
     * 新连接。
     * 
     * @return 0 允许连接，-1 禁止连接。
    */
    ABCDK_ASIO_EVENT_ACCEPT = 1,
#define ABCDK_ASIO_EVENT_ACCEPT ABCDK_ASIO_EVENT_ACCEPT

    /**
     * 已连接。
     * 
     * @return 忽略。
    */
    ABCDK_ASIO_EVENT_CONNECT = 2,
#define ABCDK_ASIO_EVENT_CONNECT ABCDK_ASIO_EVENT_CONNECT

    /**
     * 有数据到达。
     * 
     * @return 忽略。
    */
    ABCDK_ASIO_EVENT_INPUT = 3,
#define ABCDK_ASIO_EVENT_INPUT ABCDK_ASIO_EVENT_INPUT

    /**
     * 链路空闲，可以发送。
     * 
     * @return 忽略。
    */
    ABCDK_ASIO_EVENT_OUTPUT = 4,
#define ABCDK_ASIO_EVENT_OUTPUT ABCDK_ASIO_EVENT_OUTPUT

    /**
     * 已断开。
     * 
     * @warning 此事件不需要释放连接。
     * 
     * @return 忽略。
    */
    ABCDK_ASIO_EVENT_CLOSE = 5,
#define ABCDK_ASIO_EVENT_CLOSE ABCDK_ASIO_EVENT_CLOSE

    /**
     * 中断(资源不足，或禁止连接)。
     * 
     * @warning 此事件不需要释放连接。
     * 
     * @return 忽略。
    */
    ABCDK_ASIO_EVENT_INTERRUPT = 6
#define ABCDK_ASIO_EVENT_INTERRUPT ABCDK_ASIO_EVENT_INTERRUPT

}abcdk_asio_event_t;

/** 
 * 配置。
*/
typedef struct _abcdk_asio_config
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

    /**密码套件。*/
    const char *pki_cipher_list;

    /**共享密钥。*/
    const char *enigma_key_file;

    /**
     * IO-HOOK最大传输单元。
     * 
     * @note 有效范围：1~262144，默认：262144
     */
    size_t io_hook_mtu;

    /**
     * 为新连接做准备工作的通知回调函数。
     * 
     * @note 监听有效，必须指定。
     * 
     * @param [out] node 新的节点，返回时填写。
     */
    void (*prepare_cb)(abcdk_asio_node_t **node, abcdk_asio_node_t *listen);

    /**
     * 事件通知回调函数。
     *
     * @note 除ABCDK_ASIO_EVENT_ACCEPT事件外，其余事件均忽略返回值。
     */
    void (*event_cb)(abcdk_asio_node_t *node, uint32_t event, int *result);


    /**
     * 输入数据到达通知回调函数。
     *
     * @note 如果未指定，则通知ABCDK_ASIO_EVENT_INPUT事件，否则将被拦截。
     *
     * @param [out] remain 剩余的数据长度，返回时填写。
     */
    void (*input_cb)(abcdk_asio_node_t *node, const void *data, size_t size, size_t *remain);

} abcdk_asio_config_t;

/**
 * 释放。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_asio_unref(abcdk_asio_node_t **node);

/**
 * 引用。
*/
abcdk_asio_node_t *abcdk_asio_refer(abcdk_asio_node_t *src);

/**
 * 申请。
 *
 * @param [in] userdata 用户数据长度。
 * @param [in] free_cb 用户数据销毁函数。
 *
 * @return !NULL(0) 成功(指针)，NULL(0) 失败。
 */
abcdk_asio_node_t *abcdk_asio_alloc(abcdk_asio_t *ctx, size_t userdata, void (*free_cb)(void *userdata));

/** 轨迹输出。*/
void abcdk_asio_trace_output(abcdk_asio_node_t *node,int type, const char* fmt,...);

/**
 * 获取索引。
 * 
 * @note 进程内唯一。
 * 
 */
uint64_t abcdk_asio_get_index(abcdk_asio_node_t *node);

/**
 * 获取OPENSSL链路句柄。
 * 
 * @warning 应用层不能释放链路句柄。
*/
SSL *abcdk_asio_openssl_get_handle(abcdk_asio_node_t *node);

/**
 * 获取OPENSSL应用层协议名称。
 */
char *abcdk_asio_openssl_get_alpn_selected(abcdk_asio_node_t *node, char proto[255+1]);

/**
 * 获取用户环境指针。
 * 
 * @return !NULL(0) 成功(有效)，NULL(0) 失败(无效)。
*/
void *abcdk_asio_get_userdata(abcdk_asio_node_t *node);

/**
 * 设置超时。
 * 
 * @note 1、看门狗精度为1000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asio_set_timeout(abcdk_asio_node_t *node, time_t timeout);

/**
 * 获取地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asio_get_sockaddr(abcdk_asio_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote);

/**
 * 获取地址(转换成字符串)。
 * 
 * @note unix/IPv4/IPv6有效。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asio_get_sockaddr_str(abcdk_asio_node_t *node, char local[NAME_MAX],char remote[NAME_MAX]);

/**
 * 读。
 * 
 * @return > 0 已读取数据的长度，0 无数据。
*/
ssize_t abcdk_asio_recv(abcdk_asio_node_t *node, void *buf, size_t size);

/**
 * 监听输入事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asio_recv_watch(abcdk_asio_node_t *node);

/**
 * 写。
 * 
 * @warning 在SSL环境中，重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已写入数据的长度，0 链路忙。
*/
ssize_t abcdk_asio_send(abcdk_asio_node_t *node, void *buf, size_t size);

/**
 * 监听输出事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asio_send_watch(abcdk_asio_node_t *node);

/**
 * 停止通讯引擎。
 * 
 * @note 非线程安全。
 * 
 * @param [in out] ctx 环境指针。
*/
void abcdk_asio_stop(abcdk_asio_t **ctx);

/**
 * 启动通讯引擎。
 * 
 * @param [in] max 最大连接数量。<= 0 使用文件句柄数量的一半作为最大连接数量。
 * @param [in] cpu 绑定的CPU编号。从0开始，-1 不绑定。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_asio_t *abcdk_asio_start(int max,int cpu);

/**
 * 启动监听。
 * 
 * @note 在对象关闭前，配置信息必须保持有效且不能更改。
 * 
 * @param [in] node 通讯对象。
 * @param [in] addr 监听地址。
 * @param [in] cfg 配置。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_asio_listen(abcdk_asio_node_t *node, abcdk_sockaddr_t *addr,abcdk_asio_config_t *cfg);

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
int abcdk_asio_connect(abcdk_asio_node_t *node, abcdk_sockaddr_t *addr,abcdk_asio_config_t *cfg);

/**
 * 投递数据。
 * 
 * @note 投递的数据对象将被托管，应用层不可以继续访问数据对象。
 * 
 * @param [in] data 数据对象，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asio_post(abcdk_asio_node_t *node, abcdk_object_t *data);

/**
 * 投递数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
 */
int abcdk_asio_post_buffer(abcdk_asio_node_t *node, const void *data,size_t size);

/** 
 * 投递数据。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asio_post_vformat(abcdk_asio_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 投递数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asio_post_format(abcdk_asio_node_t *node, int max, const char *fmt, ...);

__END_DECLS

#endif //ABCDK_ASIO_ASIO_H