/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_ASIO_ASYNCTCP_H
#define ABCDK_ASIO_ASYNCTCP_H

#include "abcdk/util/general.h"
#include "abcdk/util/getargs.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/epollex.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/map.h"
#include "abcdk/util/time.h"
#include "abcdk/ssl/openssl.h"
#include "abcdk/ssl/easyssl.h"

__BEGIN_DECLS

/**/
#ifndef HEADER_SSL_H
typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
#define SSL_read(f,b,s) 0
#define SSL_write(f,b,s) 0
#endif //HEADER_SSL_H

/** 简单的异步TCP通讯。 */
typedef struct _abcdk_asynctcp abcdk_asynctcp_t;
/** 异步TCP节点。 */
typedef struct _abcdk_asynctcp_node abcdk_asynctcp_node_t;

/* 通知事件。*/
typedef enum _abcdk_asynctcp_event
{
    /**
     * 新连接。
     * 
     * @return 0 允许连接，-1 禁止连接。
    */
    ABCDK_ASYNCTCP_EVENT_ACCEPT = 1,
#define ABCDK_ASYNCTCP_EVENT_ACCEPT ABCDK_ASYNCTCP_EVENT_ACCEPT

    /**
     * 已连接。
     * 
     * @return 忽略。
    */
    ABCDK_ASYNCTCP_EVENT_CONNECT = 2,
#define ABCDK_ASYNCTCP_EVENT_CONNECT ABCDK_ASYNCTCP_EVENT_CONNECT

    /**
     * 有数据到达。
     * 
     * @return 忽略。
    */
    ABCDK_ASYNCTCP_EVENT_INPUT = 3,
#define ABCDK_ASYNCTCP_EVENT_INPUT ABCDK_ASYNCTCP_EVENT_INPUT

    /**
     * 链路空闲，可以发送。
     * 
     * @return 忽略。
    */
    ABCDK_ASYNCTCP_EVENT_OUTPUT = 4,
#define ABCDK_ASYNCTCP_EVENT_OUTPUT ABCDK_ASYNCTCP_EVENT_OUTPUT

    /**
     * 已断开。
     * 
     * @return 忽略。
    */
    ABCDK_ASYNCTCP_EVENT_CLOSE = 5,
#define ABCDK_ASYNCTCP_EVENT_CLOSE ABCDK_ASYNCTCP_EVENT_CLOSE

    /**
     * 中断(资源不足，或禁止连接)。
     * 
     * @return 忽略。
    */
    ABCDK_ASYNCTCP_EVENT_INTERRUPT = 6
#define ABCDK_ASYNCTCP_EVENT_INTERRUPT ABCDK_ASYNCTCP_EVENT_INTERRUPT

}abcdk_asynctcp_event_t;

/** 
 * 回调函数。
*/
typedef struct _abcdk_asynctcp_callback
{
    /**
     * 为新连接做准备工作的通知回调函数。
     * 
     * @note 监听有效，必须指定。
     * 
     * @param [out] node 新的节点，返回时填写。
     */
    void (*prepare_cb)(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen);

    /**
     * 事件通知回调函数。
     *
     * @note 除ABCDK_ASYNCTCP_EVENT_ACCEPT事件外，其余事件均忽略返回值。
     */
    void (*event_cb)(abcdk_asynctcp_node_t *node, uint32_t event, int *result);

    /**
     * 请求数据到达通知回调函数。
     *
     * @note 如果未指定，则通知ABCDK_ASYNCTCP_EVENT_INPUT事件，否则将被拦截。
     *
     * @param [out] remain 剩余的数据长度，返回时填写。
     */
    void (*request_cb)(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain);

} abcdk_asynctcp_callback_t;

/**
 * 释放。
 * 
 * @note 当引用计数为0时，对像将被删除。
*/
void abcdk_asynctcp_unref(abcdk_asynctcp_node_t **node);

/**
 * 引用。
*/
abcdk_asynctcp_node_t *abcdk_asynctcp_refer(abcdk_asynctcp_node_t *src);

/**
 * 申请节点。
 *
 * @param [in] userdata 用户数据长度。
 * @param [in] free_cb 用户数据销毁函数。
 *
 * @return !NULL(0) 成功(指针)，NULL(0) 失败。
 */
abcdk_asynctcp_node_t *abcdk_asynctcp_alloc(abcdk_asynctcp_t *ctx, size_t userdata, void (*free_cb)(void *userdata));

/**
 * 升级为openssl环境。
 *
 * @param [in] ssl_ctx SSL环境指针(仅复制，创建者放负责回收和释放)。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_asynctcp_upgrade2openssl(abcdk_asynctcp_node_t *node,SSL_CTX *ssl_ctx);

/**
 * 升级为easyssl环境。
 *
 * @param [in] ssl_ctx SSL环境指针(仅复制，创建者放负责回收和释放)。
 *
 * @return 0 成功，!0 失败。
 */
int abcdk_asynctcp_upgrade2easyssl(abcdk_asynctcp_node_t *node,abcdk_easyssl_t *ssl_ctx);

/**
 * openssl环境指针。
 * 
 * @note 连接建立后有效，且调用者不能释放。
*/
SSL *abcdk_asynctcp_openssl_ctx(abcdk_asynctcp_node_t *node);


/**
 * 用户环境指针。
 * 
 * @return 旧的指针(0号索引)。
*/
void *abcdk_asynctcp_get_userdata(abcdk_asynctcp_node_t *node);

/**
 * 设置用户环境指针。
 * 
 * @return 旧的指针(0号索引)。
*/
void *abcdk_asynctcp_set_userdata(abcdk_asynctcp_node_t *node,void *opaque);

/**
 * 设置超时。
 * 
 * @note 1、看门狗精度为1000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asynctcp_set_timeout(abcdk_asynctcp_node_t *node, time_t timeout);

/**
 * 获取地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asynctcp_get_sockaddr(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote);

/**
 * 获取地址(转换成字符串)。
 * 
 * @note unix/IPv4/IPv6有效。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asynctcp_get_sockaddr_str(abcdk_asynctcp_node_t *node, char local[NAME_MAX],char remote[NAME_MAX]);

/**
 * 读。
 * 
 * @return > 0 已读取数据的长度，0 无数据。
*/
ssize_t abcdk_asynctcp_recv(abcdk_asynctcp_node_t *node, void *buf, size_t size);

/**
 * 监听输入事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asynctcp_recv_watch(abcdk_asynctcp_node_t *node);

/**
 * 写。
 * 
 * @warning 在SSL环境中，重发数据的参数不能改变(指针和长度)。
 * 
 * @return > 0 已写入数据的长度，0 链路忙。
*/
ssize_t abcdk_asynctcp_send(abcdk_asynctcp_node_t *node, void *buf, size_t size);

/**
 * 监听输出事件。
 * 
 * @warning 当事件未被触发时，多次监听事件将会合并。
 * @warning 当事件被触发后，监听事件自动取消，在下一次监听前不会连续触发事件。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_asynctcp_send_watch(abcdk_asynctcp_node_t *node);

/**
 * 停止通讯引擎。
 * 
 * @note 非线程安全。
 * 
 * @param [in out] ctx 环境指针。
*/
void abcdk_asynctcp_stop(abcdk_asynctcp_t **ctx);

/**
 * 启动通讯引擎。
 * 
 * @param [in] max 最大连接数量。<= 0 使用文件句柄数量的一半作为最大连接数量。
 * @param [in] cpu 绑定的CPU编号。从0开始，-1 不绑定。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_asynctcp_t *abcdk_asynctcp_start(int max,int cpu);

/**
 * 监听客户端连接。
 * 
 * @param [in] node 通讯对象指针。
 * @param [in] addr 监听地址指针。
 * @param [in] cb 回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_asynctcp_listen(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *addr,abcdk_asynctcp_callback_t *cb);

/**
 * 连接远程服务器。
 * 
 * @note 仅发出连接指令，连接是否成功以消息通知。
 * 
 * @param [in] node 通讯对象指针。
 * @param [in] addr 服务端地址指针。
 * @param [in] cb 回调函数指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_asynctcp_connect(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *addr,abcdk_asynctcp_callback_t *cb);

/**
 * 投递数据。
 * 
 * @note 内存对象将被托管，应用层不可以继续访问内存对象。
 * 
 * @param [in] data 内存对象指针，索引0号元素有效。注：仅做指针复制，不会改变对象的引用计数。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asynctcp_post(abcdk_asynctcp_node_t *node, abcdk_object_t *data);

/**
 * 投递数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
 */
int abcdk_asynctcp_post_buffer(abcdk_asynctcp_node_t *node, const void *data,size_t size);

/** 
 * 投递数据。
 * 
 * @param [in] max 格式化数据最大长度。 
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asynctcp_post_vformat(abcdk_asynctcp_node_t *node, int max, const char *fmt, va_list ap);

/** 
 * 投递数据。
 * 
 * @return 0 成功，-1 失败，-2 失败(监听对象不支持投递数据)。
*/
int abcdk_asynctcp_post_format(abcdk_asynctcp_node_t *node, int max, const char *fmt, ...);

__END_DECLS

#endif //ABCDK_ASIO_ASYNCTCP_H