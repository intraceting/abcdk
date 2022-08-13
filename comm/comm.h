/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_COMM_H
#define ABCDK_COMM_COMM_H

#include "util/general.h"
#include "util/getargs.h"
#include "util/socket.h"
#include "util/epollex.h"
#include "util/openssl.h"
#include "util/thread.h"
#include "util/tree.h"
#include "util/map.h"

__BEGIN_DECLS

/**/
#ifndef HEADER_SSL_H
typedef struct ssl_ctx_st SSL_CTX;
#endif //HEADER_SSL_H

/** 通信环境。 */
typedef struct _abcdk_comm abcdk_comm_t;
/** 通信节点。 */
typedef struct _abcdk_comm_node abcdk_comm_node_t;

/* COMM事件。*/
enum _abcdk_comm_event
{
    /*已连接。*/
    ABCDK_COMM_EVENT_ACCEPT = 1,
#define ABCDK_COMM_EVENT_ACCEPT ABCDK_COMM_EVENT_ACCEPT

    /*已连接。*/
    ABCDK_COMM_EVENT_CONNECT = 2,
#define ABCDK_COMM_EVENT_CONNECT ABCDK_COMM_EVENT_CONNECT

    /*有数据到达。*/
    ABCDK_COMM_EVENT_INPUT = 3,
#define ABCDK_COMM_EVENT_INPUT ABCDK_COMM_EVENT_INPUT

    /*链路空闲，可以发送。*/
    ABCDK_COMM_EVENT_OUTPUT = 4,
#define ABCDK_COMM_EVENT_OUTPUT ABCDK_COMM_EVENT_OUTPUT

    /*已断开。*/
    ABCDK_COMM_EVENT_CLOSE = 5,
#define ABCDK_COMM_EVENT_CLOSE ABCDK_COMM_EVENT_CLOSE

    /*监听关闭。*/
    ABCDK_COMM_EVENT_LISTEN_CLOSE = 6
#define ABCDK_COMM_EVENT_LISTEN_CLOSE ABCDK_COMM_EVENT_LISTEN_CLOSE
};

/** 事件回调函数。*/
typedef void (*abcdk_comm_event_cb)(abcdk_comm_node_t *node, uint32_t event);

/**
 * 减少对象的引用计数。
 * 
 * @warning 当引用计数为0时，对像将被删除。
*/
void abcdk_comm_node_unref(abcdk_comm_node_t **node);

/**
 * 增加对象的引用计数。
*/
abcdk_comm_node_t *abcdk_comm_node_refer(abcdk_comm_node_t *src);

/**
 * 设置超时。
 * 
 * @warning 1、看门狗精度为1000毫秒；2、超时生效时间受引擎的工作周期影响。
 * 
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_set_timeout(abcdk_comm_node_t *node, time_t timeout);

/**
 * 获取地址。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_get_sockaddr(abcdk_comm_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote);

/**
 * 获取地址(转换成字符串)。
 * 
 * @note unix/IPv4/IPv6有效。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_get_sockaddr_str(abcdk_comm_node_t *node, char local[NAME_MAX],char remote[NAME_MAX]);

/**
 * 设置应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_comm_set_userdata(abcdk_comm_node_t *node, void *opaque);

/**
 * 获取应用层环境指针。
 * 
 * @return 旧的指针。
*/
void *abcdk_comm_get_userdata(abcdk_comm_node_t *node);

/**
 * 调整私有数据空间大小。
 * 
 * @return !NULL(0) 成功(私有数据指针)，NULL(0) 失败。
*/
void *abcdk_comm_private_resize(abcdk_comm_node_t *node, size_t size);

/**
 * 获取私有数据空间指针。
 * 
 * @return !NULL(0) 成功(私有数据指针)，NULL(0) 失败。
*/
void *abcdk_comm_private_data(abcdk_comm_node_t *node);

/**
 * 获取私有数据空间大小。
*/
size_t abcdk_comm_private_size(abcdk_comm_node_t *node);

/**
 * 读。
 * 
 * @note 当读权利被占用时，不会有其它线程获得读事件。
 * 
 * @return > 0 已读取数据的长度，0 无数据。
*/
ssize_t abcdk_comm_read(abcdk_comm_node_t *node, void *buf, size_t size);

/**
 * 注册读事件。
 * 
 * @note 重复调用不影使用。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_read_watch(abcdk_comm_node_t *node);

/**
 * 写。
 * 
 * @note 当写权利被占用时，不会有其它线程获得写事件。
 * 
 * @return > 0 已写入数据的长度，0 链路忙。
*/
ssize_t abcdk_comm_write(abcdk_comm_node_t *node, void *buf, size_t size);

/**
 * 监听是否可写。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_write_watch(abcdk_comm_node_t *node);

/**
 * 启动通信引擎。
 * 
 * @param [in] workers 工作线程数量，<= 0 使用CPU核心数量作为工作线程量。
 * 
 * @return !NULL(0) 成功(环境指针)，NULL(0) 失败。
*/
abcdk_comm_t *abcdk_comm_start(int workers);

/**
 * 停止通信引擎。
 * 
 * @warning 环境指针不支持多程调用。
 * 
 * @param [in out] ctx 环境指针。
*/
void abcdk_comm_stop(abcdk_comm_t **ctx);

/**
 * 监听客户端连接。
 * 
 * @param [in] ctx 通信环境指针。
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 监听地址指针。
 * @param [in] event_cb 事件回调函数指针(新的连接会复制这个指针)。
 * @param [in] opaque 监听环境指针(新的连接会复制这个指针)。
 * 
 * @return !NULL(0) 成功(节点的指针)，NULL(0) 失败。
*/
abcdk_comm_node_t *abcdk_comm_listen(abcdk_comm_t *ctx, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,abcdk_comm_event_cb event_cb,void *opaque);

/**
 * 连接远程服务器。
 * 
 * @warning 仅发出连接指令，连接是否成功以消息通知。
 * 
 * @param [in] ctx 通信环境指针。
 * @param [in] ssl_ctx SSL环境指针，NULL(0) 忽略。
 * @param [in] addr 服务端地址指针。
 * @param [in] event_cb 事件回调函数指针。
 * @param [in] opaque 客户端环境指针。
 * 
 * @return !NULL(0) 成功(节点的指针)，NULL(0) 失败。
*/
abcdk_comm_node_t *abcdk_comm_connect(abcdk_comm_t *ctx, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_event_cb event_cb, void *opaque);

__END_DECLS

#endif //ABCDK_COMM_COMM_H