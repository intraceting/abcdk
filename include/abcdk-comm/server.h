/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_SERVER_H
#define ABCDK_COMM_SERVER_H

#include "abcdk-comm/comm.h"
#include "abcdk-comm/message.h"

__BEGIN_DECLS


/*消息回调函数。*/
typedef void (*abcdk_comm_svr_message_cb)(abcdk_comm_node_t *comm,const abcdk_comm_msg_t *req, abcdk_comm_msg_t **rsp,void *opaque);

/**
 * 启动监听。
 * 
 * @return >= 0 成功(session)，< 0 失败。
*/
int abcdk_comm_svr_listen(SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr,abcdk_comm_svr_message_cb message_cb,void *opaque);



__END_DECLS

#endif //ABCDK_COMM_SERVER_H