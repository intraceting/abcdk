/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_CLIENT_H
#define ABCDK_COMM_CLIENT_H

#include "abcdk-comm/comm.h"
#include "abcdk-comm/message.h"

__BEGIN_DECLS


/**
 * 断开连接。
*/
int abcdk_comm_cli_disconnect(int session);

/**
 * 启动连接。
 * 
 * @return >= 0 成功(session)，< 0 失败。
*/
int abcdk_comm_cli_connect(SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr);

/**
 *  
*/
int abcdk_comm_cli_transmit(int session, const abcdk_comm_msg_t *req, abcdk_comm_msg_t **rsp, time_t timeout);


__END_DECLS

#endif //ABCDK_COMM_CLIENT_H