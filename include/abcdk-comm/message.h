/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_MESSAGE_H
#define ABCDK_COMM_MESSAGE_H

#include "abcdk-comm/comm.h"

__BEGIN_DECLS

/** 消息缓存对象。*/
typedef struct _abcdk_comm_msg abcdk_comm_msg_t;

/** 消息应答标志。*/
#define ABCDK_COMM_MSG_FLAG_RSP             0x01

/**
 * 释放消息缓存对象。
*/
void abcdk_comm_msg_free(abcdk_comm_msg_t **msg);

/**
 * 申请消息缓存对象。
*/
abcdk_comm_msg_t *abcdk_comm_msg_alloc(size_t size);

/**
 * 调整消息缓存大小。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_comm_msg_realloc(abcdk_comm_msg_t *msg, size_t size);

/**
 * 重置。
 * 
*/
void abcdk_comm_msg_reset(abcdk_comm_msg_t *msg);

/** 
 * 获取消息协议。
 * 
 * @return 旧的协议。
*/
uint32_t abcdk_comm_msg_protocol(abcdk_comm_msg_t *msg);

/**
 * 设置消息协议。
 * 
 * @return 旧的协议。
*/
uint32_t abcdk_comm_msg_protocol_set(abcdk_comm_msg_t *msg, uint32_t protocol);

/** 
 * 获取消息标志。
 * 
 * @return 旧的标志。
*/
uint8_t abcdk_comm_msg_flag(abcdk_comm_msg_t *msg);

/**
 * 设置消息标志。
 * 
 * @return 旧的标志。
*/
uint8_t abcdk_comm_msg_flag_set(abcdk_comm_msg_t *msg, uint8_t flag);

/** 
 * 获取消息编号。
 * 
 * @return 旧的编号。
*/
uint64_t abcdk_comm_msg_number(abcdk_comm_msg_t *msg);

/**
 * 设置消息编号。
 * 
 * @return 旧的编号。
*/
uint64_t abcdk_comm_msg_number_set(abcdk_comm_msg_t *msg, uint64_t number);

/**
 * 获取数据区指针。
*/
void *abcdk_comm_msg_data(const abcdk_comm_msg_t *msg);

/**
 * 获取数据区长度。
*/
size_t abcdk_comm_msg_size(const abcdk_comm_msg_t *msg);

/**
 * 接收消息。
 * 
 * @return 1 消息完整，0 消息不完整，-1 出错。
*/
int abcdk_comm_msg_recv(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);

/**
 * 发送消息。
 * 
 * @return 1 消息完整，0 消息不完整，-1 出错。
*/
int abcdk_comm_msg_send(abcdk_comm_node_t *node,abcdk_comm_msg_t *msg);



__END_DECLS

#endif //ABCDK_COMM_MESSAGE_H