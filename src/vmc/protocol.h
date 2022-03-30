/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_VMC_PROTOCOL_H
#define ABCDK_VMC_PROTOCOL_H

#include "util/general.h"

/**
 * 名字(告知)。
 * 
 * 请求: cmd(2 bytes) + uuid(36) + name(255)
 *  cmd：指令。
 *  uuid: 主机ID。
 *  name: 主机名称。
 * 
 * 应答：对端收到请求指令后，以请求方式回复。 
*/
#define ABCDK_VMC_CMD_NAME     1

/**
 * 回显(心跳)。
 * 
 * 请求: cmd(2 bytes) + node_num(2 bytes) + [ uuid(36) + name(255) + ... ]
 *  node_num：已知的节点数量。
 *  uuid: 主机ID。
 *  name: 主机名称。
 * 
 * 应答：对端收到请求指令后，以请求方式回复。
*/
#define ABCDK_VMC_CMD_ECHO     2

/**
 * 投票(选主)。 
 * 
 * 请求：cmd(2 bytes) + uuid(36)
 *  uuid: 主机ID(侯选人)。
 * 
 * 应答：对端收到请求指令后，赞成侯选人则静默，反对侯选人则以请求方式回复。
*/
#define ABCDK_VMC_CMD_VOTE     3


#endif //ABCDK_VMC_PROTOCOL_H