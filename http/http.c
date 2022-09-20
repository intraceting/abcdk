/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "http/http.h"

/** HTTP连接。*/
typedef struct _abcdk_http_connection
{
    /** 接收缓冲区。*/
    abcdk_comm_message_t *in_buffer;

    /** 发送队列。*/
    abcdk_comm_queue_t *out_queue;

    /** 请求头环境变量表。*/
    abcdk_map_t *req_envs;

    /** 请求头环境变量表。*/
    abcdk_map_t *rsp_envs;

} abcdk_http_connection_t;
