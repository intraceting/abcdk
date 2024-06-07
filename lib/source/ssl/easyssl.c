/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ssl/easyssl.h"

/** 简单的SSL通讯。 */
struct _abcdk_easyssl
{
    /**发送加密环境。*/
    abcdk_enigma_t *en_send_ctx;

    /**接收加密环境。*/
    abcdk_enigma_t *en_recv_ctx;

    /**发送缓存区。*/
    abcdk_object_t *send_buf;


    /**句柄。*/
    int fd;
    
};//abcdk_easyssl_t;