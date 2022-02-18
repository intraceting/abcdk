/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_VMTX_CONTEXT_H
#define ABCDK_VMTX_CONTEXT_H

#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"
#include "comm/easy.h"
#include "shell/proc.h"

/**/
typedef struct _abcdk_vmtx
{
    int errcode;
    abcdk_tree_t *args;

    int lock_pid;
    int lock_fd;
    const char *lock_file;

    abcdk_sockaddr_t listen_addr;
    const char *listen;

    volatile int isleader;

} abcdk_vmtx_t;

/**/
typedef struct _abcdk_vmtx_node
{
    /** 引用计数器。*/
    volatile int refcount;

    abcdk_vmtx_t *ctx;

    struct _abcdk_vmtx_easy *uplink;
    struct _abcdk_vmtx_easy *downlink;
    
    char hostname[100];
    char hostuuid[37];
    char hostaddr[37];

}abcdk_vmtx_node_t;

/**/
typedef struct _abcdk_vmtx_easy
{
    /** 引用计数器。*/
    volatile int refcount;

    int flag;
#define ABCDK_VMTX_EASY_FLAG_LISTEN     1 //listen
#define ABCDK_VMTX_EASY_FLAG_ACCEPT     2 //accept

    abcdk_comm_easy_t *easy;

    abcdk_sockaddr_t sockname;
    abcdk_sockaddr_t peername;
    char sockname_str[100];
    char peername_str[100];

    abcdk_vmtx_node_t *node;

}abcdk_vmtx_easy_t;

/**
 * 回显测试(心跳)。
 * 
 * req: cmd(2 bytes) + flag(1 byte)
 *  flag(0x80): 我是领导者。
 *  
 * rsp: cmd(2 bytes) + errno(4 bytes)
 *  errno(0): 无错误。
 *  errno(1): 领导者身份被拒绝。
 *  errno(2~N): 未定义。
*/
#define ABCDK_VMTX_CMD_ECHO     1 

#endif //ABCDK_VMTX_CONTEXT_H