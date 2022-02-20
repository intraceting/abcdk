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
    const char *listen_str;

    /** 0：非领导者；!0：领导者。*/
    volatile int isleader;

} abcdk_vmtx_t;

/**/
typedef struct _abcdk_vmtx_node
{
    /** 大环境。*/
    abcdk_vmtx_t *vmtx;

    /** 上行线路。*/
    abcdk_comm_easy_t *uplink;

    /** 下行线路。*/
    abcdk_comm_easy_t *downlink;
    
    /** 主机ID。*/
    char uuid[37];

    /** 主机名字。*/
    char name[256];

    /** 主机地址。*/
    char addr[256];

}abcdk_vmtx_node_t;


#endif //ABCDK_VMTX_CONTEXT_H