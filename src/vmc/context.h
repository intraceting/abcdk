/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_VMC_CONTEXT_H
#define ABCDK_VMC_CONTEXT_H

#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"
#include "comm/easy.h"
#include "shell/proc.h"
#include "util/log.h"
#include "util/signal.h"


enum _abcdk_stbs_vmc_constant
{
    /** 备机。*/
    ABCDK_VMC_ROLE_STANDBY = 1,
#define ABCDK_VMC_ROLE_STANDBY ABCDK_VMC_ROLE_STANDBY

    /** 主机。*/
    ABCDK_VMC_ROLE_MASTER = 2,
#define ABCDK_VMC_ROLE_MASTER ABCDK_VMC_ROLE_MASTER

    /** 从机。*/
    ABCDK_VMC_ROLE_SLAVE = 3,
#define ABCDK_VMC_ROLE_SLAVE ABCDK_VMC_ROLE_SLAVE

    
    ABCDK_VMC_STATUS_REGISTER = 1
#define ABCDK_VMC_STATUS_REGISTER
};

typedef struct _abcdk_vmc_node
{
    /** 节点状态。*/
    volatile int status;

    /** 节点ID。*/
    uint16_t id;

    /** 节点名字。*/
    char name[256];

    /** 节点地址。*/
    char addr[256];

} abcdk_vmc_node_t;

/**主机(master)信息。*/
typedef struct _abcdk_vmc_master
{
    /** 地址。*/
    abcdk_sockaddr_t addr;

    /** 链路。*/
    abcdk_comm_easy_t *easy;

    /** 角色。*/
    volatile int role;

}abcdk_vmc_master_t;

/**服务环境。*/
typedef struct _abcdk_vmc
{
    int errcode;
    abcdk_tree_t *args;

    int lock_pid;
    int lock_fd;
    const char *lock_file;

    /** ID。*/
    uint16_t id;
        
    /** 角色。*/
    volatile int role;

    /** */
    abcdk_vmc_master_t masters[2];
    
} abcdk_vmc_t;


#endif // ABCDK_VMC_CONTEXT_H