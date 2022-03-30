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

/**/
enum _abcdk_stbs_vmc_constant
{
    /** 备机。*/
    ABCDK_VMC_STATUS_STANDBY = 1,
#define ABCDK_VMC_STATUS_STANDBY ABCDK_VMC_STATUS_STANDBY

    /** 主机。*/
    ABCDK_VMC_STATUS_MASTER = 2,
#define ABCDK_VMC_STATUS_MASTER ABCDK_VMC_STATUS_MASTER

    /** 从机。*/
    ABCDK_VMC_STATUS_SLAVE = 3
#define ABCDK_VMC_STATUS_SLAVE ABCDK_VMC_STATUS_SLAVE

    
};

/**/
typedef struct _abcdk_vmc_master
{

}abcdk_vmc_master_t;

/**/
typedef struct _abcdk_vmc_node
{
    /** 节点状态。*/
    volatile int status;

    /** 节点ID。*/
    char uuid[37];

    /** 节点名字。*/
    char name[256];

    /** 节点地址。*/
    char addr[256];

} abcdk_vmc_node_t;

/**/
typedef struct _abcdk_vmc
{
    int errcode;
    abcdk_tree_t *args;

    int lock_pid;
    int lock_fd;
    const char *lock_file;

    /** 节点ID。*/
    char uuid[37];
        
    /** 节点状态。*/
    volatile int status;

    abcdk_sockaddr_t master_addr[2];
    volatile uint64_t master_active[2];
    abcdk_comm_easy_t *master_easy[2];


} abcdk_vmc_t;


#endif // ABCDK_VMC_CONTEXT_H