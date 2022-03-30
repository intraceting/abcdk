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

/**/
enum _abcdk_stbs_vmc_constant
{
    /** 待机。*/
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
typedef struct _abcdk_vmc_address
{
    /** SOCK地址。*/
    abcdk_sockaddr_t sockaddr;

    /** */
    char straddr[NAME_MAX];

}abcdk_vmc_address_t;

/**/
typedef struct _abcdk_vmc_node
{
    /** 线路。*/
    abcdk_comm_easy_t *easy;

    /** 主机ID。*/
    char uuid[37];

    /** 主机名字。*/
    char name[256];

    /** 主机地址。*/
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

    /** 主机ID。*/
    char uuid[37];


    abcdk_sockaddr_t master_team[2];
    volatile uint64_t master_active[2];
    abcdk_comm_easy_t *master_easy[2];

    volatile int status;

} abcdk_vmc_t;


#endif // ABCDK_VMC_CONTEXT_H