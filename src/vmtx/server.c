/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"
#include "comm/easy.h"
#include "shell/proc.h"
#include "util/log.h"
#include "util/signal.h"
#include "entry.h"
#include "protocol.h"

enum _abcdk_vmtxsvr_constant
{
    /** 备机。*/
    ABCDK_VMTXSVR_ROLE_STANDBY = 1,
#define ABCDK_VMTXSVR_ROLE_STANDBY ABCDK_VMTXSVR_ROLE_STANDBY

    /** 主机。*/
    ABCDK_VMTXSVR_ROLE_MASTER = 2,
#define ABCDK_VMTXSVR_ROLE_MASTER ABCDK_VMTXSVR_ROLE_MASTER

    /** 从机。*/
    ABCDK_VMTXSVR_ROLE_SLAVE = 3,
#define ABCDK_VMTXSVR_ROLE_SLAVE ABCDK_VMTXSVR_ROLE_SLAVE
};

/** 环境。*/
typedef struct _abcdk_vmtxsvr
{
    int errcode;
    abcdk_tree_t *args;

    /*退出标记。*/
    volatile int exit_flag;

    /*运行锁。*/
    int lock_pid;
    int lock_fd;
    const char *lock_file;

    /* 客户端地址(Unix Socket)。*/
    const char *client_listen;
    abcdk_sockaddr_t client_sockaddr;

    /* Master地址(主、备)。*/
    const char *master_addr[2];
    abcdk_sockaddr_t master_sockaddr[2];

    /* 角色。*/
    volatile int role;

    /* 主节点链路。*/
    volatile abcdk_comm_easy_t *master_easy;

} abcdk_vmtxsvr_t;

void _abcdk_vmtxsvr_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n\t--help\n");

    fprintf(stderr, "\n\t--client-listen < FILE >\n");
    fprintf(stderr, "\t\t客户端命令监听地址（unixsock）。默认：/tmp/abcdk-vmtx.sock\n");

    fprintf(stderr, "\n\t--master-address < ADDRESS [ ADDRESS ] >\n");
    fprintf(stderr, "\t\t主管理节点地址（IPv4,IPv6）。\n");
    fprintf(stderr, "\n\t\tIPv4：Address:Port\n");
    fprintf(stderr, "\t\tIPv6：[Address]:Port\n");
}

int _abcdk_vmtxsvr_signal_cb(const siginfo_t *info, void *opaque)
{
    abcdk_vmtxsvr_t *ctx;
    int chk;

    ctx = (abcdk_vmtxsvr_t *)opaque;

    switch (info->si_code)
    {
    case SI_USER:
        abcdk_log_printf(LOG_INFO, "signo(%d);errno(%d);code(%d);pid(%d);uid=(%d)",
                         info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);
        break;
    default:
        abcdk_log_printf(LOG_INFO, "signo(%d);errno(%d);code(%d)", info->si_signo, info->si_errno, info->si_code);
        break;
    }

    switch (info->si_signo)
    {
    case SIGINT:
    case SIGTERM:
    case SIGTSTP:
        chk = -1;
        break;
    default:
        chk = 1;
        break;
    }

    if(chk == -1)
        abcdk_atomic_store(&ctx->exit_flag,1);

    return chk;
}

void _abcdk_vmtxsvr_register_signal(abcdk_vmtxsvr_t *ctx)
{
    abcdk_signal_t sig;

    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = _abcdk_vmtxsvr_signal_cb;
    sig.opaque = ctx;

    abcdk_sigwaitinfo_async(&sig);
}

void _abcdk_vmtxsvr_elect_leader_req(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdk_vmtxsvr_t *ctx;
    char localaddr[64] = {0},remoteaddr[64] = {0};
    uint8_t cmd;
    uint8_t vote;
    uint8_t rsp[7];
    
    ctx = (abcdk_vmtxsvr_t *)abcdk_comm_easy_get_userdata(easy);
    
    abcdk_comm_easy_get_sockaddr_str(easy,localaddr,remoteaddr);

    cmd = abcdk_endian_b_to_h16(ABCDK_PTR2U16(req,0));
    vote = ABCDK_PTR2I8(req,2);
    if(vote == 1) //选对方。
    {
        if (ctx->role == ABCDK_VMTXSVR_ROLE_MASTER)
        {
            ABCDK_PTR2U32(rsp, 2) = 0;
            ABCDK_PTR2U8(rsp, 6) = 2;//不同意。
        }
        else
        {
            if (abcdk_strcmp(localaddr, remoteaddr, 1) > 0)
            {
                ABCDK_PTR2U32(rsp, 2) = 0;
                ABCDK_PTR2U8(rsp, 6) = 1;//同意。
            }
            else
            {
                ABCDK_PTR2U32(rsp, 2) = 0;
                ABCDK_PTR2U8(rsp, 6) = 2;//不同意。
            }
        }
    }
    else if(vote == 2) //选自己。
    {
        if (ctx->role == ABCDK_VMTXSVR_ROLE_MASTER)
        {
            ABCDK_PTR2U32(rsp, 2) = 0;
            ABCDK_PTR2U8(rsp, 6) = 1;//同意。
        }
        else
        {
            if (abcdk_strcmp(localaddr, remoteaddr, 1) > 0)
            {
                ABCDK_PTR2U32(rsp, 2) = 0;
                ABCDK_PTR2U8(rsp, 6) = 2;//不同意。
            }
            else
            {
                ABCDK_PTR2U32(rsp, 2) = 0;
                ABCDK_PTR2U8(rsp, 6) = 1;//同意。
            }
        }
    }
    else
    {
        ABCDK_PTR2U32(rsp, 2) = abcdk_endian_h_to_b32(EINTR);//不支持。
    }

    abcdk_comm_easy_response(easy,rsp,7);
}

static struct _abcdk_vmtxsvr_request_entry
{
    uint16_t cmd;
    void (*func_cb)(abcdk_comm_easy_t *easy, const void *req, size_t len);
} abcdk_vmtxsvr_request_entry[] = {
    {ABCDK_VMTX_COMMAND_ELECT_LEADER, _abcdk_vmtxsvr_elect_leader_req}
};

struct _abcdk_vmtxsvr_request_entry *
_abcdk_vmtxsvr_find_entry(uint16_t cmd)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_vmtxsvr_request_entry); i++)
    {
        if (abcdk_vmtxsvr_request_entry[i].cmd != cmd)
            continue;

        return &abcdk_vmtxsvr_request_entry[i];
    }

    return NULL;
}

void _abcdk_vmtxsvr_server_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdk_vmtxsvr_t *ctx;
    struct _abcdk_vmtxsvr_request_entry * entry_cb = NULL;
    uint16_t cmd;

    ctx = (abcdk_vmtxsvr_t *)abcdk_comm_easy_get_userdata(easy);

    if (req != NULL && len > 0)
    {
        cmd = abcdk_endian_b_to_h16(ABCDK_PTR2U16(req, 0));
        entry_cb = _abcdk_vmtxsvr_find_entry(cmd);
        if (entry_cb)
            entry_cb->func_cb(easy, req, len);
    }
    else
    {

    }
}

void _abcdk_vmtxsvr_client_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdk_vmtxsvr_t *ctx;
    struct _abcdk_vmtxsvr_request_entry * entry_cb = NULL;
    uint16_t cmd;

    ctx = (abcdk_vmtxsvr_t *)abcdk_comm_easy_get_userdata(easy);

    if (req != NULL && len > 0)
    {
        cmd = abcdk_endian_b_to_h16(ABCDK_PTR2U16(req, 0));
        entry_cb = _abcdk_vmtxsvr_find_entry(cmd);
        if (entry_cb)
            entry_cb->func_cb(easy, req, len);
    }
    else
    {

    }
}

void _abcdk_vmtxsvr_server_client_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
}


int _abcdk_vmtxsvr_start(abcdk_vmtxsvr_t *ctx)
{
    abcdk_comm_easy_t *easy_tmp;
    int chk;

    /*删除可能存在的unixsock文件。*/
    unlink(ctx->client_listen);

    /*监听客户端命令地址。*/
    ctx->client_sockaddr.family = AF_UNIX;
    strncpy(ctx->client_sockaddr.addr_un.sun_path, ctx->client_listen, 108);
    easy_tmp = abcdk_comm_easy_listen(NULL, &ctx->client_sockaddr, _abcdk_vmtxsvr_client_request_cb, ctx);
    if (!easy_tmp)
    {
        abcdk_log_printf(LOG_ERR, "监听'%s'失败。", ctx->client_listen);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);
    }
    else
    {
        /*允许所有用户连接。*/
        chmod(ctx->client_listen,0777);
        /*减少引用计数。*/
        abcdk_comm_easy_unref(&easy_tmp);
    }

    /*监听服务端命令地址。*/
    for (int i = 0; i < 2; i++)
    {
        /*跳过不在本机的IP。*/
        if (!abcdk_sockaddr_where(&ctx->master_sockaddr[i], 1))
            continue;

        easy_tmp = abcdk_comm_easy_listen(NULL, &ctx->master_sockaddr[i], _abcdk_vmtxsvr_server_request_cb, ctx);
        if (!easy_tmp)
        {
            abcdk_log_printf(LOG_ERR, "监听'%s'失败。", ctx->master_addr[i]);
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);
        }
        else
        {
            /*减少引用计数。*/
            abcdk_comm_easy_unref(&easy_tmp);
        }
    }

    abcdk_comm_start(1);
    return 0;

final_error:

    return -1;
}

void _abcdk_vmtxsvr_stop(abcdk_vmtxsvr_t *ctx)
{
    abcdk_comm_stop();
}

int _abcdk_vmtxsvr_do_elect_leader(abcdk_comm_easy_t *easy, uint8_t *iswin)
{
    abcdk_comm_message_t *rsp = NULL;
    uint8_t *rsp_p = NULL;
    uint8_t req[3];
    int chk;

    ABCDK_PTR2U16(req, 0) = abcdk_endian_h_to_b16(ABCDK_VMTX_COMMAND_ELECT_LEADER);
    ABCDK_PTR2U8(req, 2) = 1;

    chk = abcdk_comm_easy_request(easy, req, 3, &rsp);
    if (chk == 0)
    {
        rsp_p = abcdk_comm_message_data(rsp);

        *iswin = (ABCDK_PTR2I8(rsp_p, 6) == 1 ? 1 : 0);
    }

    return chk;
}

void _abcdk_vmtxsvr_runloop(abcdk_vmtxsvr_t *ctx)
{
    abcdk_comm_easy_t *master_easy[2] = {0};
    int easy_pipe = 0;
    uint8_t iswin = 0;
    int chk;

    while(1)
    {
        for (int i = 0; i < 2; i++)
        {
            if (!master_easy[i])
                master_easy[i] = abcdk_comm_easy_connect(NULL, &ctx->master_sockaddr[i], _abcdk_vmtxsvr_server_client_request_cb, NULL);
            if (master_easy[i])
                abcdk_comm_easy_set_timeout(master_easy[i],-1);
        }

        if(ctx->role == ABCDK_VMTXSVR_ROLE_STANDBY)
        {
            easy_pipe = 1;
            if(abcdk_sockaddr_where(&ctx->master_sockaddr[1], 1))
                easy_pipe = 0;
            else if(abcdk_sockaddr_where(&ctx->master_sockaddr[1], 2))
                easy_pipe = 1;

            chk = _abcdk_vmtxsvr_do_elect_leader(master_easy[easy_pipe],&iswin);

            if(chk != 0)
            {
                abcdk_comm_easy_unref(&master_easy[easy_pipe]);
            }
            else
            {
                ctx->role = ABCDK_VMTXSVR_ROLE_MASTER;
                abcdk_log_printf(LOG_INFO,"I am MASTER!!!");
            }
        }


        sleep(1);
    }
}

void _abcdk_vmtxsvr_dowork(abcdk_vmtxsvr_t *ctx)
{
    int addr_num = 0;
    int chk;

    abcdk_thread_setname("dowork");

    ctx->client_listen = abcdk_option_get(ctx->args, "--client-listen", 0, "/tmp/abcdk-vmtx.sock");

    for (int i = 0; i < 2; i++)
    {
        ctx->master_addr[i] = abcdk_option_get(ctx->args, "--master-address", i, NULL);
        if (!ctx->master_addr[i])
            continue;

        ctx->master_sockaddr[i].family = AF_INET; //设置默认的IP协议。
        chk = abcdk_sockaddr_from_string(&ctx->master_sockaddr[i], ctx->master_addr[i], 1);
        if (chk != 0)
        {
            abcdk_log_printf(LOG_WARNING, "‘%s’格式错误或无法解析。", ctx->master_addr[i]);
            continue;
        }

        addr_num += 1;
    }

    if (addr_num <= 0)
    {
        abcdk_log_printf(LOG_ERR, "集群需要指定一个主节点（可以是本机），一个备用主节点（非必须）。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    /*检查是地址否指向本节点。*/
    chk = 0;
    for (int i = 0; i < 2; i++)
    {
        if (ctx->master_sockaddr[i].family == AF_INET || ctx->master_sockaddr[i].family == AF_INET6)
            chk += (abcdk_sockaddr_where(&ctx->master_sockaddr[i], 1) ? 1 : 0);
    }

    /*如果地址未同时指向本节点，则检查地址是否指向相同的节点。*/
    if (chk != 2 && addr_num == 2)
    {
        if (abcdk_sockaddr_compare(&ctx->master_sockaddr[0], &ctx->master_sockaddr[1]) & 0x01)
            chk = 2;
    }

    if (chk == 2)
    {
        abcdk_log_printf(LOG_ERR, "主节点和备用主节点需要在不同的主机部署。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }
    else if(chk == 1)
    {
        ctx->role = ABCDK_VMTXSVR_ROLE_STANDBY;
    }
    else 
    {
        ctx->role = ABCDK_VMTXSVR_ROLE_SLAVE;
    }

    chk = _abcdk_vmtxsvr_start(ctx);
    if (chk != 0)
        goto final;

    _abcdk_vmtxsvr_register_signal(ctx);
    _abcdk_vmtxsvr_runloop(ctx);

final:

    _abcdk_vmtxsvr_stop(ctx);

    return;
}

int abcdk_vmtx_server(abcdk_tree_t *args)
{
    abcdk_vmtxsvr_t ctx = {0};
    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_vmtxsvr_usage();
    }
    else
    {
        _abcdk_vmtxsvr_dowork(&ctx);
    }

    return ctx.errcode;
}