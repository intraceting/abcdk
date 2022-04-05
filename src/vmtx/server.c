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
    int chk;

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

    return chk;
}

void _abcdk_vmtxsvr_wait_signal(abcdk_vmtxsvr_t *ctx)
{
    abcdk_signal_t sig;

    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = _abcdk_vmtxsvr_signal_cb;
    sig.opaque = ctx;

    abcdk_sigwaitinfo(&sig, -1);
}

void _abcdk_vmtxsvr_server_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
}

void _abcdk_vmtxsvr_client_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
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

void _abcdk_vmtxsvr_dowork(abcdk_vmtxsvr_t *ctx)
{
    int addr_num = 0;
    int chk;

    abcdk_thread_setname("vmtxsvr-dowork");

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

    chk = _abcdk_vmtxsvr_start(ctx);
    if (chk != 0)
        goto final;

    _abcdk_vmtxsvr_wait_signal(ctx);

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