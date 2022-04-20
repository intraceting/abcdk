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

enum _abcdk_vmtx_server_constant
{
    /** 主机。*/
    ABCDK_VMTX_SERVER_STATUS_MASTER = 1,
#define ABCDK_VMTX_SERVER_STATUS_MASTER ABCDK_VMTX_SERVER_STATUS_MASTER

    /** 备机。*/
    ABCDK_VMTX_SERVER_STATUS_STANDBY = 2,
#define ABCDK_VMTX_SERVER_STATUS_STANDBY ABCDK_VMTX_SERVER_STATUS_STANDBY

    /** 从机。*/
    ABCDK_VMTX_SERVER_STATUS_SLAVE = 3,
#define ABCDK_VMTX_SERVER_STATUS_SLAVE ABCDK_VMTX_SERVER_STATUS_SLAVE

    /**
     * 注册。
     *
     * REQ: cmd(2)
     * RSP: cmd(2) + errno(4) + status(1)
     *      status: 1 主机，2 备机。
    */
    ABCDK_VMTX_SERVER_COMMAND_REGISTER = 1,
#define ABCDK_VMTX_SERVER_COMMAND_REGISTER ABCDK_VMTX_SERVER_COMMAND_REGISTER

    /**
     * 选举主节点。比较IP地址，地址大的为领导者。
     *
     * REQ: cmd(2)
     * RSP: cmd(2) + errno(4) +  opinion(1)
     *      opinion: 1 同意，2 不同意。
    */
    ABCDK_VMTX_SERVER_COMMAND_VOTE = 2
#define ABCDK_VMTX_SERVER_COMMAND_VOTE ABCDK_VMTX_SERVER_COMMAND_VOTE

};

typedef struct _abcdk_vmtx_server
{
    int errcode;
    abcdk_tree_t *args;

    /* 退出标记。*/
    volatile int exit_flag;

    /* 运行锁。*/
    int lock_pid;
    int lock_fd;
    const char *lock_file;

    /* 客户端地址(Unix Socket)。*/
    const char *client_listen;
    abcdk_sockaddr_t client_sockaddr;

    /* Master地址(主、备)。*/
    const char *master_addr[2];
    abcdk_sockaddr_t master_sockaddr[2];
    abcdk_comm_easy_t *master_easy[2];

    /* 状态。*/
    int status;

} abcdk_vmtx_server_t;

typedef struct _abcdk_vmtx_server_io
{
    uint8_t mod;
    abcdk_vmtx_server_t *ctx;
} abcdk_vmtx_server_io_t;

void _abcdk_vmtx_server_usage()
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

int _abcdk_vmtx_server_signal_cb(const siginfo_t *info, void *opaque)
{
    abcdk_vmtx_server_t *ctx;
    int chk;

    ctx = (abcdk_vmtx_server_t *)opaque;

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

void _abcdk_vmtx_server_register_signal(abcdk_vmtx_server_t *ctx)
{
    abcdk_signal_t sig;

    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = _abcdk_vmtx_server_signal_cb;
    sig.opaque = ctx;

    abcdk_sigwaitinfo_async(&sig);
}

void _abcdk_vmtx_server_msg_vote(abcdk_vmtx_server_t *ctx,abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdk_sockaddr_t localaddr = {0},remoteaddr = {0};
    uint8_t rsp[7];
    
    abcdk_comm_easy_get_sockaddr(easy,&localaddr,&remoteaddr);

    ABCDK_PTR2U16(rsp, 0) = ABCDK_PTR2U16(req,0);

    if (ctx->status == ABCDK_VMTX_SERVER_STATUS_MASTER)
    {
        /*如果本机为领导者，则拒绝其它主机选主。*/
        ABCDK_PTR2U32(rsp, 2) = 0;
        ABCDK_PTR2U8(rsp, 6) = 2;
    }
    else if (ctx->status == ABCDK_VMTX_SERVER_STATUS_STANDBY)
    {
        /*当两个主机同时在线，选IP地址大的为主节点。*/
        ABCDK_PTR2U32(rsp, 2) = 0;
        ABCDK_PTR2U8(rsp, 6) = ((abcdk_sockaddr_compare(&localaddr, &remoteaddr) > 0) ? 2 : 1);
    }
    else
    {
        /* 
         * 1：当前是跟随者，拒绝当前的选主流程。
         * 2：应当不会走这个分支。
        */
        ABCDK_PTR2U32(rsp, 2) = abcdk_endian_h_to_b32(EINTR); //不支持。
        ABCDK_PTR2U8(rsp, 6) = 2;
    }

    abcdk_comm_easy_response(easy,rsp,7);
}

void _abcdk_vmtx_server_msg_reg(abcdk_vmtx_server_t *ctx,abcdk_comm_easy_t *easy, const void *req, size_t len)
{

}

void _abcdk_vmtx_server_msg_reg_rsp(abcdk_vmtx_server_t *ctx,abcdk_comm_easy_t *easy, const void *req, size_t len)
{

}

static struct _abcdk_vmtx_server_msg_entry
{
    uint8_t mod;
    uint16_t cmd;
    void (*func_cb)(abcdk_vmtx_server_t *ctx, abcdk_comm_easy_t *easy, const void *req, size_t len);
} abcdk_vmtx_server_msg_entry[] = {
    {2,ABCDK_VMTX_SERVER_COMMAND_REGISTER,_abcdk_vmtx_server_msg_reg},
    {2,ABCDK_VMTX_SERVER_COMMAND_VOTE, _abcdk_vmtx_server_msg_vote},
    {3,ABCDK_VMTX_SERVER_COMMAND_REGISTER,_abcdk_vmtx_server_msg_reg_rsp}
};

void _abcdk_vmtx_server_msg_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdk_vmtx_server_io_t *io;
    uint16_t cmd;

    io = (abcdk_vmtx_server_io_t *)abcdk_comm_easy_get_userdata(easy);

    if (req != NULL && len > 0)
    {
        cmd = abcdk_endian_b_to_h16(ABCDK_PTR2U16(req, 0));

        for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_vmtx_server_msg_entry); i++)
        {
            if (abcdk_vmtx_server_msg_entry[i].mod != io->mod)
                continue;

            if (abcdk_vmtx_server_msg_entry[i].cmd != cmd)
                continue;

            abcdk_vmtx_server_msg_entry[i].func_cb(io->ctx, easy, req, len);
        }
    }
    else
    {
        printf("%s\n", __FUNCTION__);
    }
}

int _abcdk_vmtx_server_start(abcdk_vmtx_server_t *ctx)
{
    abcdk_comm_easy_t *easy_tmp;
    int chk;

    /*删除可能存在的unixsock文件。*/
    unlink(ctx->client_listen);

    /*监听客户端命令地址。*/
    abcdk_sockaddr_from_string(&ctx->client_sockaddr,ctx->client_listen,0);

    static abcdk_vmtx_server_io_t client_io = {0};
    client_io.mod = 1, client_io.ctx = ctx;

    easy_tmp = abcdk_comm_easy_listen(NULL, &ctx->client_sockaddr, _abcdk_vmtx_server_msg_cb, &client_io);
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

        static abcdk_vmtx_server_io_t service_io = {0};
        service_io.mod = 2, service_io.ctx = ctx;

        easy_tmp = abcdk_comm_easy_listen(NULL, &ctx->master_sockaddr[i], _abcdk_vmtx_server_msg_cb, &service_io);
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

void _abcdk_vmtx_server_stop(abcdk_vmtx_server_t *ctx)
{
    abcdk_comm_stop();
}

void _abcdk_vmtx_server_do_vote(abcdk_vmtx_server_t *ctx)
{
    abcdk_comm_message_t *rsp = NULL;
    abcdk_comm_easy_t *easy = NULL;
    int pipe = 0;
    uint8_t req[2];
    uint8_t *rsp_p;
    uint32_t err;
    uint8_t opinion;
    int chk;

    /*只给远程的master发就行了。*/
    pipe = (abcdk_sockaddr_where(&ctx->master_sockaddr[1], 1) ? 1 : 0);

    if (!ctx->master_easy[pipe])
        return;

    /*假定远程同意本次选举。*/
    opinion = 1;

    /*填充请求命令。*/
    ABCDK_PTR2U16(req, 0) = abcdk_endian_h_to_b16(ABCDK_VMTX_SERVER_COMMAND_VOTE);

    chk = abcdk_comm_easy_request(ctx->master_easy[pipe], req, 2, &rsp);
    if (chk == 0)
        goto final;

    if (rsp)
    {
        /*解析应答命令。*/
        rsp_p = abcdk_comm_message_data(rsp);
        err = abcdk_endian_b_to_h32(ABCDK_PTR2U32(rsp_p, 2));
        opinion = ABCDK_PTR2I8(rsp_p, 6);
    }

final:

    /*选举成功后，更改自身的状态。*/
    if (opinion == 1)
        ctx->status = ABCDK_VMTX_SERVER_STATUS_MASTER;

    abcdk_comm_message_unref(&rsp);
}

void _abcdk_vmtx_server_runloop(abcdk_vmtx_server_t *ctx)
{
    while(1)
    {
        for (int i = 0; i < 2; i++)
        {
            if (!ctx->master_easy[i])
            {
                static abcdk_vmtx_server_io_t service_io = {0};
                service_io.mod = 3, service_io.ctx = ctx;

                ctx->master_easy[i] = abcdk_comm_easy_connect(NULL, &ctx->master_sockaddr[i], _abcdk_vmtx_server_msg_cb, &service_io);
                if (ctx->master_easy[i])
                    abcdk_comm_easy_set_timeout(ctx->master_easy[i], 5);
            }
        }

        if(ctx->status == ABCDK_VMTX_SERVER_STATUS_STANDBY)
        {

        }
        else
        {

        }

        sleep(1);
    }
}

void _abcdk_vmtx_server_dowork(abcdk_vmtx_server_t *ctx)
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
        if (abcdk_sockaddr_compare(&ctx->master_sockaddr[0], &ctx->master_sockaddr[1]) == 0)
            chk = 2;
    }

    if (chk == 2)
    {
        abcdk_log_printf(LOG_ERR, "主节点和备用主节点需要在不同的主机部署。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    /*初始化状态。*/
    ctx->status = ((chk == 1) ? ABCDK_VMTX_SERVER_STATUS_STANDBY : ABCDK_VMTX_SERVER_STATUS_SLAVE);

    chk = _abcdk_vmtx_server_start(ctx);
    if (chk != 0)
        goto final;

    _abcdk_vmtx_server_register_signal(ctx);
    _abcdk_vmtx_server_runloop(ctx);

final:

    _abcdk_vmtx_server_stop(ctx);

    return;
}

int abcdk_vmtx_server(abcdk_tree_t *args)
{
    abcdk_vmtx_server_t ctx = {0};
    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_vmtx_server_usage();
    }
    else
    {
        _abcdk_vmtx_server_dowork(&ctx);
    }

    return ctx.errcode;
}