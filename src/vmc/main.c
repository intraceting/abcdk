/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include <uuid/uuid.h>
#include "context.h"
#include "service.h"

void _abcdk_vmc_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n\t--help\n");

    fprintf(stderr, "\n\t--id < NUMBER >\n");
    fprintf(stderr, "\t\t节点ID（1～65535）。\n");

    fprintf(stderr, "\n\t--master-address < ADDRESS [ ADDRESS ] >\n");
    fprintf(stderr, "\t\t主管理节点地址（IPv4,IPv6）。\n");
    fprintf(stderr, "\n\t\tIPv4：Address:Port\n");
    fprintf(stderr, "\t\tIPv6：[Address]:Port\n");
}

int _abcdk_vmc_signal_cb(const siginfo_t *info, void *opaque)
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

void _abcdk_vmc_wait_signal(abcdk_vmc_t *ctx)
{
    abcdk_signal_t sig;

    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = _abcdk_vmc_signal_cb;
    sig.opaque = ctx;

    abcdk_sigwaitinfo(&sig, -1);
}


void _abcdk_vmc_dowork(abcdk_vmc_t *ctx)
{
    abcdk_sockaddr_t addr = {0};
    const char *addr_p = NULL;
    int addr_num = 0;
    int chk;

    abcdk_thread_setname("main-thread");

    ctx->id = abcdk_option_get_int(ctx->args, "--id", 0, 0, 0);
    if (ctx->id == 0)
    {
        abcdk_log_printf(LOG_ERR, "节点ID范围在1～65535之间。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    for (int i = 0; i < 2; i++)
    {
        addr_p = abcdk_option_get(ctx->args, "--master-address", i, NULL);
        if (!addr_p)
            continue;

        chk = abcdk_sockaddr_from_string(&addr, addr_p, 0);
        if (chk != 0)
        {
            abcdk_log_printf(LOG_WARNING, "‘%s’格式错误。",addr_p);
            continue;
        }

        ctx->masters[i].addr = addr;
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
        if (ctx->masters[i].addr.family == AF_INET || ctx->masters[i].addr.family == AF_INET6)
            chk += (abcdk_sockaddr_where(&ctx->masters[i].addr, 1)? 1 : 0);
    }

    /*如果地址未同时指向本节点，则检查地址是否指向相同的节点。*/
    if (chk != 2 && addr_num == 2)
    {
        if (abcdk_sockaddr_compare(&ctx->masters[0].addr, &ctx->masters[1].addr) & 0x01)
            chk = 2;
    }

    if (chk == 2)
    {
        abcdk_log_printf(LOG_ERR, "主节点和备用主节点需要在不同的主机部署。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    abcdk_vmc_start(ctx);
    _abcdk_vmc_wait_signal(ctx);
    abcdk_vmc_stop(ctx);

final:

    return;
}

int main(int argc, char **argv)
{
    abcdk_vmc_t ctx = {0};

    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

    /*申请参数存储空间。*/
    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        ABCDK_ERRNO_AND_GOTO1(ctx.errcode = errno, final);

    /*解析参数。*/
    abcdk_getargs(ctx.args, argc, argv, "--");

    /*记录日志。*/
    abcdk_log_open(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_vmc_usage();
    }
    else
    {
        _abcdk_vmc_dowork(&ctx);
    }

final:

    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}