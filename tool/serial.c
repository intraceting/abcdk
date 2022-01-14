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
#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/termios.h"
#include "abcdk-util/thread.h"

typedef struct _abcdkserial_ctx
{
    int errcode;

    abcdk_tree_t *args;

    abcdk_thread_t io;
    const char *dev;
    int fd;
    int baudrate;
    int bits;
    int parity;
    int stop;
    
}abcdkserial_ctx;

void _abcdkserial_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d-%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --dev < DEVICE > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n  简单的串口调试工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t  显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t  显示版本信息。\n");

    fprintf(stderr, "\n\t--dev < DEVICE >\n");
    fprintf(stderr, "\t  串口设备。\n");

    fprintf(stderr, "\n\t--baudrate < NUMBER >\n");
    fprintf(stderr, "\t  波特率。默认：9600\n");

    fprintf(stderr, "\n\t--bits < NUMBER >\n");
    fprintf(stderr, "\t  数据位。默认：8\n");
    fprintf(stderr, "\n\t  5：5 bits\n");
    fprintf(stderr, "\t  6：6 bits\n");
    fprintf(stderr, "\t  7：7 bits\n");
    fprintf(stderr, "\t  8：8 bits\n");

    fprintf(stderr, "\n\t--parity < NUMBER >\n");
    fprintf(stderr, "\t  效验位。默认：无\n");
    fprintf(stderr, "\n\t  1：奇校验\n");
    fprintf(stderr, "\t  2：偶校验\n");

    fprintf(stderr, "\n\t--stop < NUMBER >\n");
    fprintf(stderr, "\t  停止位。默认：1\n");
    fprintf(stderr, "\n\t  1：1 bits\n");
    fprintf(stderr, "\t  2：2 bits\n");



    ABCDK_ERRNO_AND_RETURN0(0);
}

void *_abcdkserial_io(void *param)
{
    abcdkserial_ctx *ctx = (abcdkserial_ctx*)param;
    int events = 0;
    uint8_t rbuf[100] = {0};
    ssize_t rlen = -1;

    for (;;)
    {
        events = abcdk_poll(ctx->fd, 0x01, -1);

        if (events & 0x02)
        {
        }

        if (events & 0x01)
        {
            memset(rbuf, 0, 100);
            rlen = read(ctx->fd, rbuf, 100);
            if (rlen == 0)
                break;
            else if (rlen == -1)
            {
                if (errno != EAGAIN)
                    break;
            }
            else
            {
                fprintf(stdout, "%s", rbuf);
            }
        }
    }

    return NULL;
}


void _abcdkserial_work(abcdkserial_ctx *ctx)
{
    ctx->io.handle = -1;
    ctx->fd = -1;
    int chk;
    
    ctx->dev = abcdk_option_get(ctx->args, "--dev", 0, NULL);
    ctx->baudrate = abcdk_option_get_int(ctx->args, "--baudrate", 0, 9600);
    ctx->bits = abcdk_option_get_int(ctx->args, "--bits", 0, 8);
    ctx->parity = abcdk_option_get_int(ctx->args, "--parity", 0, -1);
    ctx->stop = abcdk_option_get_int(ctx->args, "--stop", 0, 1);

    if (!ctx->dev || !*ctx->dev)
    {
        syslog(LOG_ERR, "'--dev DEVICE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->dev, F_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->dev, strerror(errno));
        goto final;
    }

    ctx->fd = abcdk_open(ctx->dev, 1, 1, 0);
    if (ctx->fd < 0)
    {
        syslog(LOG_WARNING, "以读写权限打开设备失败，尝试以只读权限打开。");
        
        ctx->fd = abcdk_open(ctx->dev, 0, 1, 0);
    }

    if (ctx->fd < 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->dev, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    ctx->io.routine = _abcdkserial_io;
    ctx->io.opaque = ctx;

    chk = abcdk_thread_create(&ctx->io, 1);
    if (chk != 0)
    {
        syslog(LOG_ERR, " %s.",strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    abcdk_thread_join(&ctx->io);

final:

    abcdk_closep(&ctx->fd);
}

int main(int argc, char **argv)
{
    abcdkserial_ctx ctx = {0};

    /*中文，UTF-8*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        goto final;

    abcdk_getargs(ctx.args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkserial_print_usage(ctx.args, 0);
    }
    else if (abcdk_option_exist(ctx.args, "--version"))
    {
        _abcdkserial_print_usage(ctx.args, 1);
    }
    else
    {
        _abcdkserial_work(&ctx);
    }

final:
    
    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}