/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

/*简单的虚拟网络。*/
typedef struct _abcdkvnet
{
    int errcode;
    abcdk_option_t *args;

    /*日志。*/
    abcdk_logger_t *logger;

    

}abcdkvnet_t;

typedef struct _abcdkvnet_node
{
    abcdk_sockaddr_t addr4;
    abcdk_sockaddr_t addr6;

}abcdkvnet_node_t;


static void _abcdkvnet_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的虚拟网络。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--log-path < PATH >\n");
    fprintf(stderr, "\t\t日志路径。默认：/tmp/abcdk/log/\n");

    fprintf(stderr, "\n\t--daemon < INTERVAL > \n");
    fprintf(stderr, "\t\t启用后台守护模式(秒)，1～60之间有效。默认：30\n");
    fprintf(stderr, "\t\t注：此功能不支持supervisor或类似的工具。\n");

    fprintf(stderr, "\n\t--max-client < NUMBER >\n");
    fprintf(stderr, "\t\t最大连接数。默认：系统限定的1/2\n");
}

static void _abcdkvnet_process(abcdkvnet_t *ctx)
{

}

static int __abcdkvnet_daemon_process_cb(void *opaque)
{
    abcdkvnet_t *ctx = (abcdkvnet_t *)opaque;

    _abcdkvnet_process(ctx);

    return 0;
}

static void _abcdkvnet_daemon(abcdkvnet_t *ctx)
{
    abcdk_logger_t *logger;
    const char *log_path = NULL;
    int interval;

    log_path = abcdk_option_get(ctx->args, "--log-path", 0, "/tmp/abcdk/log/");
    interval = abcdk_option_get_int(ctx->args, "--daemon", 0, 30);
    interval = ABCDK_CLAMP(interval, 1, 60);

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path, "proxy-daemon.log", "proxy-daemon.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_set_log(abcdk_logger_from_trace, logger);

    abcdk_proc_daemon(interval, __abcdkvnet_daemon_process_cb, ctx);

    /*关闭日志。*/
    abcdk_logger_close(&logger);
}

int abcdk_tool_vnet(abcdk_option_t *args)
{
    abcdkvnet_t ctx = {0};
    int chk;

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkvnet_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args, "--daemon"))
        {
            fprintf(stderr, "进入后台守护模式。\n");
            daemon(1, 0);

            _abcdkvnet_daemon(&ctx);
        }
        else
        {
            _abcdkvnet_process(&ctx);
        }
    }

    return 0;
}

