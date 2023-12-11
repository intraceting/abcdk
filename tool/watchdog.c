/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

typedef struct _abcdkwatchdog
{
    int errcode;
    abcdk_option_t *args;

    abcdk_logger_t *logger;

    /*0：运行，1：退出。*/
    volatile int exitflag;

}abcdkwatchdog_t;


void _abcdkwatchdog_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的看门狗。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

}

void _abcdkwatchdog_work_monitor(abcdkwatchdog_t *ctx, abcdk_option_t *conf)
{
    const char *root_path_p = abcdk_option_get(conf,"--root-path",0,NULL);
    const char *work_path_p = abcdk_option_get(conf,"--work-path",0,NULL);
    const char *cmd_p = abcdk_option_get(conf,"--cmd",0,"");
}

void _abcdkwatchdog_work_process(void *opaque, uint32_t tid)
{
    abcdkwatchdog_t *ctx = (abcdkwatchdog_t*)opaque;
    
    const char *conf_p = abcdk_option_get(ctx->args,"--conf",tid,"");

    while (1)
    {
        if (abcdk_atomic_compare(&ctx->exitflag, 1))
            break;

        abcdk_option_t *conf_args = abcdk_option_alloc("--");
        abcdk_getargs_file(conf_args, conf_p, '\n', '#', NULL);

        _abcdkwatchdog_work_monitor(ctx, conf_args);

        abcdk_option_free(&conf_args);
    }
}

void _abcdkwatchdog_wait_signal(abcdkwatchdog_t *ctx)
{
    siginfo_t info = {0};
    int chk;

    while (1)
    {
        chk = abcdk_signal_wait(&info,NULL, -1);
        if (chk <= 0)
            break;

        abcdk_logger_dump_siginfo(ctx->logger, LOG_WARNING, &info);

        if (SIGILL == info.si_signo || SIGTERM == info.si_signo || SIGINT == info.si_signo || SIGQUIT == info.si_signo)
            break;
        else
            abcdk_logger_printf(ctx->logger, LOG_WARNING, "如果希望停止服务，按Ctrl+c组合键或发送SIGTERM(15)信号。例：kill -s 15 %d\n", getpid());
    }

    return;
}

void _abcdkwatchdog_wrok(abcdkwatchdog_t *ctx)
{
    sigset_t sigs = {0};
    abcdk_parallel_t *parallel_ctx = NULL;
    int conf_num;

    /*阻塞信号。*/
    abcdk_signal_fill(&sigs, SIGTRAP, SIGKILL, SIGSEGV, SIGSTOP, -1);
    abcdk_signal_block(&sigs, NULL);

    conf_num = abcdk_option_count(ctx->args, "--conf");

    parallel_ctx = abcdk_parallel_alloc(conf_num);

    /*并发执行所有任务。*/
    abcdk_parallel_invoke(parallel_ctx,conf_num,ctx,_abcdkwatchdog_work_process);

    _abcdkwatchdog_wait_signal(ctx);

    abcdk_atomic_store(&ctx->exitflag,1);

    abcdk_parallel_free(&parallel_ctx);
}

int abcdk_tool_watchdog(abcdk_option_t *args)
{
    abcdkwatchdog_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkwatchdog_print_usage(ctx.args);
    }
    else
    {
        if (abcdk_option_exist(ctx.args,"--daemon"))
        {
            fprintf(stderr, "进入后台驻留模式。\n");
            daemon(1, 0);
        }

        /*打开日志。*/
        ctx.logger = abcdk_logger_open("/tmp/abcdk/log/watchlog.log", "watchlog.%d.log", 10, 10, 0, 1);

        abcdk_logger_printf(ctx.logger, LOG_INFO, "启动……");

        _abcdkwatchdog_wrok(&ctx);

        abcdk_logger_printf(ctx.logger, LOG_INFO, "停止。");

        /*关闭日志。*/
        abcdk_logger_close(&ctx.logger);
    }

    return ctx.errcode;
}