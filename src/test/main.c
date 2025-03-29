/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"


static struct _abcdk_test_entry
{
    /** 名字。*/
    const char *name;

    /** 
     * 回调函数。
     * 
     * @return 出错码。
    */
    int (*func_cb)(abcdk_option_t *args);
}abcdk_test_entry[] = {
    {"http",abcdk_test_http},
    {"http2",abcdk_test_http2},
    {"uri",abcdk_test_uri},
    {"log",abcdk_test_log},
    {"any",abcdk_test_any},
    {"exec",abcdk_test_exec},
    {"com",abcdk_test_com},
    {"path",abcdk_test_path},
    {"ffmpeg",abcdk_test_ffmpeg},
    {"drm",abcdk_test_drm},
    {"worker",abcdk_test_worker},
    {"ping",abcdk_test_ping},
    {"onvif",abcdk_test_onvif},
    {"dhcp",abcdk_test_dhcp},
    {"tipc",abcdk_test_tipc},
    {"timer",abcdk_test_timer},
    {"tun",abcdk_test_tun},
    {"srpc",abcdk_test_srpc},
    {"fmp4",abcdk_test_fmp4},
    {"sudp",abcdk_test_sudp},
    {"pem",abcdk_test_pem},
    {"usb",abcdk_test_usb},
    {"rand",abcdk_test_rand},
    {"runonce",abcdk_test_runonce},
    {"ncurses",abcdk_test_ncurses},
    {"gtk",abcdk_test_gtk},
    {"torch",abcdk_test_torch},
    {"stitcher",abcdk_test_stitcher},
};

void _abcdk_test_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR, ABCDK_VERSION_RELEASE);

    fprintf(stderr, "\n命令：\n");
    fprintf(stderr, "\n\t");

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_test_entry); i++)
    {
        fprintf(stderr, "%s ",abcdk_test_entry[i].name);
    }

    fprintf(stderr, "\n");

    fprintf(stderr, "\n示例：\n");
    fprintf(stderr, "\n\t%s < CMD > [ ... ]\n", name);
}

struct _abcdk_test_entry *_abcdk_test_entry_find(abcdk_option_t *args)
{
    const char *name_p = abcdk_option_get(args,"--",1,NULL);

    if(!name_p)
        return NULL;
    
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_test_entry); i++)
    {
        if(abcdk_strcmp(abcdk_test_entry[i].name,name_p,0)==0)
            return &abcdk_test_entry[i];
    }

    return NULL;
}

int _abcdk_test_dispatch(abcdk_option_t *args)
{
    int errcode = 0;
    struct _abcdk_test_entry *entry_p = NULL;

    entry_p = _abcdk_test_entry_find(args);

    if (!entry_p)
    {
        _abcdk_test_print_usage();
        ABCDK_ERRNO_AND_GOTO1(errcode = EINVAL, final);
    }

    errcode = entry_p->func_cb(args);

final:

    return errcode;
}

void *_abcdk_test_signal_cb(void *opaque)
{
    siginfo_t info = {0};
    int chk;

    while (1)
    {
#if 1
        chk = abcdk_signal_wait(&info,NULL, -1);
#else 
        sigset_t sigs = {0};
        sigemptyset(&sigs);
        abcdk_signal_set(&sigs,0,SIGABRT,-1);
        chk = abcdk_signal_wait(&info,&sigs, -1);
#endif 
        if (chk <= 0)
            break;
        
        if (SI_USER == info.si_code)
            fprintf(stderr, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info.si_signo, info.si_errno, info.si_code, info.si_pid, info.si_uid);
        else
            fprintf(stderr, "signo(%d),errno(%d),code(%d)\n", info.si_signo, info.si_errno, info.si_code);

    }

    return NULL;
}

int main(int argc, char **argv)
{
    abcdk_option_t *args = NULL;
    sigset_t sigs = {0};
    abcdk_thread_t sig_thread = {0};
    int errcode = 0;


    // abcdk_signal_fill(&sigs,SIGTRAP,SIGKILL,SIGSEGV,SIGSTOP,-1);
    // abcdk_signal_block(&sigs,NULL);

    // sig_thread.routine = _abcdk_test_signal_cb;
    // abcdk_thread_create(&sig_thread,0);
    
    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

    abcdk_openssl_init();


#ifdef __cuda_cuda_h__

    abcdk_torch_init_cuda(0);

#endif //__cuda_cuda_h__

    abcdk_torch_init_host(0);

    args = abcdk_option_alloc("--");
    if (!args)
        ABCDK_ERRNO_AND_GOTO1(errcode = errno,final_end);
   
    /*解析参数。*/
    abcdk_getargs(args, argc, argv);

    abcdk_getargs_fprintf(args,stderr,"\n","");

    abcdk_logger_t *logger;
    const char *log_path = abcdk_option_get(args, "--log-path", 0, "/tmp/abcdk/log/");

    /*打开日志。*/
    logger = abcdk_logger_open2(log_path, "test.log", "test.%d.log", 10, 10, 0, 1);

    /*注册为轨迹日志。*/
    abcdk_trace_printf_set_callback(abcdk_logger_from_trace, logger);

#ifdef HAVE_FFMPEG
    abcdk_avlog_redirect2trace();
#endif //HAVE_FFMPEG

    errcode = _abcdk_test_dispatch(args);

    /*关闭日志。*/
    abcdk_logger_close(&logger);

final_end:

    abcdk_openssl_cleanup();
    
    abcdk_option_free(&args);

    exit(errcode);
}