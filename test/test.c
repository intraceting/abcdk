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
    int (*func_cb)(abcdk_tree_t *args);
}abcdk_test_entry[] = {
    {"http",abcdk_test_http},
    {"uri",abcdk_test_uri},
    {"log",abcdk_test_log},
    {"rpc",abcdk_test_rpc},
    {"iconv",abcdk_test_iconv},
    {"exec",abcdk_test_exec},
    {"com",abcdk_test_com},
    {"path",abcdk_test_path}
};

void _abcdk_test_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

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

struct _abcdk_test_entry *_abcdk_test_entry_find(abcdk_tree_t *args)
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

int _abcdk_test_dispatch(abcdk_tree_t *args)
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

int _abcdk_test_signal_cb(const siginfo_t *info, void *opaque)
{

    fprintf(stderr, "收到信号（%d)。", info->si_signo);

    if (SIGILL == info->si_signo ||
        SIGTERM == info->si_signo ||
        SIGINT == info->si_signo ||
        SIGQUIT == info->si_signo)
    {
        abcdk_atomic_store((uint32_t *)abcdk_register(32, 255), 1);
        return -1;
    }
    else
    {

        fprintf(stderr, "如果希望停止服务，按Ctrl+c组合键，或发送SIGTERM(15)信号。例：kill -s 15 %d", getpid());
    }

    return 1;
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args = NULL;
    abcdk_signal_t sig = {0};
    int errcode = 0;

    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = _abcdk_test_signal_cb;
    abcdk_sigwaitinfo_async(&sig);

    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

#ifdef HAVE_OPENSSL

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();

#endif //HAVE_OPENSSL

    /*申请参数存储空间。*/
    args = abcdk_tree_alloc3(1);
    if (!args)
        ABCDK_ERRNO_AND_GOTO1(errcode = errno,final);
    
    /*解析参数。*/
    abcdk_getargs(args, argc, argv, "--");

    errcode = _abcdk_test_dispatch(args);

final:
    
    abcdk_tree_free(&args);

    return errcode;
}