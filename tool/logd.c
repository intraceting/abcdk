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
#include "util/general.h"
#include "util/thread.h"
#include "util/getargs.h"
#include "util/openssl.h"
#include "util/signal.h"
#include "util/map.h"
#include "shell/file.h"
#include "shell/proc.h"
#include "comm/easy.h"

#define ABCDKLOGD_SERVICE_MAX 65536

typedef struct _abcdklogd_policy
{
    /*工作路径。*/
    const char *workspace;
    /*分段数量。*/
    size_t segment_max;
    /*分段大小(MB)。*/
    size_t segment_size;

} abcdklogd_policy_t;


typedef struct _abcdklogd
{
    int errcode;
    abcdk_tree_t *args;

    int daemon;

    const char *ca_file;
    const char *ca_path;
    int ca_check_crl;

    const char *listen;
    abcdklogd_policy_t *policys;

    abcdk_mutex_t node_mutex;
    abcdk_map_t node_lists;
    abcdk_comm_t *comm;
    abcdk_comm_easy_t *listen_easy;

} abcdklogd_t;

typedef struct _abcdklogd_node
{
    abcdklogd_t *ctx;

    const char *from;

    abcdk_mutex_t service_mutex;
    abcdk_map_t service_lists;
} abcdklogd_node_t;

typedef struct _abcdklogd_service
{
    abcdklogd_node_t *node;

    uint16_t sid;
    abcdklogd_policy_t *policy;

    abcdk_mutex_t mutex;
    const char *pathfile;
    const char *namefmt;
    FILE *fp;

} abcdklogd_service_t;

void _abcdklogd_print_usage()
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 版本 %d.%d.%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);
    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);

    fprintf(stderr, "\n\t--daemon\n");
    fprintf(stderr, "\t\t驻留到后台。\n");

    fprintf(stderr, "\n\t--listen < ADDRESS >\n");
    fprintf(stderr, "\t\t监听地址。默认：127.0.0.1:65535\n");

    fprintf(stderr, "\n\t\t例：IPV4:PORT\n");
    fprintf(stderr, "\t\t例：IPV6,PORT\n");
    fprintf(stderr, "\t\t例：[IPV6]:PORT\n");

    fprintf(stderr, "\n\t--policy-conf < CONF >\n");
    fprintf(stderr, "\t\t策略配置文件。\n");
    fprintf(stderr, "\t\t注：配置文件中的key和value分行输入，当行首字符为#时，忽略此行。\n");

    fprintf(stderr, "\n\t\t--service < ID >\n");
    fprintf(stderr, "\t\t\t服务编号。\n");
    fprintf(stderr, "\t\t\t注：全局配置忽略此项。\n");

    fprintf(stderr, "\n\t\t--workspace < PATH >\n");
    fprintf(stderr, "\t\t\t工作路径。默认：/tmp/abcdk.logd/\n");

    fprintf(stderr, "\n\t\t--segment-max < SIZE >\n");
    fprintf(stderr, "\t\t\t分段数量。默认：10\n");

    fprintf(stderr, "\n\t\t--segment-size < SIZE >\n");
    fprintf(stderr, "\t\t\t分段大小(MB)。默认：10\n");
}

int _abcdklogd_signal_cb(const siginfo_t *info, void *opaque)
{
    if(SI_USER == info->si_code)
        fprintf(stderr, "signo(%d),errno(%d),code(%d),pid(%d),uid(%d)\n", info->si_signo, info->si_errno, info->si_code,info->si_pid,info->si_uid);
    else
        fprintf(stderr, "signo(%d),errno(%d),code(%d)\n", info->si_signo, info->si_errno, info->si_code);

    if (SIGILL == info->si_signo || SIGTERM == info->si_signo || SIGINT == info->si_signo || SIGQUIT == info->si_signo)
        return -1;
    else
        fprintf(stderr, "如果希望停止服务，按Ctrl+c组合键，或发送SIGTERM(15)信号。例：kill -s 15 %d", getpid());

    return 0;
}

void _abcdklogd_file_request(abcdklogd_t *ctx, abcdklogd_service_t *svc, uint64_t ts, uint16_t sid, pid_t pid, const char *name, const char *msg)
{
    int64_t fsize = 0;
    struct tm tm = {0};

    abcdk_mutex_lock(&svc->mutex,1);

    if(svc->fp)
    {
        fsize = abcdk_fsize(svc->fp);
        if (fsize >= svc->policy->segment_size * 1024 * 1024)
        {
            abcdk_fclosep(&svc->fp);
            abcdk_file_segment(svc->pathfile,svc->namefmt,svc->policy->segment_max);
        }
    }

    if (!svc->fp)
        svc->fp = fopen(svc->pathfile, "a");

    if (svc->fp)
    {
        abcdk_time_sec2tm(&tm, ts / 1000000, 0);
        fprintf(svc->fp, "%d%02d%02d.%02d%02d%02d.%06lu s%hu.p%d %s: %s\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec,
                ts % 1000000, sid, pid, name, msg);
        fflush(svc->fp);
    }

    abcdk_mutex_unlock(&svc->mutex);
}

void _abcdklogd_service_request(abcdklogd_t *ctx,const char *from, abcdklogd_node_t *node, const void *req, size_t len)
{
    abcdk_object_t *obj = NULL;
    uint64_t ts = 0;
    uint16_t sid = 0;
    pid_t pid = -1;
    char name[17] = {0};
    uint32_t dlen = 0;

    ts = abcdk_endian_b_to_h64(ABCDK_PTR2I64(req, 0));
    sid = abcdk_endian_b_to_h16(ABCDK_PTR2I16(req, 8));
    pid = abcdk_endian_b_to_h32(ABCDK_PTR2I32(req, 10));
    strncpy(name, ABCDK_PTR2I8PTR(req, 14), 16);
    dlen = abcdk_endian_b_to_h32(ABCDK_PTR2I32(req, 31));
    // fprintf(stderr,"%s\n",ABCDK_PTR2I8PTR(req,35));

    abcdk_mutex_lock(&node->service_mutex,1);
    obj = abcdk_map_find2(&node->service_lists,&sid,sizeof(abcdklogd_service_t));
    if(obj)
        obj = abcdk_object_refer(obj);
    abcdk_mutex_unlock(&node->service_mutex);

    if(obj)
    {
        _abcdklogd_file_request(node->ctx, (abcdklogd_service_t *)obj->pptrs[ABCDK_MAP_VALUE],
                                ts, sid, pid, name, ABCDK_PTR2I8PTR(req, 35));
        abcdk_object_unref(&obj);
    }
}

void _abcdklogd_node_request(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    abcdklogd_t *ctx = (abcdklogd_t *)abcdk_comm_easy_get_userdata(easy);
    char remote[NAME_MAX] = {0};
    abcdk_object_t *obj = NULL;

    abcdk_comm_easy_get_sockaddr_str(easy,NULL,remote);
    
    if(!req)
    {
        abcdk_mutex_lock(&ctx->node_mutex,1);
        abcdk_map_remove(&ctx->node_lists,remote,strlen(remote));
        abcdk_mutex_unlock(&ctx->node_mutex);
        return;
    }
    else
    {
        abcdk_mutex_lock(&ctx->node_mutex, 1);
        obj = abcdk_map_find(&ctx->node_lists, remote, strlen(remote), sizeof(abcdklogd_node_t));
        if(obj)
            obj = abcdk_object_refer(obj);
        abcdk_mutex_unlock(&ctx->node_mutex);

        if(obj)
        {
            _abcdklogd_service_request(ctx, remote, (abcdklogd_node_t *)obj->pptrs[ABCDK_MAP_VALUE], req, len);
            abcdk_object_unref(&obj);
        }
    }
}

void _abcdklogd_service_construct(abcdk_object_t *alloc, void *opaque)
{
    abcdklogd_node_t *node = (abcdklogd_node_t *)opaque;
    uint16_t sid = ABCDK_PTR2I16(alloc->pptrs[ABCDK_MAP_KEY],0);
    abcdklogd_service_t *svc = (abcdklogd_service_t*)alloc->pptrs[ABCDK_MAP_VALUE];

    svc->node = node;
    svc->sid = sid;
    svc->policy = &node->ctx->policys[sid];
    abcdk_mutex_init2(&svc->mutex,0);
    svc->pathfile = abcdk_heap_alloc(PATH_MAX);
    svc->namefmt = abcdk_heap_alloc(NAME_MAX);
    svc->fp = NULL;

    snprintf((char*)svc->pathfile,PATH_MAX,"%s/%s/s%hu/s%hu.log",svc->policy->workspace,node->from,sid,sid);
    snprintf((char*)svc->namefmt,NAME_MAX,"s%hu_%%d.log",sid);

    abcdk_mkdir(svc->pathfile,0600);

}

void _abcdklogd_service_destructor(abcdk_object_t *alloc, void *opaque)
{
    abcdklogd_service_t *svc = (abcdklogd_service_t*)alloc->pptrs[ABCDK_MAP_VALUE];

    abcdk_fclosep(&svc->fp);
    abcdk_heap_free((void*)svc->pathfile);
    abcdk_heap_free((void*)svc->namefmt);
    abcdk_mutex_destroy(&svc->mutex);
}

void _abcdklogd_node_construct(abcdk_object_t *alloc, void *opaque)
{
    abcdklogd_t *ctx = (abcdklogd_t*)opaque;
    char *key = (char*) alloc->pptrs[ABCDK_MAP_KEY];
    abcdklogd_node_t *node = (abcdklogd_node_t*)alloc->pptrs[ABCDK_MAP_VALUE];

    node->ctx = ctx;
    abcdk_mutex_init2(&node->service_mutex,0);
    abcdk_map_init(&node->service_lists,100);
    node->service_lists.construct_cb = _abcdklogd_service_construct;
    node->service_lists.destructor_cb = _abcdklogd_service_destructor;
    node->service_lists.opaque = node;

    node->from = (char*)abcdk_heap_alloc(NAME_MAX);
    memcpy((void*)node->from,alloc->pptrs[ABCDK_MAP_KEY],alloc->sizes[ABCDK_MAP_KEY]);
}

void _abcdklogd_node_destructor(abcdk_object_t *alloc, void *opaque)
{
    abcdklogd_node_t *node = (abcdklogd_node_t*)alloc->pptrs[ABCDK_MAP_VALUE];

    abcdk_map_destroy(&node->service_lists);
    abcdk_mutex_destroy(&node->service_mutex);
    abcdk_heap_free2((void**)&node->from);
}

void _abcdklogd_work(abcdklogd_t *ctx)
{
    abcdk_signal_t sig;
    abcdk_sockaddr_t addr = {0};
    const char *policy_file = NULL;
    abcdk_tree_t *policy_args = NULL;
    int sid = -1;
    int chk;
    
    ctx->policys = abcdk_heap_alloc(sizeof(abcdklogd_policy_t) * ABCDKLOGD_SERVICE_MAX);
    abcdk_mutex_init2(&ctx->node_mutex,0);
    abcdk_map_init(&ctx->node_lists,100);
    ctx->node_lists.construct_cb = _abcdklogd_node_construct;
    ctx->node_lists.destructor_cb = _abcdklogd_node_destructor;
    ctx->node_lists.opaque = ctx;

    ctx->daemon = (abcdk_option_exist(ctx->args, "--daemon") ? 1 : 0);
    ctx->listen = abcdk_option_get(ctx->args,"--listen",0,"127.0.0.1:65535");

    chk = abcdk_sockaddr_from_string(&addr, ctx->listen, 1);
    if (chk != 0)
        goto END;

    if(!abcdk_sockaddr_where(&addr,1))
        goto END;

    for (int i = 0; i < ABCDKLOGD_SERVICE_MAX; i++)
    {
        policy_file = abcdk_option_get(ctx->args, "--policy-conf", i, NULL);
        if (!policy_file)
            break;

        policy_args = abcdk_tree_alloc3(1);
        abcdk_getargs_file(policy_args,policy_file,'\n','#',NULL,"--");

        sid = abcdk_option_get_int(policy_args, "--service", i, -1);
        if (sid < 0 || sid >= ABCDKLOGD_SERVICE_MAX)
            continue;

        ctx->policys[sid].workspace = abcdk_option_get(policy_args, "--workspace", 0, "/tmp/abcdk.logd/");
        ctx->policys[sid].segment_max = abcdk_option_get_int(policy_args, "--segment-max", 0, 10);
        ctx->policys[sid].segment_size = abcdk_option_get_int(policy_args, "--segment-size", 0, 10);
    }

    /*未指定特殊策略的，初始化为全局策略。*/
    for (int i = 0; i < ABCDKLOGD_SERVICE_MAX; i++)
    {
        if (ctx->policys[i].workspace)
            continue;

        ctx->policys[i].workspace = abcdk_option_get(ctx->args, "--workspace", 0, "/tmp/abcdk.logd/");
        ctx->policys[i].segment_max = abcdk_option_get_int(ctx->args, "--segment-max", 0, 10);
        ctx->policys[i].segment_size = abcdk_option_get_int(ctx->args, "--segment-size", 0, 10);
    }

    ctx->comm = abcdk_comm_start(1);
    ctx->listen_easy = abcdk_comm_easy_alloc(ctx->comm);
    abcdk_comm_easy_set_userdata(ctx->listen_easy,ctx);
    chk = abcdk_comm_easy_listen(ctx->listen_easy,NULL,&addr,_abcdklogd_node_request);
    if(chk != 0)
        goto END;

    /*填充信号及回调函数。*/
    sig.opaque = NULL;
    sig.signal_cb = _abcdklogd_signal_cb;
    sigfillset(&sig.signals);
    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    /*等待退出信号。*/
    abcdk_sigwaitinfo(&sig,-1);
    //sleep(30000);
END:
    abcdk_comm_stop(&ctx->comm);
    abcdk_comm_easy_unref(&ctx->listen_easy);
    abcdk_heap_free(ctx->policys);
    abcdk_map_destroy(&ctx->node_lists);
    abcdk_mutex_destroy(&ctx->node_mutex);
}

int abcdk_tool_logd(abcdk_tree_t *args)
{
    abcdklogd_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
        _abcdklogd_print_usage(ctx.args);
    else
        _abcdklogd_work(&ctx);

    return ctx.errcode;
}