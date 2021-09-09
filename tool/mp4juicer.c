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
#include "abcdk-mp4/demuxer.h"

typedef struct _abcdkm4j_ctx
{
    abcdk_tree_t *args;

    int err;
    const char *file;
    const char *save;

    int in_fd;
    int out_fd[16];

    abcdk_tree_t *doc;
    abcdk_tree_t *moov_p;
    abcdk_tree_t *mvex_p;
    abcdk_tree_t *trak_p;
    abcdk_tree_t *hdlr_p;

}abcdkm4j_ctx;

void _abcdkm4j_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d-%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --file < FILE > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tMP4视音频提取器，仅支持H264和ACC。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\t文件(包括路径)。\n");

    fprintf(stderr, "\n\t--save < PATH >\n");
    fprintf(stderr, "\t\t保存路径。默认：./\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdkm4j_fmp4_dump(abcdkm4j_ctx *ctx)
{
    
}

void _abcdkm4j_dump_video(abcdkm4j_ctx *ctx)
{
    
}

void _abcdkm4j_dump_audio(abcdkm4j_ctx *ctx)
{
    
}

void _abcdkm4j_dump(abcdkm4j_ctx *ctx)
{
    abcdk_mp4_atom_t *hdlr = NULL;

    /*一定要初始化，否则关闭时可能会出现异想不到的问题。*/
    for (int i = 0; i <= ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
        ctx->out_fd[i] = -1;

    ctx->moov_p = abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_MOOV,1,1);
    if(!ctx->moov_p)
        ABCDK_ERRNO_AND_GOTO1(ESPIPE, final);

    ctx->mvex_p = abcdk_mp4_find2(ctx->moov_p,ABCDK_MP4_ATOM_TYPE_MVEX,1,1);

    if(ctx->mvex_p)
    {
        _abcdkm4j_fmp4_dump(ctx);
        goto final;
    }

    for (int i = 0; i < ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
    {
        ctx->trak_p = abcdk_mp4_find2(ctx->moov_p, ABCDK_MP4_ATOM_TYPE_TRAK, i + 1, 0);
        if (!ctx->trak_p)
            ABCDK_ERRNO_AND_GOTO1(0, final);

        ctx->hdlr_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_HDLR, 1, 1);
        if(!ctx->hdlr_p)
            continue;

        hdlr = (abcdk_mp4_atom_t*)ctx->hdlr_p->alloc->pptrs[0];

        if(hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('v', 'i', 'd', 'e'))
            _abcdkm4j_dump_video(ctx);
        else if(hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('s', 'o', 'u', 'n'))
            _abcdkm4j_dump_audio(ctx);
        
    }

final:

    ctx->err = errno;
    for (int i = 0; i <= ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
        abcdk_closep(&ctx->out_fd[i]);
    errno = ctx->err;
}

void _abcdkm4j_work(abcdkm4j_ctx *ctx)
{
    ctx->in_fd = -1;

    ctx->file = abcdk_option_get(ctx->args, "--file", 0, NULL);
    ctx->save = abcdk_option_get(ctx->args, "--save", 0, "./");

    /*Clear errno.*/
    errno = 0;

    if (!ctx->file || !*ctx->file)
    {
        syslog(LOG_ERR, "'--file FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(ctx->file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->file, strerror(errno));
        goto final;
    }

    if (!ctx->save || !*ctx->save)
    {
        syslog(LOG_ERR, "'--save PATH' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (access(ctx->save, W_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->save, strerror(errno));
        goto final;
    }

    ctx->in_fd = abcdk_open(ctx->file, 0, 0, 0);
    if (ctx->in_fd < 0)
        goto final;

    ctx->doc = abcdk_mp4_read_probe2(ctx->in_fd, 0, -1UL, 0);
    if (!ctx->doc)
        goto final;

    if(!abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_FTYP,1,1))
    {
        syslog(LOG_WARNING, "'%s' 可能不是MP4文件，或尚未支持此格式。", ctx->file);
        ABCDK_ERRNO_AND_GOTO1(EPERM, final);
    }

    _abcdkm4j_dump(ctx);

final:

    ctx->err = errno;
    abcdk_closep(&ctx->in_fd);
    abcdk_tree_free(&ctx->doc);
    errno = ctx->err;

}

int main(int argc, char **argv)
{
    abcdkm4j_ctx ctx = {0};

    /*中文，UTF-8*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        goto final;

    abcdk_getargs(ctx.args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkm4j_print_usage(ctx.args, 0);
    }
    else if (abcdk_option_exist(ctx.args, "--version"))
    {
        _abcdkm4j_print_usage(ctx.args, 1);
    }
    else
    {
        _abcdkm4j_work(&ctx);
    }

final:

    abcdk_tree_free(&ctx.args);

    return errno;
}