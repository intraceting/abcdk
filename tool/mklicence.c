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
#include "abcdk-auth/auth.h"

typedef struct _abcdkmkl_ctx
{
    int errcode;

    abcdk_tree_t *args;

    const char *save;
    uint32_t days;
    uint32_t delay;

    abcdk_tree_t *auth;
    abcdk_allocator_t *dec_auth;
    abcdk_allocator_t *enc_auth;
    int out_fd;
    
}abcdkmkl_ctx;

void _abcdkmkl_print_usage(abcdk_tree_t *args, int only_version)
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

    fprintf(stderr, "\n\t简单的许可证制作工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--save < FILE >\n");
    fprintf(stderr, "\t\t许可证文件(包括路径)。\n");

    fprintf(stderr, "\n\t--valid-days < NUMBER >\n");
    fprintf(stderr, "\t\t有效时长(天)。默认：30\n");

    fprintf(stderr, "\n\t--time-on-delay < NUMBER >\n");
    fprintf(stderr, "\t\t生效延迟(天)。默认：0\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdkmkl_work(abcdkmkl_ctx *ctx)
{
    ctx->out_fd = -1;
    int chk;

    ctx->save = abcdk_option_get(ctx->args, "--save", 0, NULL);
    ctx->days = abcdk_option_get_int(ctx->args,"--valid-days",0,30);
    ctx->delay = abcdk_option_get_int(ctx->args,"--time-on-delay",0,0);

    if (!ctx->save || !*ctx->save)
    {
        syslog(LOG_ERR, "'--save FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    ctx->out_fd = abcdk_open(ctx->save, 1, 0, 1);
    if (ctx->out_fd < 0)
    {
        syslog(LOG_ERR, "'%s' %s。", ctx->save,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    ctx->auth = abcdk_tree_alloc3(1);
    if(!ctx->auth)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    if (ctx->days > 365 * 99)
        syslog(LOG_WARNING, ABCDK_ANSI_COLOR_RED "注意，有效期限超过99年。" ABCDK_ANSI_COLOR_RESET);

    if (ctx->delay > 30)
        syslog(LOG_WARNING, ABCDK_ANSI_COLOR_RED "注意，生效延迟超过30天。" ABCDK_ANSI_COLOR_RESET);

    abcdk_auth_add_valid_period2(ctx->auth,ctx->days,ctx->delay);
    abcdk_auth_add_salt(ctx->auth);
    abcdk_auth_add_salt(ctx->auth);

    ctx->dec_auth = abcdk_auth_serialize(ctx->auth);
    ctx->enc_auth = abcdk_auth_encrypt(ctx->dec_auth,ABCDK_AUTH_DEFAULT_KEY);

    chk = abcdk_auth_save(ctx->out_fd,ctx->enc_auth->pptrs[0],ctx->enc_auth->sizes[0],ABCDK_AUTH_DEFAULT_MAGIC);
    if(chk != 0)
    {
        syslog(LOG_ERR, "'%s' %s。", ctx->save,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

final:

    abcdk_allocator_unref(&ctx->enc_auth);
    abcdk_allocator_unref(&ctx->dec_auth);
    abcdk_tree_free(&ctx->auth);
    abcdk_closep(&ctx->out_fd);

}

int main(int argc, char **argv)
{
    abcdkmkl_ctx ctx = {0};

    /*中文，UTF-8*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        goto final;

    abcdk_getargs(ctx.args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkmkl_print_usage(ctx.args, 0);
    }
    else if (abcdk_option_exist(ctx.args, "--version"))
    {
        _abcdkmkl_print_usage(ctx.args, 1);
    }
    else
    {
        _abcdkmkl_work(&ctx);
    }

final:
    
    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}