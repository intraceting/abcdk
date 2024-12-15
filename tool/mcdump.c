/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "entry.h"

typedef struct _abcdk_mcdump
{
    int errcode;
    abcdk_option_t *args;

    const char *outfile;

}abcdk_mcdump_t;

void _abcdk_mcdump_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t获取硬件散列值。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--output < FILE >\n");
    fprintf(stderr, "\t\t输出到文件(包括路径)。默认：终端\n");

    fprintf(stderr, "\n\t--stuff < STUFF >\n");
    fprintf(stderr, "\t\t填充物。\n");

    fprintf(stderr, "\n\t--exclude-mac\n");
    fprintf(stderr, "\t\t不包括网卡地址(仅物理网卡)。\n");

    fprintf(stderr, "\n\t--exclude-scsi\n");
    fprintf(stderr, "\t\t不包括SCSI设备。\n");

    fprintf(stderr, "\n\t--exclude-mmc\n");
    fprintf(stderr, "\t\t不包括MMC设备。\n");
}

void _abcdk_mcdump_work(abcdk_mcdump_t *ctx)
{
    const char *out = abcdk_option_get(ctx->args,"--output",0,NULL);
    const char *stuff = abcdk_option_get(ctx->args,"--stuff",0,NULL);
    int exclude_mac = abcdk_option_exist(ctx->args,"--exclude-mac");
    int exclude_scsi = abcdk_option_exist(ctx->args,"--exclude-scsi");
    int exclude_mmc = abcdk_option_exist(ctx->args,"--exclude-mmc");

    uint32_t flag = (0xf & ~ABCDK_DMI_HASH_USE_STUFF);
    uint8_t uuid[16] = {0};
    char str[33] = {0};

    if(exclude_mac)
        flag &= ~ABCDK_DMI_HASH_USE_DEVICE_MAC;

    if(exclude_scsi)
        flag &= ~ABCDK_DMI_HASH_USE_DEVICE_SCSI;

    if(exclude_mmc)
        flag &= ~ABCDK_DMI_HASH_USE_DEVICE_MMC;

    if(stuff && *stuff)
        flag |= ABCDK_DMI_HASH_USE_STUFF;

    if(flag == 0)
    {
        fprintf(stderr,"至少保留一项。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
    }

    if(out && *out)
    {
        if(abcdk_reopen(STDOUT_FILENO,out,1,0,1)<0)
        {
            fprintf(stderr, "'%s' %s.\n",out, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno,final);
        }
    }

    abcdk_dmi_hash(uuid,flag,stuff);

    abcdk_bin2hex(str,uuid,16,0);

    fprintf(stdout,"%s\n",str);

final:

    return;
}

int abcdk_tool_mcdump(abcdk_option_t *args)
{
    abcdk_mcdump_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_mcdump_print_usage(ctx.args);
    }
    else
    {
        _abcdk_mcdump_work(&ctx);
    }

    return ctx.errcode;
}