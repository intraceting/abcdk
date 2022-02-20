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
#include "util/getargs.h"
#include "util/scsi.h"
#include "util/iconv.h"
#include "util/unicode.h"
#include "entry.h"

#ifdef HAVE_ARCHIVE
#include <archive.h>
#include <archive_entry.h>
#endif

#if defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)

typedef struct _abcdkarchive_ctx
{
    int errcode;
    abcdk_tree_t *args;

    int cmd;

    struct
    {
        int src_bksize;
        int src_num;
        const char *src[256];
        const char *dst;

        int flt;
        int fmt;
        const char *opt;
        const char *pwd;

        uid_t uid;
        gid_t gid;
        uid_t oid;

        const char *ctime;
        const char *mtime;

        struct archive *src_fd;
        struct archive_entry *src_entry;
        int dst_fd;
    } r;

    struct
    {
    } w;

} abcdkarchive_ctx;

/** 常量。*/
enum _abcdkarchive_constant
{
    /** 回迁。*/
    ABCDKARCHIVE_READ = 1,
#define ABCDKARCHIVE_READ ABCDKARCHIVE_READ

    /** 归档。*/
    ABCDKARCHIVE_WRITE = 2
#define ABCDKARCHIVE_WRITE ABCDKARCHIVE_WRITE
};

static struct _abcdkarchive_filter_dict
{
    uint16_t code;
    uint32_t filter;
    const char *name;
} abcdkarchive_filter_dict[] = {
    {1, ARCHIVE_FILTER_NONE, "NONE"},
    {2, ARCHIVE_FILTER_GZIP, "GZIP"},
    {3, ARCHIVE_FILTER_BZIP2, "BZIP2"},
    {4, ARCHIVE_FILTER_COMPRESS, "COMPRESS"},
    {5, ARCHIVE_FILTER_PROGRAM, "PROGRAM"},
    {6, ARCHIVE_FILTER_XZ, "XZ"},
    {7, ARCHIVE_FILTER_UU, "UU"},
    {8, ARCHIVE_FILTER_RPM, "RPM"},
    {9, ARCHIVE_FILTER_LZIP, "LZIP"},
    {10, ARCHIVE_FILTER_LRZIP, "LRZIP"},
    {11, ARCHIVE_FILTER_LZOP, "LZOP"},
    {12, ARCHIVE_FILTER_GRZIP, "GRZIP"},
    {13, ARCHIVE_FILTER_LZ4, "LZ4"},
    {14, ARCHIVE_FILTER_ZSTD, "ZSTD"},
};

static struct _abcdkarchive_format_dict
{
    uint16_t code;
    uint32_t format;
    const char *name;
} abcdkarchive_format_dict[] = {
    {1, ARCHIVE_FORMAT_CPIO, "CPIO"},
    {2, ARCHIVE_FORMAT_CPIO_POSIX, "CPIO_POSIX"},
    {3, ARCHIVE_FORMAT_CPIO_BIN_LE, "CPIO_BIN_LE"},
    {4, ARCHIVE_FORMAT_CPIO_BIN_BE, "CPIO_BIN_BE"},
    {5, ARCHIVE_FORMAT_CPIO_SVR4_CRC, "CPIO_SVR4_CRC"},
    {6, ARCHIVE_FORMAT_CPIO_AFIO_LARGE, "CPIO_AFIO_LARGE"},
    {7, ARCHIVE_FORMAT_SHAR, "SHAR"},
    {8, ARCHIVE_FORMAT_SHAR_BASE, "SHAR_BASE"},
    {9, ARCHIVE_FORMAT_SHAR_DUMP, "SHAR_DUMP"},
    {10, ARCHIVE_FORMAT_TAR, "TAR"},
    {11, ARCHIVE_FORMAT_TAR_USTAR, "TAR_USTAR"},
    {12, ARCHIVE_FORMAT_TAR_PAX_INTERCHANGE, "TAR_PAX_INTERCHANGE"},
    {13, ARCHIVE_FORMAT_TAR_PAX_RESTRICTED, "TAR_PAX_RESTRICTED"},
    {14, ARCHIVE_FORMAT_TAR_GNUTAR, "TAR_GNUTAR"},
    {15, ARCHIVE_FORMAT_ISO9660, "ISO9660"},
    {16, ARCHIVE_FORMAT_ISO9660_ROCKRIDGE, "ISO9660_ROCKRIDGE"},
    {17, ARCHIVE_FORMAT_ZIP, "ZIP"},
    {18, ARCHIVE_FORMAT_EMPTY, "EMPTY"},
    {19, ARCHIVE_FORMAT_AR, "AR"},
    {20, ARCHIVE_FORMAT_AR_GNU, "AR_GNU"},
    {21, ARCHIVE_FORMAT_AR_BSD, "AR_BSD"},
    {22, ARCHIVE_FORMAT_MTREE, "MTREE"},
    {23, ARCHIVE_FORMAT_RAW, "RAW"},
    {24, ARCHIVE_FORMAT_XAR, "XAR"},
    {25, ARCHIVE_FORMAT_LHA, "LHA"},
    {26, ARCHIVE_FORMAT_CAB, "CAB"},
    {27, ARCHIVE_FORMAT_RAR, "RAR"},
    {28, ARCHIVE_FORMAT_7ZIP, "7ZIP"},
    {29, ARCHIVE_FORMAT_WARC, "WARC"}
};

int abcdkarchive_find_filter(int code)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_filter_dict); i++)
    {
        if(abcdkarchive_filter_dict[i].code == code)
            return abcdkarchive_filter_dict[i].filter;
    }

    return -1;
}

int abcdkarchive_find_format(int code)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_format_dict); i++)
    {
        if(abcdkarchive_format_dict[i].code == code)
            return abcdkarchive_format_dict[i].format;
    }

    return 0;
}

void _abcdkarchive_print_usage(abcdk_tree_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的文件归档工具。\n");
    fprintf(stderr, "\n\t%s\n", archive_version_details());

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--cmd \n");
    fprintf(stderr, "\t\t操作码。默认：%d\n", ABCDKARCHIVE_READ);

    fprintf(stderr, "\t\t%d：回迁。\n", ABCDKARCHIVE_READ);
    fprintf(stderr, "\t\t%d：归档。\n", ABCDKARCHIVE_WRITE);

    fprintf(stderr, "\n\t--filter < NUMBER >\n");
    fprintf(stderr, "\t\t过滤器。默认：自动\n");

    fprintf(stderr, "\n");
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_filter_dict); i++)
    {
        fprintf(stderr, "\t\t%hu: %-16s", abcdkarchive_filter_dict[i].code, abcdkarchive_filter_dict[i].name);
        if ((i + 1) % 4 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "\n\t--format < NUMBER >\n");
    fprintf(stderr, "\t\t格式。默认：自动\n");

    fprintf(stderr, "\n");
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_format_dict); i++)
    {
        fprintf(stderr, "\t\t%hu: %-16s ", abcdkarchive_format_dict[i].code, abcdkarchive_format_dict[i].name);
        if ((i + 1) % 4 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "\n\t--passphrase < STRING >\n");
    fprintf(stderr, "\t\t密码。默认：无\n");

    fprintf(stderr, "\n\t--option < STRING >\n");
    fprintf(stderr, "\t\t选项。见：man archive_write_set_options 或 man archive_read_set_options\n");
}

void _abcdkarchive_read(abcdkarchive_ctx *ctx)
{
    int chk;

    ctx->r.src_bksize = abcdk_option_get_int(ctx->args, "--filter", 0, 0);
    ctx->r.src_num = abcdk_option_count(ctx->args, "--src");
    ctx->r.src_num = ABCDK_CLAMP(ctx->r.src_num, 1, 255);
    for (int i = 0; i < ctx->r.src_num; i++)
        ctx->r.src[i] = abcdk_option_get(ctx->args, "--src", i, NULL);
    ctx->r.dst = abcdk_option_get(ctx->args, "--dst", 0, NULL);
    ctx->r.flt = abcdk_option_get_int(ctx->args, "--filter", 0, -1);
    ctx->r.fmt = abcdk_option_get_int(ctx->args, "--format", 0, -1);
    ctx->r.pwd = abcdk_option_get(ctx->args, "--passphrase", 0, NULL);
    ctx->r.opt = abcdk_option_get(ctx->args, "--option", 0, NULL);
    ctx->r.uid = 0;
    ctx->r.gid = 0;
    ctx->r.oid = 0;
    ctx->r.ctime = NULL;
    ctx->r.mtime = NULL;
    ctx->r.src_fd = NULL;
    ctx->r.dst_fd = -1;

    ctx->r.src_fd = archive_read_new();
    if (!ctx->r.src_fd)
    {
        syslog(LOG_ERR, "%s。", strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);
    }

    if (ctx->r.flt > 0)
    {
        chk = archive_read_append_filter(ctx->r.src_fd, abcdkarchive_find_filter(ctx->r.flt));
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }
    else
    {
        chk = archive_read_support_filter_all(ctx->r.src_fd);
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }

    if (ctx->r.fmt > 0)
    {
        chk = archive_read_set_format(ctx->r.src_fd, abcdkarchive_find_format(ctx->r.fmt));
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }
    else
    {
        chk = archive_read_support_format_all(ctx->r.src_fd);
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }

    if (ctx->r.opt)
    {
        chk = archive_read_set_options(ctx->r.src_fd, ctx->r.opt);
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }

    if (ctx->r.pwd)
    {
        chk = archive_read_add_passphrase(ctx->r.src_fd, ctx->r.pwd);
        if (chk != ARCHIVE_OK)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
    }

    chk = archive_read_open_filenames(ctx->r.src_fd, ctx->r.src, ctx->r.src_bksize);
    if (chk != ARCHIVE_OK)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);

    while (archive_read_next_header(ctx->r.src_fd, &ctx->r.src_entry) == ARCHIVE_OK)
    {
        char *src = archive_entry_pathname(ctx->r.src_entry);
        int src_len = strlen(src);

        ssize_t m =  abcdk_verify_utf8(src,src_len);
        if(m!=src_len)
        {
            m =  abcdk_verify_gbk(src,src_len);
            char buf[100]={0};
            if(m == src_len)
            {
                ssize_t n =  abcdk_iconv2("GBK","UTF-8",src,src_len,buf,100,NULL);
            }
            printf("'%s'\n", buf);
        }
        else
        {
            printf("'%s'\n", src);
        }

        
         
        

        // char buf[100];
        // ssize_t s = archive_read_data(ctx->r.src_fd,buf,100);

     //   chk = archive_read_data_into_fd(ctx->r.src_fd, STDERR_FILENO);

        archive_read_data_skip(ctx->r.src_fd);
    }

    /*No error.*/
    goto final;

final_error:

    if (ctx->r.src_fd)
        syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));

final:

    abcdk_closep(&ctx->r.dst_fd);

    if (ctx->r.src_fd)
    {
        archive_read_close(ctx->r.src_fd);
        archive_read_free(ctx->r.src_fd);
        ctx->r.src_fd = NULL;
    }
}

void _abcdkarchive_work(abcdkarchive_ctx *ctx)
{
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDKARCHIVE_READ);

    if (ctx->cmd == ABCDKARCHIVE_READ)
        _abcdkarchive_read(ctx);
    else
    {
        syslog(LOG_INFO, "尚未支持。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

final:

    return;
}

#endif // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)

int abcdk_tool_archive(abcdk_tree_t *args)
{
#if defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)
    abcdkarchive_ctx ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkarchive_print_usage(ctx.args);
    }
    else
    {
        _abcdkarchive_work(&ctx);
    }

    return ctx.errcode;

#else // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)

    syslog(LOG_INFO, "当前构建版本未包含此工具。\n");
    return EPERM;

#endif // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)
}