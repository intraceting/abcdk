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
#include "util/charset.h"
#include "util/cap.h"
#include "util/dirent.h"
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

    int blk;
    int flt;
    int fmt;
    const char *opt;
    const char *pwd;

    struct
    {
        /*getuid(), not geteuid().*/
        uid_t uid;

        /* 0 no power, !0 have power.*/
        int chown_power;

        int src_num;
        const char *src[256];
        const char *dst;
        int justlist;

        struct archive *src_fd;
        struct archive_entry *src_entry;
        int dst_fd;
    } r;

    struct
    {
        int src_num;
        const char *src[256];
        const char *dst;

        int src_fd;
        const char src_file[PATH_MAX];
        const char src_link[PATH_MAX];
        struct stat src_stat;
        abcdk_tree_t *src_dir;
        struct archive *dst_fd;
        struct archive_entry *dst_entry;
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
#if ARCHIVE_VERSION_NUMBER >= 3003003
    {13, ARCHIVE_FILTER_LZ4, "LZ4"},
    {14, ARCHIVE_FILTER_ZSTD, "ZSTD"},
#endif // ARCHIVE_VERSION_NUMBER >= 3003003
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
#if ARCHIVE_VERSION_NUMBER >= 3003003
    {29, ARCHIVE_FORMAT_WARC, "WARC"}
#endif // ARCHIVE_VERSION_NUMBER >= 3003003
};

int abcdkarchive_find_filter(int code)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_filter_dict); i++)
    {
        if (abcdkarchive_filter_dict[i].code == code)
            return abcdkarchive_filter_dict[i].filter;
    }

    return -1;
}

int abcdkarchive_find_format(int code)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_format_dict); i++)
    {
        if (abcdkarchive_format_dict[i].code == code)
            return abcdkarchive_format_dict[i].format;
    }

    return 0;
}

void _abcdkarchive_print_usage(abcdk_tree_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的文件归档工具。\n");
    fprintf(stderr, "\n\t%s\n",
#if ARCHIVE_VERSION_NUMBER >= 3003003
            archive_version_details());
#else  // ARCHIVE_VERSION_NUMBER >= 3003003
            archive_version_string());
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

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

#if ARCHIVE_VERSION_NUMBER >= 3003003
    fprintf(stderr, "\n\t--passphrase < STRING >\n");
    fprintf(stderr, "\t\t密码。默认：无\n");
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

    fprintf(stderr, "\n\t--option < STRING >\n");
    fprintf(stderr, "\t\t选项。见：man archive_write_set_options 或 man archive_read_set_options\n");

    fprintf(stderr, "\n\t--blksize < SIZE >\n");
    fprintf(stderr, "\t\t每次读写块大小(字节)。默认：自动\n");

    fprintf(stderr, "\n\t--filename < FILENAME [ FILENAME-part1 FILENAME-part2 ... ] >\n");
    fprintf(stderr, "\t\t档案文件名(包括路径)。\n");

    fprintf(stderr, "\n\t--extract-to < PATH >\n");
    fprintf(stderr, "\t\t回迁文件保存位置。默认：./\n");

    fprintf(stderr, "\n\t--extract-just-list\n");
    fprintf(stderr, "\t\t仅打印待回迁文件列表。\n");
}

char *_abcdkarchive_name2local(const char *src, char dst[PATH_MAX])
{
    size_t slen;
    size_t slen_vf;

    /*猜测可能的长度。*/
    slen = strlen(src);

    slen_vf = abcdk_verify_utf8(src, PATH_MAX);
    if (slen_vf != slen)
    {
        slen_vf = abcdk_verify_gbk(src, PATH_MAX);
        if (slen_vf == slen)
        {
            abcdk_iconv2("cp936", "UTF-8", src, slen_vf, dst, PATH_MAX, NULL);
        }
        else
        {
            slen_vf = abcdk_verify_ucs2(src, PATH_MAX, 0);
            abcdk_iconv2("UCS-2", "UTF-8", src, slen_vf, dst, PATH_MAX, NULL);
        }
    }
    else
    {
        strncpy(dst, src, slen_vf);
    }

    return dst;
}

int _abcdkarchive_read_one(abcdkarchive_ctx *ctx)
{
    const char *name = NULL;
    char name_cp[PATH_MAX] = {0};
    char pathfile[PATH_MAX] = {0};
    struct stat file_stat = {0};
    size_t lkname_len = 0;
    const char *lkname = NULL;
    char lkname_cp[PATH_MAX] = {0};
    ssize_t lkname_cp_len = 0;
    int chk = 0;

    name = archive_entry_pathname(ctx->r.src_entry);
    _abcdkarchive_name2local(name, name_cp);

    syslog(LOG_INFO, "%s\n", name_cp);

    if (ctx->r.justlist)
    {
        archive_read_data_skip(ctx->r.src_fd);
    }
    else
    {
        file_stat = *archive_entry_stat(ctx->r.src_entry);

        abcdk_dirdir(pathfile, ctx->r.dst);
        abcdk_dirdir(pathfile, name_cp);

        if (S_ISLNK(file_stat.st_mode))
        {
            if (access(pathfile, F_OK) == 0)
            {
                syslog(LOG_WARNING, "%s -> 同名软链接已经存在，跳过。 \n", name_cp);
                goto final;
            }

            lkname = archive_entry_symlink(ctx->r.src_entry);
            _abcdkarchive_name2local(lkname, lkname_cp);

            abcdk_mkdir(pathfile, 0700);
            chk = symlink(lkname, pathfile);
            if (chk != 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            /*不需要恢复软链接属性。*/
            goto final;
        }
        else if (S_ISDIR(file_stat.st_mode))
        {
            if (access(pathfile, F_OK) == 0)
                goto final;

            abcdk_dirdir(pathfile, "/");
            pathfile[strlen(pathfile) - 1] = '\0';

            abcdk_mkdir(pathfile, 0700);
            chk = mkdir(pathfile, 0700);
            if (chk != 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            ctx->r.dst_fd = open(pathfile, __O_DIRECTORY | __O_CLOEXEC, 0); //打开目录。
            if (ctx->r.dst_fd < 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);
        }
        else if (S_ISREG((file_stat.st_mode)))
        {
            if (access(pathfile, F_OK) == 0)
            {
                syslog(LOG_WARNING, "%s -> 同名文件已经存在，跳过。 \n", name_cp);
                goto final;
            }

            abcdk_mkdir(pathfile, 0700);
            ctx->r.dst_fd = abcdk_open(pathfile, 1, 0, 1); //打开文件。
            if (ctx->r.dst_fd < 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            chk = archive_read_data_into_fd(ctx->r.src_fd, ctx->r.dst_fd);
            if (chk != ARCHIVE_OK)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final_error);
        }
        else
        {
            syslog(LOG_WARNING, "%s -> 不支持的类型，跳过。 \n", name_cp);
            goto final;
        }

        /*恢复文件(目录)属性的时间。*/
        chk = abcdk_futimens(ctx->r.dst_fd, &file_stat.st_atim, &file_stat.st_mtim);
        if (chk != 0)
            syslog(LOG_WARNING, "%s -> 未能恢复文件时间，忽略。\n", name_cp);
#if 0
        /*恢复文件(目录)属性的权限。*/
        if(file_stat.st_mode & ACCESSPERMS)
        {
            chk = fchmod(ctx->r.dst_fd, file_stat.st_mode & ACCESSPERMS);
            if (chk != 0)
                syslog(LOG_WARNING, "%s -> 未能恢复文件权限，忽略。\n", name_cp);
        }

        /*仅所有者或特权者才能改变文件(目录)的所有者和所属组。*/
        if (ctx->r.uid == file_stat.st_uid || ctx->r.chown_power || ctx->r.uid == 0)
        {
            chk = fchown(ctx->r.dst_fd, file_stat.st_uid, file_stat.st_gid);
            if (chk != 0)
                syslog(LOG_WARNING, "%s -> 未能恢复文件的用户和组，忽略。\n", name_cp);
        }
#endif
        /*忽略恢复文件(目录)属性过程中发生的错误。*/
        chk = 0;
    }

    /*No error.*/
    goto final;

final_error:

    /*error.*/
    chk = -1;

final:

    abcdk_closep(&ctx->r.dst_fd);

    return chk;
}

void _abcdkarchive_read(abcdkarchive_ctx *ctx)
{
    int chk;

    ctx->flt = abcdk_option_get_int(ctx->args, "--filter", 0, -1,0);
    ctx->fmt = abcdk_option_get_int(ctx->args, "--format", 0, -1,0);
    ctx->pwd = abcdk_option_get(ctx->args, "--passphrase", 0, NULL);
    ctx->opt = abcdk_option_get(ctx->args, "--option", 0, NULL);
    ctx->blk = abcdk_option_get_int(ctx->args, "--blksize", 0, 0,0);

    ctx->r.uid = getuid();
#ifdef _SYS_CAPABILITY_H
    ctx->r.chown_power = abcdk_cap_get_pid(getpid(), CAP_CHOWN, CAP_EFFECTIVE);
#endif //_SYS_CAPABILITY_H

    ctx->r.src_num = abcdk_option_count(ctx->args, "--filename");
    ctx->r.src_num = ABCDK_CLAMP(ctx->r.src_num, 1, 255);
    for (int i = 0; i < ctx->r.src_num; i++)
        ctx->r.src[i] = abcdk_option_get(ctx->args, "--filename", i, NULL);
    ctx->r.dst = abcdk_option_get(ctx->args, "--extract-to", 0, "./");
    ctx->r.justlist = abcdk_option_exist(ctx->args, "--extract-just-list");

    ctx->r.src_fd = NULL;
    ctx->r.dst_fd = -1;

    if (ctx->r.src[0] == NULL || *ctx->r.src[0] == '\0')
    {
        syslog(LOG_ERR, "'--filename FILENAME [ FILENAME-part1 FILENAME-part2 ... ] ' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    for (int i = 0; i < ctx->r.src_num; i++)
    {
        if (access(ctx->r.src[0], R_OK) != 0)
        {
            syslog(LOG_ERR, "'%s' %s。", ctx->r.src[i], strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }
    }

    if (!ctx->r.justlist)
    {
        if (ctx->r.dst == NULL || *ctx->r.dst == '\0')
        {
            syslog(LOG_ERR, "'--extract PATH ' 不能省略，且不能为空。");
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
        }

        if (access(ctx->r.dst, W_OK) != 0)
        {
            syslog(LOG_ERR, "'%s' %s。", ctx->r.dst, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }
    }

    ctx->r.src_fd = archive_read_new();
    if (!ctx->r.src_fd)
    {
        syslog(LOG_ERR, "%s。", strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    if (ctx->flt > 0)
    {
        chk = archive_read_append_filter(ctx->r.src_fd, abcdkarchive_find_filter(ctx->flt));
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }
    else
    {
        chk = archive_read_support_filter_all(ctx->r.src_fd);
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }

    if (ctx->fmt > 0)
    {
        chk = archive_read_set_format(ctx->r.src_fd, abcdkarchive_find_format(ctx->fmt));
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }
    else
    {
        chk = archive_read_support_format_all(ctx->r.src_fd);
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }

    if (ctx->opt)
    {
        chk = archive_read_set_options(ctx->r.src_fd, ctx->opt);
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }

#if ARCHIVE_VERSION_NUMBER >= 3003003
    if (ctx->pwd)
    {
        chk = archive_read_add_passphrase(ctx->r.src_fd, ctx->pwd);
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }
    }
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

    chk = archive_read_open_filenames(ctx->r.src_fd, ctx->r.src, ctx->blk);
    if (chk != ARCHIVE_OK)
    {
        syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
    }

    while (1)
    {
        chk = archive_read_next_header(ctx->r.src_fd, &ctx->r.src_entry);
        if (chk == ARCHIVE_EOF)
            goto final;
        if (chk != ARCHIVE_OK)
        {
            syslog(LOG_ERR, "%s", archive_error_string(ctx->r.src_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->r.src_fd), final);
        }

        chk = _abcdkarchive_read_one(ctx);
        if (chk != 0)
            goto final;
    }

final:

    if (ctx->r.src_fd)
    {
        archive_read_free(ctx->r.src_fd);
        ctx->r.src_fd = NULL;
    }
}

void _abcdkarchive_work(abcdkarchive_ctx *ctx)
{
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDKARCHIVE_READ,0);

    if (ctx->cmd == ABCDKARCHIVE_READ)
    {
        _abcdkarchive_read(ctx);
    }
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