/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "entry.h"

#ifdef HAVE_ARCHIVE
#include <archive.h>
#include <archive_entry.h>
#endif

#ifdef HAVE_ARCHIVE

#if ARCHIVE_VERSION_NUMBER < 3000000
#define	ARCHIVE_FILTER_NONE ARCHIVE_COMPRESSION_NONE	
#define	ARCHIVE_FILTER_GZIP ARCHIVE_COMPRESSION_GZIP	
#define	ARCHIVE_FILTER_BZIP2 ARCHIVE_COMPRESSION_BZIP2	
#define	ARCHIVE_FILTER_COMPRESS ARCHIVE_COMPRESSION_COMPRESS	
#define	ARCHIVE_FILTER_PROGRAM ARCHIVE_COMPRESSION_PROGRAM	
#define	ARCHIVE_FILTER_LZMA ARCHIVE_COMPRESSION_LZMA	
#define	ARCHIVE_FILTER_XZ ARCHIVE_COMPRESSION_XZ		
#define	ARCHIVE_FILTER_UU ARCHIVE_COMPRESSION_UU		
#define	ARCHIVE_FILTER_RPM ARCHIVE_COMPRESSION_RPM		
#define	ARCHIVE_FILTER_LZIP ARCHIVE_COMPRESSION_LZIP	
#define	ARCHIVE_FILTER_LRZIP ARCHIVE_COMPRESSION_LRZIP	
#endif

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

typedef struct _abcdkarchive
{
    int errcode;
    abcdk_option_t *args;

    int cmd;

    int blk;
    int flt;
    int fmt;
    const char *opt;
    const char *pwd;

    const char *md_cst;
    int md_cst_w;

    const char *wksp;

    int volume_num;
    const char *volumes[256];

    int justlist;

    int file_num;
    const char *files[256];
    int save_fullpath;

    struct archive *arch_fd;
    struct archive_entry *arch_entry;

    int fd[256];
    size_t buf_size;
    void *buf;
    abcdk_reader_t *reader;

    /*
     * 属性列表(后进先出)，用于回迁后的属性恢复。
     * 
     * 路径长度，路径，属性。
    */
    abcdk_object_t *attr_list;
    off_t attr_list_pos;

    /*属性列表临时文件名。*/
    char attr_list_tmpname[NAME_MAX];

} abcdkarchive_t;

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
#if ARCHIVE_VERSION_NUMBER >= 3000000
    {9, ARCHIVE_FILTER_LZIP, "LZIP"},
    {10, ARCHIVE_FILTER_LRZIP, "LRZIP"},
    {11, ARCHIVE_FILTER_LZOP, "LZOP"},
    {12, ARCHIVE_FILTER_GRZIP, "GRZIP"},
#endif // ARCHIVE_VERSION_NUMBER >= 3000000
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
#if ARCHIVE_VERSION_NUMBER >= 3000000
    {6, ARCHIVE_FORMAT_CPIO_AFIO_LARGE, "CPIO_AFIO_LARGE"},
#endif // ARCHIVE_VERSION_NUMBER >= 3000000
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
#if ARCHIVE_VERSION_NUMBER >= 3000000
    {25, ARCHIVE_FORMAT_LHA, "LHA"},
    {26, ARCHIVE_FORMAT_CAB, "CAB"},
    {27, ARCHIVE_FORMAT_RAR, "RAR"},
    {28, ARCHIVE_FORMAT_7ZIP, "7ZIP"},
#endif // ARCHIVE_VERSION_NUMBER >= 3000000
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

    return -1;
}


void _abcdkarchive_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的文件回迁和归档工具。\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--cmd \n");
    fprintf(stderr, "\t\t操作码。默认：%d\n", ABCDKARCHIVE_READ);

    fprintf(stderr, "\n\t\t%d：回迁。\n", ABCDKARCHIVE_READ);
    fprintf(stderr, "\t\t%d：归档。\n", ABCDKARCHIVE_WRITE);

    fprintf(stderr, "\n\t--filter < NUMBER >\n");
    fprintf(stderr, "\t\t过滤器。\n");

    fprintf(stderr, "\n");
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_filter_dict); i++)
    {
        fprintf(stderr, "\t\t%hu: %-16s", abcdkarchive_filter_dict[i].code, abcdkarchive_filter_dict[i].name);
        if ((i + 1) % 4 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "\n\t--format < NUMBER >\n");
    fprintf(stderr, "\t\t格式。\n");

    fprintf(stderr, "\n");
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkarchive_format_dict); i++)
    {
        fprintf(stderr, "\t\t%hu: %-16s ", abcdkarchive_format_dict[i].code, abcdkarchive_format_dict[i].name);
        if ((i + 1) % 4 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");

#if ARCHIVE_VERSION_NUMBER >= 3003003
    fprintf(stderr, "\n\t--password < STRING >\n");
    fprintf(stderr, "\t\t密码。默认：无\n");
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

    fprintf(stderr, "\n\t--option < STRING >\n");
    fprintf(stderr, "\t\t附加选项。见：man archive_write_set_options 或 man archive_read_set_options\n");

    fprintf(stderr, "\n\t--block-size < SIZE >\n");
    fprintf(stderr, "\t\t每次读写块大小（字节）。默认：10240\n");

    fprintf(stderr, "\n\t--metadata-charset < CODE >\n");
    fprintf(stderr, "\t\t元数据编码。默认：UTF-8，其它参考iconv -l\n");

    fprintf(stderr, "\n\t--metadata-char-width\n");
    fprintf(stderr, "\t\t元数据编码字符宽度。默认：1\n");

    fprintf(stderr, "\n\t\t1：多字节(变长)。\n");
    fprintf(stderr, "\t\t2：二字节(定长)。\n");
    fprintf(stderr, "\t\t4：四字节(定长)。\n");

    fprintf(stderr, "\n\t--work-space < PATH >\n");
    fprintf(stderr, "\t\t工作路径。默认：./\n");

    fprintf(stderr, "\n回迁选项:\n");
#if ARCHIVE_VERSION_NUMBER >= 3000000
    fprintf(stderr, "\n\t--volume < NAME [ NAME-part2 NAME-part3 ... ] >\n");
    fprintf(stderr, "\t\t卷名和分卷名（包括路径）。注：最大支持254个分卷\n");
#else 
    fprintf(stderr, "\n\t--volume < NAME >\n");
    fprintf(stderr, "\t\t卷名和分卷名（包括路径）。\n");
#endif //ARCHIVE_VERSION_NUMBER >= 3000000

    fprintf(stderr, "\n\t--file-list < FILE|DIR [ FILE|DIR ... ] >\n");
    fprintf(stderr, "\t\t文件或目录。注：最大支持254个。\n");

    fprintf(stderr, "\n\t\t*：匹配多个连续字符。\n");
    fprintf(stderr, "\t\t?：匹配单个字符。\n");

    fprintf(stderr, "\n\t--just-list\n");
    fprintf(stderr, "\t\t仅打印文件列表。\n");

    fprintf(stderr, "\n归档选项:\n");

    fprintf(stderr, "\n\t--volume < NAME [ NAME-copy NAME-copy ... ] >\n");
    fprintf(stderr, "\t\t卷名和副本（包括路径）。注：最大支持254个副本\n");

    fprintf(stderr, "\n\t--file-list < FILE|DIR [ FILE|DIR ... ] >\n");
    fprintf(stderr, "\t\t文件或目录。注：最大支持254个。\n");

    fprintf(stderr, "\n\t--save-fullpath\n");
    fprintf(stderr, "\t\t保留完整路径。默认：不保留。\n");
}

void _abcdkarchive_read_push_attr(abcdkarchive_t *ctx,const char *name,struct stat *stat)
{
    int all_len;
    int name_len;
    char buf[PATH_MAX];

    if (!ctx->attr_list)
    {
        strncpy(ctx->attr_list_tmpname, "/tmp/XXXXXX", NAME_MAX);

        int fd = mkstemp(ctx->attr_list_tmpname);
        if (fd < 0)
            return;

        ctx->attr_list = abcdk_mmap_fd(fd, (1UL << 31) - 1, 1, 0);
        if(ctx->attr_list)
            ctx->attr_list_pos = ctx->attr_list->sizes[0];

        abcdk_closep(&fd);
    }

    if (!ctx->attr_list)
        return;

    name_len = strlen(name);
    all_len = 2 + name_len + sizeof(*stat);
    if (ctx->attr_list_pos < all_len)
        return;

    ctx->attr_list_pos -= (all_len);

    ABCDK_PTR2U16(ctx->attr_list->pptrs[0], ctx->attr_list_pos) = name_len;
    memcpy(ABCDK_PTR2I8PTR(ctx->attr_list->pptrs[0], ctx->attr_list_pos + 2), name, name_len);
    memcpy(ABCDK_PTR2I8PTR(ctx->attr_list->pptrs[0], ctx->attr_list_pos + 2 + name_len), stat, sizeof(*stat));
    
}

int _abcdkarchive_read_pop_attr(abcdkarchive_t *ctx, char *name, struct stat *stat)
{
    int name_len = 0;

    if (!ctx->attr_list)
        return -1;

    if (ctx->attr_list->sizes[0] <= ctx->attr_list_pos)
        return -1;

    name_len = ABCDK_PTR2U16(ctx->attr_list->pptrs[0], ctx->attr_list_pos);
    memcpy(name, ABCDK_PTR2I8PTR(ctx->attr_list->pptrs[0], ctx->attr_list_pos + 2), name_len);
    memcpy(stat, ABCDK_PTR2I8PTR(ctx->attr_list->pptrs[0], ctx->attr_list_pos + 2 + name_len), sizeof(*stat));

    ctx->attr_list_pos += (2 + name_len + sizeof(*stat));

    return 0;
}

int _abcdkarchive_read_one(abcdkarchive_t *ctx)
{
    const char *name = NULL;
    char name_cp[PATH_MAX] = {0};
    char *bname_p = NULL;
    char pathfile[PATH_MAX] = {0};
    struct stat file_stat = {0};
    size_t lkname_len = 0;
    const char *lkname = NULL;
    char lkname_cp[PATH_MAX] = {0};
    ssize_t lkname_cp_len = 0;
    int chk = 0;

    name = archive_entry_pathname(ctx->arch_entry);

    /*转成本地的字符集编码。*/
    abcdk_iconv2(ctx->md_cst, "UTF-8", name, abcdk_cslen(name,ctx->md_cst_w) * ctx->md_cst_w, name_cp, PATH_MAX, NULL);

#if 0
    fprintf(stderr, "%s\n", name_cp);
#endif 

    /*去掉冗余的路径信息。*/
    abcdk_abspath(name_cp,0);
    if (name_cp[0] == 0)
        return 0;

    /*如果不存在文件列表，则默认匹配成功。*/
    if (ctx->file_num > 0)
    {
        for (int i = 0; i < ctx->file_num; i++)
        {
            if (abcdk_fnmatch(name_cp,ctx->files[i], 1, 1) == 0)
                goto MATCH_OK;
        }

        return 0;
    }

MATCH_OK:

    fprintf(stderr, "%s\n", name_cp);

    if (ctx->justlist)
    {
        chk = archive_read_data_skip(ctx->arch_fd);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final_error);
        }
    }
    else
    {
        file_stat = *archive_entry_stat(ctx->arch_entry);

        abcdk_dirdir(pathfile, ctx->wksp);
        abcdk_dirdir(pathfile, name_cp);

        if (S_ISLNK(file_stat.st_mode))
        {
            if (access(pathfile, F_OK) == 0)
            {
                fprintf(stderr, "%s -> 同名文件已经存在，跳过。 \n", name_cp);
                ABCDK_ERRNO_AND_GOTO1(chk = 0, final);
            }

            lkname = archive_entry_symlink(ctx->arch_entry);

            /*转成本地的字符集编码。*/
            abcdk_iconv2(ctx->md_cst, "UTF-8", lkname, abcdk_cslen(lkname,ctx->md_cst_w) * ctx->md_cst_w, lkname_cp, PATH_MAX, NULL);

            abcdk_mkdir(pathfile, 0700);
            chk = symlink(lkname, pathfile);
            if (chk != 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            /*不需要恢复软链接属性。*/
            ABCDK_ERRNO_AND_GOTO1(chk = 0, final);
        }
        else if (S_ISDIR(file_stat.st_mode))
        {
            if (access(pathfile, F_OK) == 0)
                ABCDK_ERRNO_AND_GOTO1(chk = 0, final);

            abcdk_dirdir(pathfile, "/");
            pathfile[strlen(pathfile) - 1] = '\0';

            abcdk_mkdir(pathfile, 0700);
            chk = mkdir(pathfile, 0700);
            if (chk != 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            ctx->fd[0] = open(pathfile, O_DIRECTORY | O_CLOEXEC, 0); //打开目录。
            if (ctx->fd[0] < 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);
        }
        else if (S_ISREG((file_stat.st_mode)))
        {
            if (access(pathfile, F_OK) == 0)
            {
                fprintf(stderr, "%s -> 同名文件已经存在，跳过。 \n", name_cp);
                ABCDK_ERRNO_AND_GOTO1(chk = 0, final);
            }

            abcdk_mkdir(pathfile, 0700);
            ctx->fd[0] = abcdk_open(pathfile, 1, 0, 1); //打开文件。
            if (ctx->fd[0] < 0)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final_error);

            chk = archive_read_data_into_fd(ctx->arch_fd, ctx->fd[0]);
            if (chk != ARCHIVE_OK)
                ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final_error);
        }
        else
        {
            fprintf(stderr, "%s -> 不支持的类型，跳过。 \n", name_cp);
            ABCDK_ERRNO_AND_GOTO1(chk = 0, final);
        }

        /*属性和路径缓存起来。*/
        _abcdkarchive_read_push_attr(ctx,name_cp,&file_stat);
    }

    /*No error.*/
    goto final;

final_error:

    /*error.*/
    chk = -1;

final:

    abcdk_closep(&ctx->fd[0]);

    return chk;
}

void _abcdkarchive_read_recover_attr(abcdkarchive_t *ctx)
{
    char name_cp[PATH_MAX] = {0};
    char pathfile[PATH_MAX] = {0};
    struct stat file_stat = {0};
    int fd = -1;
    int chk;

    if(ctx->justlist)
        return;

    while (1)
    {
        memset(name_cp,0,PATH_MAX);
        chk = _abcdkarchive_read_pop_attr(ctx, name_cp, &file_stat);
        if (chk != 0)
            break;
        
        memset(pathfile,0,PATH_MAX);
        abcdk_dirdir(pathfile, ctx->wksp);
        abcdk_dirdir(pathfile, name_cp);

        abcdk_closep(&ctx->fd[0]);

        if (S_ISDIR(file_stat.st_mode))
        {
            ctx->fd[0] = open(pathfile, O_DIRECTORY | O_CLOEXEC, 0); //打开目录。
        }
        else if (S_ISREG((file_stat.st_mode)))
        {
            ctx->fd[0] = abcdk_open(pathfile, 1, 0, 0);
        }
        else
        {
            /*跳过所有不支持的类型。*/
            continue;
        }

        /*恢复文件(目录)属性的时间。*/
        chk = abcdk_futimens(ctx->fd[0], &file_stat.st_atim, &file_stat.st_mtim);
        if (chk != 0)
            fprintf(stderr, "%s -> 未能恢复文件时间，忽略。\n", name_cp);

        /*恢复文件(目录)属性的权限。*/
        if (file_stat.st_mode & ACCESSPERMS)
        {
            chk = fchmod(ctx->fd[0], file_stat.st_mode & ACCESSPERMS);
            if (chk != 0)
                fprintf(stderr, "%s -> 未能恢复文件权限，忽略。\n", name_cp);
        }

        /*恢复文件(目录)所有者和所属组。*/
        if(getuid() == 0)
        {
            chk = fchown(ctx->fd[0], file_stat.st_uid, file_stat.st_gid);
            if (chk != 0)
                fprintf(stderr, "%s -> 未能恢复文件的用户和组，忽略。\n", name_cp);
        }
    }

    abcdk_closep(&ctx->fd[0]);
}

void _abcdkarchive_read_real(abcdkarchive_t *ctx)
{
    int chk;
#if ARCHIVE_VERSION_NUMBER >= 3000000
    chk = archive_read_open_filenames(ctx->arch_fd, ctx->volumes, ctx->blk);
#else 
    chk = archive_read_open_filename(ctx->arch_fd, ctx->volumes[0], ctx->blk);
#endif //ARCHIVE_VERSION_NUMBER >= 3000000
    if (chk != ARCHIVE_OK)
    {
        fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
    }

    while (1)
    {
        chk = archive_read_next_header(ctx->arch_fd, &ctx->arch_entry);
        if (chk == ARCHIVE_EOF)
        {
            /*尝试恢复属性。*/
            _abcdkarchive_read_recover_attr(ctx);
            goto final;
        }
        else if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }

        chk = _abcdkarchive_read_one(ctx);
        if (chk != 0)
            goto final;
    }

final:

    return;
}

void _abcdkarchive_read(abcdkarchive_t *ctx)
{
    int chk;
    
    ctx->file_num = abcdk_option_count(ctx->args, "--file-list");
    for (int i = 0; i < ctx->file_num && i < 256; i++)
        ctx->files[i] = abcdk_option_get(ctx->args, "--file-list", i, NULL);

    ctx->justlist = abcdk_option_exist(ctx->args, "--just-list");

    ctx->arch_fd = NULL;
    ctx->fd[0] = -1;

    if (ctx->volumes[0] == NULL || ctx->volumes[0][0] == '\0')
    {
        fprintf(stderr, "'--volume NAME [ NAME-part1 NAME-part2 ... ]' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    for (int i = 0; i < ctx->volume_num; i++)
    {
        if (access(ctx->volumes[0], R_OK) != 0)
        {
            fprintf(stderr, "'%s' %s。\n", ctx->volumes[i], strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }
    }

    if (!ctx->justlist)
    {
        if (ctx->wksp == NULL || *ctx->wksp == '\0')
        {
            fprintf(stderr, "'--workspace PATH' 不能省略，且不能为空。\n");
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
        }

        if (access(ctx->wksp, W_OK) != 0)
        {
            fprintf(stderr, "'%s' %s。\n", ctx->wksp, strerror(errno));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
        }
    }

    ctx->arch_fd = archive_read_new();
    if (!ctx->arch_fd)
    {
        fprintf(stderr, "%s。\n", strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

#if ARCHIVE_VERSION_NUMBER >= 3000000
    if (ctx->flt > 0)
    {
        chk = archive_read_append_filter(ctx->arch_fd, abcdkarchive_find_filter(ctx->flt));
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
    else
    {
        chk = archive_read_support_filter_all(ctx->arch_fd);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
#endif // ARCHIVE_VERSION_NUMBER >= 3000000

#if ARCHIVE_VERSION_NUMBER >= 3000000
    if (ctx->fmt > 0)
    {
        chk = archive_read_set_format(ctx->arch_fd, abcdkarchive_find_format(ctx->fmt));
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
    else
#endif // ARCHIVE_VERSION_NUMBER >= 3000000
    {
        chk = archive_read_support_format_all(ctx->arch_fd);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }

    if (ctx->opt)
    {
        chk = archive_read_set_options(ctx->arch_fd, ctx->opt);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }

#if ARCHIVE_VERSION_NUMBER >= 3003003
    if (ctx->pwd)
    {
        chk = archive_read_add_passphrase(ctx->arch_fd, ctx->pwd);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

    _abcdkarchive_read_real(ctx);

final:

    if (ctx->arch_fd)
    {
#if ARCHIVE_VERSION_NUMBER >= 3000000
        archive_read_free(ctx->arch_fd);
#else 
        archive_read_close(ctx->arch_fd);
#endif //ARCHIVE_VERSION_NUMBER >= 3000000
        ctx->arch_fd = NULL;
    }

    abcdk_object_unref(&ctx->attr_list);
}

int _abcdkarchive_write_one(abcdkarchive_t *ctx,const char *file,struct stat *attr)
{
    struct archive_entry *entry = NULL;
    char linkname[PATH_MAX] = {0};
    int fd = -1;
    ssize_t rlen = 0, wlen = 0;
    int chk = -1, reader_chk = -1;

    if (!(S_ISREG(attr->st_mode) || S_ISLNK(attr->st_mode) || S_ISDIR(attr->st_mode)))
    {
        fprintf(stderr, "'%s' 不支持此类文件归档。\n", file);
        return -2;
    }

    fprintf(stderr, "%s\n", file + strlen(ctx->wksp));

    entry = archive_entry_new();
    if(!entry)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    if (ctx->save_fullpath)
        archive_entry_copy_pathname(entry, file);
    else
        archive_entry_copy_pathname(entry, file + strlen(ctx->wksp));

    if(S_ISLNK(attr->st_mode))
    {
        readlink(file,linkname,PATH_MAX);
        archive_entry_copy_symlink(entry, linkname);
    }

    archive_entry_copy_stat(entry,attr);

    chk = archive_write_header(ctx->arch_fd, entry);
    if (chk != ARCHIVE_OK)
    {
        fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
    }

    /*目录和软链接没有实体。*/
    if (S_ISDIR(attr->st_mode) || S_ISLNK(attr->st_mode))
        ABCDK_ERRNO_AND_GOTO1(chk = 0, final);

    
    fd = abcdk_open(file, 0, 0, 0);
    if (fd < 0)
    {
        fprintf(stderr, "'%s' %s。\n", file,strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    /*当文件大小大于读写缓存时启用加速器。*/
    if (attr->st_size > ctx->buf_size)
    {
        if (!ctx->reader)
            ctx->reader = abcdk_reader_create(ctx->buf_size);
        if (!ctx->reader)
            fprintf(stderr, "'启动加速器失败' %s。\n", strerror(errno));

        if(ctx->reader)
            reader_chk = abcdk_reader_start(ctx->reader, fd);
    }

    if(!ctx->buf)
    {    
        ctx->buf = abcdk_heap_alloc(ctx->buf_size);
        if(!ctx->buf)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    for(;;)
    {
        if(reader_chk == 0)
            rlen = abcdk_reader_read(ctx->reader, ctx->buf, ctx->buf_size);
        else 
            rlen = abcdk_read(fd, ctx->buf, ctx->buf_size);
        if (rlen <= 0)
            break;

        wlen = archive_write_data(ctx->arch_fd, ctx->buf, rlen);
        if (wlen != rlen)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }

    chk = archive_write_finish_entry(ctx->arch_fd);
    if (chk != ARCHIVE_OK)
    {
        fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
    }
   
    /*No error.*/
    chk = 0;

final:

    if(ctx->reader)
        abcdk_reader_stop(ctx->reader);
    abcdk_closep(&fd);
    
    if(entry)
        archive_entry_free(entry);

    return chk;
}

void _abcdkarchive_write_real(abcdkarchive_t *ctx)
{
    struct stat attr = {0};
    char file[PATH_MAX] = {0};
    abcdk_tree_t *dir = NULL;
    char *bname_p = NULL;
    int chk;

    dir = abcdk_tree_alloc3(1);
    if(!dir)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    for (int i = 0; i < ctx->file_num; i++)
    {
        memset(file,0,PATH_MAX);
        abcdk_dirdir(file,ctx->wksp);
        abcdk_dirdir(file,ctx->files[i]);

        /*跳过.和..*/
        bname_p = strrchr(file, '/');
        bname_p = (bname_p?(bname_p + 1):file);
        if (abcdk_strcmp(bname_p, ".", 1) == 0 || abcdk_strcmp(bname_p, "..", 1) == 0)
            continue;

        chk = lstat(file,&attr);
        if(chk != 0)
        {
            fprintf(stderr, "'%s' %s。\n",file,strerror(errno));
            continue;
        }

        if(S_ISDIR(attr.st_mode))
        {
            chk = abcdk_dirent_open(&dir,file);
            if(chk != 0)
            {
                fprintf(stderr, "'%s' %s。\n",file,strerror(errno));
                continue;
            }
        }

        chk = _abcdkarchive_write_one(ctx, file, &attr);
        if (chk != 0 && chk != -2)
            goto final;
    }

    for(;;)
    {
        memset(file,0,PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, file);
        if (chk != 0)
            break;

        chk = lstat(file, &attr);
        if (chk != 0)
        {
            fprintf(stderr, "'%s' %s。\n", file, strerror(errno));
            continue;
        }

        if(S_ISDIR(attr.st_mode))
        {
            chk = abcdk_dirent_open(&dir,file);
            if(chk != 0)
            {
                fprintf(stderr, "'%s' %s。\n",file,strerror(errno));
                continue;
            }
        }

        chk = _abcdkarchive_write_one(ctx, file, &attr);
        if (chk != 0 && chk != -2)
            goto final;
    }
    
final:

    abcdk_tree_free(&dir);
}

int _abcdkarchive_write_open_cb(struct archive *fd, void *opaque)
{
    abcdkarchive_t *ctx = (abcdkarchive_t *)opaque;

    for (int i = 0; i < ctx->volume_num; i++)
    {
        ctx->fd[i] = abcdk_open(ctx->volumes[i], 1, 0, 1);
        if (ctx->fd[i] < 0)
        {
            fprintf(stderr, "'%s' %s。\n", ctx->volumes[i], strerror(errno));
            ABCDK_ERRNO_AND_RETURN1(ctx->errcode = errno, ARCHIVE_FAILED);
        }
    }

    return ARCHIVE_OK;
}

ssize_t _abcdkarchive_write_write_cb(struct archive *fd, void *opaque, const void *_buffer, size_t _length)
{
    abcdkarchive_t *ctx = (abcdkarchive_t *)opaque;
    ssize_t wall = 0;

#pragma omp parallel for num_threads(ctx->volume_num)
    for (int i = 0; i < ctx->volume_num; i++)
    {
        ssize_t wlen = abcdk_write(ctx->fd[i], _buffer, _length);
#pragma omp atomic
        wall += ((wlen > 0) ? wlen : 0);
    }
    
    /*任何一路写失败就报错。*/
    return (((wall / ctx->volume_num) == _length)?_length:-1);
}

int _abcdkarchive_write_close_cb(struct archive *fd, void *opaque)
{
    abcdkarchive_t *ctx = (abcdkarchive_t *)opaque;

#pragma omp parallel for num_threads(ctx->volume_num)
    for (int i = 0; i < ctx->volume_num; i++)
    {
        abcdk_closep(&ctx->fd[i]);
    }

    return ARCHIVE_OK;
}

void _abcdkarchive_write(abcdkarchive_t *ctx)
{
    int chk;

    ctx->file_num = abcdk_option_count(ctx->args, "--file-list");
    for (int i = 0; i < ctx->file_num && i < 256; i++)
        ctx->files[i] = abcdk_option_get(ctx->args, "--file-list", i, NULL);
    ctx->save_fullpath = abcdk_option_exist(ctx->args,"--save-fullpath");
    
    ctx->arch_fd = NULL;
    ctx->reader = NULL;
    ctx->buf_size = 1024*1024;
    ctx->buf = NULL;
    for (int i = 0; i < 256; i++)
        ctx->fd[i] = -1;

    if (ctx->volumes[0] == NULL || ctx->volumes[0][0] == '\0')
    {
        fprintf(stderr, "'--volume NAME [ NAME-1 NAME-2 ... ]' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (ctx->files[0] == NULL || *ctx->files[0] == '\0')
    {
        fprintf(stderr, "'--file-from FILE|DIR [ FILE|DIR ... ]' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (ctx->wksp == NULL || *ctx->wksp == '\0')
    {
        fprintf(stderr, "'--workspace PATH' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->wksp, R_OK) != 0)
    {
        fprintf(stderr, "'%s' %s。\n", ctx->wksp, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    ctx->arch_fd = archive_write_new();
    if (!ctx->arch_fd)
    {
        fprintf(stderr, "%s。\n", strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    chk = archive_write_set_bytes_per_block(ctx->arch_fd, ctx->blk);
    if (chk != ARCHIVE_OK)
    {
        fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
    }
    
#if ARCHIVE_VERSION_NUMBER >= 3000000
    if (ctx->flt > 0)
    {
        chk = archive_write_add_filter(ctx->arch_fd, abcdkarchive_find_filter(ctx->flt));
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
#endif //ARCHIVE_VERSION_NUMBER >= 3000000

    if (ctx->fmt > 0)
    {
        chk = archive_write_set_format(ctx->arch_fd, abcdkarchive_find_format(ctx->fmt));
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }

    if (ctx->opt)
    {
        chk = archive_write_set_options(ctx->arch_fd, ctx->opt);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }

#if ARCHIVE_VERSION_NUMBER >= 3003003
    if (ctx->pwd)
    {
        chk = archive_write_set_passphrase(ctx->arch_fd, ctx->pwd);
        if (chk != ARCHIVE_OK)
        {
            fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
        }
    }
#endif // ARCHIVE_VERSION_NUMBER >= 3003003

    chk = archive_write_open(ctx->arch_fd,ctx,_abcdkarchive_write_open_cb,_abcdkarchive_write_write_cb,_abcdkarchive_write_close_cb);
    if (chk != ARCHIVE_OK)
    {
        fprintf(stderr, "%s。\n", archive_error_string(ctx->arch_fd));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = archive_errno(ctx->arch_fd), final);
    }
    
    _abcdkarchive_write_real(ctx);

final:

    if (ctx->arch_fd)
    {
#if ARCHIVE_VERSION_NUMBER >= 3000000
        archive_write_free(ctx->arch_fd);
#else 
        archive_write_close(ctx->arch_fd);
#endif //ARCHIVE_VERSION_NUMBER >= 3000000
        ctx->arch_fd = NULL;
    }

    abcdk_heap_free2(&ctx->buf);
    abcdk_reader_destroy(&ctx->reader);
    for (int i = 0; i < 256; i++)
        abcdk_closep(&ctx->fd[i]);
}

void _abcdkarchive_work(abcdkarchive_t *ctx)
{
    ctx->cmd = abcdk_option_get_int(ctx->args, "--cmd", 0, ABCDKARCHIVE_READ);
    ctx->flt = abcdk_option_get_int(ctx->args, "--filter", 0, -1);
    ctx->fmt = abcdk_option_get_int(ctx->args, "--format", 0, -1);
    ctx->pwd = abcdk_option_get(ctx->args, "--password", 0, NULL);
    ctx->opt = abcdk_option_get(ctx->args, "--option", 0, NULL);
    ctx->blk = abcdk_option_get_int(ctx->args, "--block-size", 0, 10240);
    ctx->md_cst = abcdk_option_get(ctx->args, "--metadata-charset", 0, "UTF-8");
    ctx->md_cst_w = abcdk_option_get_int(ctx->args, "--metadata-char-width", 0, 1);
    ctx->wksp = abcdk_option_get(ctx->args, "--work-space", 0, "./");
    ctx->volume_num = ABCDK_CLAMP(abcdk_option_count(ctx->args, "--volume"), 1, 255);
    for (int i = 0; i < ctx->volume_num; i++)
        ctx->volumes[i] = abcdk_option_get(ctx->args, "--volume", i, NULL);

    if (ctx->cmd == ABCDKARCHIVE_READ)
    {
        _abcdkarchive_read(ctx);
    }
    else if (ctx->cmd == ABCDKARCHIVE_WRITE)
    {
        _abcdkarchive_write(ctx);
    }
    else
    {
        fprintf(stderr, "尚未支持。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

final:

    return;
}

#endif // HAVE_ARCHIVE

int abcdk_tool_archive(abcdk_option_t *args)
{
#ifdef HAVE_ARCHIVE
    abcdkarchive_t ctx = {0};

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

#else // HAVE_ARCHIVE

    fprintf(stderr, "当前构建版本未包含此工具。\n");
    return EPERM;

#endif // HAVE_ARCHIVE
}