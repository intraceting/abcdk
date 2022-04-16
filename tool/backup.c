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

typedef struct _abcdkbcakup_ctx
{
    int errcode;
    abcdk_tree_t *args;


}abcdkbcakup_ctx;

void _abcdkbcakup_print_usage(abcdk_tree_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的备份工具。\n");

    fprintf(stderr, "\n通用选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--blksize < SIZE >\n");
    fprintf(stderr, "\t\t块大小（字节）。默认：10240\n");

    fprintf(stderr, "\n\t--workspace < PATH >\n");
    fprintf(stderr, "\t\t工作目录。默认：./\n");

    fprintf(stderr, "\n\t--file-list < FILE|DIR [ FILE|DIR ... ] >\n");
    fprintf(stderr, "\t\t文件来源。\n");

    fprintf(stderr, "\n\t--volume < NAME [ NAME ... ] >\n");
    fprintf(stderr, "\t\t卷名（包括路径）。\n");

}

ssize_t _abcdkbackup_write_cb(struct archive *fd, void *opaque, const void *data, size_t size)
{

}

int _abcdkbackup_open_cb(struct archive *fd, void *opaque)
{

}

int _abcdkbackup_close_cb(struct archive *fd, void *opaque)
{

}

void _abcdkbcakup_work(abcdkbcakup_ctx *ctx)
{

}

#endif // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)


int abcdk_tool_backup(abcdk_tree_t *args)
{
#if defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)
    abcdkbcakup_ctx ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkbcakup_print_usage(ctx.args);
    }
    else
    {
        _abcdkbcakup_work(&ctx);
    }

    return ctx.errcode;

#else // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)

    syslog(LOG_INFO, "当前构建版本未包含此工具。\n");
    return EPERM;

#endif // defined(ARCHIVE_H_INCLUDED) && defined(ARCHIVE_ENTRY_H_INCLUDED)
}