/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/sqlite/util.h"

int abcdk_sqlite_backup(abcdk_sqlite_backup_param *param)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    sqlite3_backup *backup_ctx = NULL;
    int chk;

    assert(param != NULL);

    assert(param->dst != NULL && param->dst_name != NULL);
    assert(param->src != NULL && param->src_name != NULL);
    assert(param->step > 0 && param->sleep > 0);

    backup_ctx = sqlite3_backup_init(param->dst, param->dst_name, param->src, param->src_name);
    if (!backup_ctx)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, SQLITE_ERROR);

    do
    {
        chk = sqlite3_backup_step(backup_ctx, param->step);

        if (param->progress_cb)
            param->progress_cb(sqlite3_backup_remaining(backup_ctx), sqlite3_backup_pagecount(backup_ctx), param->opaque);

        if (chk == SQLITE_BUSY || chk == SQLITE_LOCKED)
            sqlite3_sleep(param->sleep);

    } while (chk == SQLITE_OK || chk == SQLITE_BUSY || chk == SQLITE_LOCKED);

    if (chk == SQLITE_DONE)
        chk = sqlite3_backup_finish(backup_ctx);

    return ((chk == SQLITE_DONE) ? SQLITE_OK : chk);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_close(sqlite3 *ctx)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(ctx != NULL);

    return sqlite3_close(ctx);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_busy_melt(void *opaque, int count)
{
    /**/
    sched_yield();

    /*
     * 1: try again.
     * 0: not.
     */

    return 1;
}

sqlite3 *abcdk_sqlite_open(const char *name)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return NULL;
#else //#ifndef HAVE_SQLITE
    sqlite3 *ctx = NULL;
    int chk;

    assert(name != NULL);

    chk = sqlite3_open(name, &ctx);
    if (chk != SQLITE_OK)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, NULL);

    /*注册忙碌处理函数。*/
    sqlite3_busy_handler(ctx, abcdk_sqlite_busy_melt, ctx);

    return ctx;
#endif //#ifndef HAVE_SQLITE
}

sqlite3 *abcdk_sqlite_memopen()
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return NULL;
#else //#ifndef HAVE_SQLITE
    return abcdk_sqlite_open(":memory:");
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_tran_begin(sqlite3 *ctx)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(ctx != NULL);

    return sqlite3_exec(ctx, "begin;", NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_tran_commit(sqlite3 *ctx)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(ctx != NULL);

    return sqlite3_exec(ctx, "commit;", NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_tran_rollback(sqlite3 *ctx)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(ctx != NULL);

    return sqlite3_exec(ctx, "rollback;", NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_tran_vacuum(sqlite3 *ctx)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(ctx != NULL);

    return sqlite3_exec(ctx, "vacuum;", NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_pagesize(sqlite3 *ctx, int size)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    char sql[100] = {0};

    assert(ctx != NULL && size > 0);

    assert(size % 4096 == 0);

    snprintf(sql, 100, "PRAGMA page_size = %d;", size);

    return sqlite3_exec(ctx, sql, NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_journal_mode(sqlite3 *ctx, int mode)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    const char *sql = "";
    int chk;

    assert(ctx != NULL && mode >= ABCDK_SQLITE_JOURNAL_OFF && mode <= ABCDK_SQLITE_JOURNAL_WAL);

    switch (mode)
    {
    case ABCDK_SQLITE_JOURNAL_OFF:
        sql = "PRAGMA journal_mode = OFF;";
        break;
    case ABCDK_SQLITE_JOURNAL_DELETE:
        sql = "PRAGMA journal_mode = DELETE;";
        break;
    case ABCDK_SQLITE_JOURNAL_TRUNCATE:
        sql = "PRAGMA journal_mode = TRUNCATE;";
        break;
    case ABCDK_SQLITE_JOURNAL_PERSIST:
        sql = "PRAGMA journal_mode = PERSIST;";
        break;
    case ABCDK_SQLITE_JOURNAL_MEMORY:
        sql = "PRAGMA journal_mode = MEMORY;";
        break;
    case ABCDK_SQLITE_JOURNAL_WAL:
        sql = "PRAGMA journal_mode = WAL;";
        break;
    default:
        sql = "";
    }

    return sqlite3_exec(ctx, sql, NULL, NULL, NULL);
#endif //#ifndef HAVE_SQLITE
}

sqlite3_stmt *abcdk_sqlite_prepare(sqlite3 *ctx, const char *sql)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return NULL;
#else //#ifndef HAVE_SQLITE
    sqlite3_stmt *stmt = NULL;
    int chk;

    assert(ctx != NULL && sql != NULL);

    chk = sqlite3_prepare(ctx, sql, -1, &stmt, NULL);
    if (SQLITE_OK != chk)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, NULL);

    return stmt;
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_step(sqlite3_stmt *stmt)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return -1;
#else //#ifndef HAVE_SQLITE
    int chk;

    assert(stmt != NULL);

    chk = sqlite3_step(stmt);
    if (chk == SQLITE_ROW)
        return 1;
    else if (chk == SQLITE_DONE)
        return 0;
    else
        return -1;

    return -1;
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_finalize(sqlite3_stmt *stmt)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    assert(stmt != NULL);

    return sqlite3_finalize(stmt);
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_exec_direct(sqlite3 *ctx, const char *sql)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return SQLITE_ERROR;
#else //#ifndef HAVE_SQLITE
    sqlite3_stmt *stmt;
    int chk;

    assert(ctx != NULL && sql != NULL);

    stmt = abcdk_sqlite_prepare(ctx, sql);
    if (!stmt)
        return -1;

    chk = abcdk_sqlite_step(stmt);
    abcdk_sqlite_finalize(stmt);

    return chk;
#endif //#ifndef HAVE_SQLITE
}

int abcdk_sqlite_get_data(sqlite3_stmt *ctx, int column, int type, void *buf, int max)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return -1;
#else //#ifndef HAVE_SQLITE

    int val_len = sqlite3_column_bytes(ctx, column);
    if(val_len < 0)
        return -1;

    int read_len = ABCDK_MIN(max,val_len);

    if(type == ABCDK_SQLITE_COLUMN_BLOB)
        memcpy(buf,sqlite3_column_blob(ctx,column),read_len);
    else if(type == ABCDK_SQLITE_COLUMN_VARCHAR)
        memcpy(buf,sqlite3_column_text(ctx,column),read_len);
    else if(type == ABCDK_SQLITE_COLUMN_DOUBLE)
    {
        double val = sqlite3_column_double(ctx,column);
        memcpy(buf,&val,read_len);
    }
    else if(type == ABCDK_SQLITE_COLUMN_INT64)
    {
        int64_t val = sqlite3_column_int64(ctx,column);
        memcpy(buf,&val,read_len);
    }
    else if(type == ABCDK_SQLITE_COLUMN_INT)
    {
        int val = sqlite3_column_int(ctx,column);
        memcpy(buf,&val,read_len);
    }
    else
    {
        return -1;
    }

    return read_len;

#endif //#ifndef HAVE_SQLITE
}


int abcdk_sqlite_name2index(sqlite3_stmt *stmt, const char *name)
{
#ifndef HAVE_SQLITE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含SQLITE工具。"));
    return -1;
#else //#ifndef HAVE_SQLITE
    int count;
    const char *tmp;
    int idx = -1;

    assert(stmt != NULL && name != NULL);

    count = sqlite3_column_count(stmt);

    for (idx = count - 1; idx >= 0; idx--)
    {
        tmp = sqlite3_column_name(stmt, idx);

        if (abcdk_strcmp(name, tmp, 0) == 0)
            break;
    }

    return idx;
#endif //#ifndef HAVE_SQLITE
}
