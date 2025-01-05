/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
*/
#include "abcdk/database/sqlite.h"

#if defined(_SQLITE3_H_) || defined(SQLITE3_H)

int abcdk_sqlite_backup(abcdk_sqlite_backup_param *param)
{
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

    if(chk == SQLITE_DONE)
        chk = sqlite3_backup_finish(backup_ctx);

    return ((chk == SQLITE_DONE) ? SQLITE_OK : chk);
}

int  abcdk_sqlite_close(sqlite3 *ctx)
{
    assert(ctx != NULL);

    return sqlite3_close(ctx);
}

int abcdk_sqlite_busy_melt(void *opaque,int count)
{
    /**/
    sched_yield();

    /*
     * 1: try again.
     * 0: not.
    */

    return 1;
}

sqlite3* abcdk_sqlite_open(const char *name)
{
    sqlite3 *ctx = NULL;
    int chk;

    assert(name != NULL);

    chk = sqlite3_open(name, &ctx);
    if(chk != SQLITE_OK)
        ABCDK_ERRNO_AND_RETURN1(EINVAL,NULL);

    /*注册忙碌处理函数。*/
    sqlite3_busy_handler(ctx,abcdk_sqlite_busy_melt,ctx);

    return ctx;
}

int abcdk_sqlite_pagesize(sqlite3 * ctx,int size)
{
    char sql[100] = {0};

    assert(ctx != NULL && size > 0);

    assert(size % 4096 == 0);

    snprintf(sql, 100, "PRAGMA page_size = %d;", size);

    return sqlite3_exec(ctx, sql, NULL, NULL, NULL);
}

int abcdk_sqlite_journal_mode(sqlite3 *ctx, int mode)
{
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
}

sqlite3_stmt* abcdk_sqlite_prepare(sqlite3 *ctx,const char *sql)
{
    sqlite3_stmt *stmt = NULL;
    int chk;

    assert(ctx != NULL && sql != NULL);

    chk = sqlite3_prepare(ctx, sql, -1, &stmt, NULL);
    if (SQLITE_OK != chk)
        ABCDK_ERRNO_AND_RETURN1(EINVAL,NULL);

    return stmt;
}

int abcdk_sqlite_step(sqlite3_stmt *stmt)
{
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
}

int abcdk_sqlite_finalize(sqlite3_stmt *stmt)
{
    assert(stmt != NULL);

    return sqlite3_finalize(stmt);
}

int abcdk_sqlite_exec_direct(sqlite3 *ctx, const char *sql)
{
    sqlite3_stmt *stmt;
    int chk;

    assert(ctx != NULL && sql != NULL);

    stmt = abcdk_sqlite_prepare(ctx, sql);
    if (!stmt)
        return -1;

    chk = abcdk_sqlite_step(stmt);
    abcdk_sqlite_finalize(stmt);

    return chk;
}

int abcdk_sqlite_name2index(sqlite3_stmt *stmt, const char *name)
{
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
}

#endif //defined(_SQLITE3_H_) || defined(SQLITE3_H)