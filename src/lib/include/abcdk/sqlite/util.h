/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SQLITE_UTIL_H
#define ABCDK_SQLITE_UTIL_H

#include "abcdk/sqlite/sqlite.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"

__BEGIN_DECLS


/**
 * 字段类型。
 * 
*/
typedef enum _abcdk_sqlite_column_type
{
    ABCDK_SQLITE_COLUMN_INT = 1,
#define ABCDK_SQLITE_COLUMN_INT ABCDK_SQLITE_COLUMN_INT

    ABCDK_SQLITE_COLUMN_INT64 = 2,
#define ABCDK_SQLITE_COLUMN_INT64 ABCDK_SQLITE_COLUMN_INT64

    ABCDK_SQLITE_COLUMN_DOUBLE = 3,
#define ABCDK_SQLITE_COLUMN_DOUBLE ABCDK_SQLITE_COLUMN_DOUBLE

    ABCDK_SQLITE_COLUMN_VARCHAR = 4,
#define ABCDK_SQLITE_COLUMN_VARCHAR ABCDK_SQLITE_COLUMN_VARCHAR

    ABCDK_SQLITE_COLUMN_BLOB = 5
#define ABCDK_SQLITE_COLUMN_BLOB ABCDK_SQLITE_COLUMN_BLOB

}abcdk_sqlite_column_type_t;

/**
 * 日志模式。
*/
typedef enum _abcdk_sqlite_journal_mode
{
    ABCDK_SQLITE_JOURNAL_OFF = 0,
#define ABCDK_SQLITE_JOURNAL_OFF ABCDK_SQLITE_JOURNAL_OFF

    ABCDK_SQLITE_JOURNAL_DELETE = 1,
#define ABCDK_SQLITE_JOURNAL_DELETE ABCDK_SQLITE_JOURNAL_DELETE

    ABCDK_SQLITE_JOURNAL_TRUNCATE = 2,
#define ABCDK_SQLITE_JOURNAL_TRUNCATE ABCDK_SQLITE_JOURNAL_TRUNCATE

    ABCDK_SQLITE_JOURNAL_PERSIST = 3,
#define ABCDK_SQLITE_JOURNAL_PERSIST ABCDK_SQLITE_JOURNAL_PERSIST

    ABCDK_SQLITE_JOURNAL_MEMORY = 4,
#define ABCDK_SQLITE_JOURNAL_MEMORY ABCDK_SQLITE_JOURNAL_MEMORY

    ABCDK_SQLITE_JOURNAL_WAL = 5
#define ABCDK_SQLITE_JOURNAL_WAL ABCDK_SQLITE_JOURNAL_WAL
}abcdk_sqlite_journal_mode_t;

/**
 * 备份参数。
*/
typedef struct _abcdk_sqlite_backup_param
{
    /**目标库的指针。*/
    sqlite3 *dst;

    /**目标库的名字的指针。*/
    const char *dst_name;

    /**源库的指针。*/
    sqlite3 *src;

    /**源库的名字的指针。*/
    const char *src_name;

    /**备份步长(页数量)。*/
    int step;

    /*忙时休息时长(毫秒)。*/
    int sleep;

    /**
     * 进度函数。
     * 
     * @param remaining  剩余页数量。
     * @param total 总页数量。
     * @param opaque 环境指针。
    */
    void (*progress_cb)(int remaining, int total, void *opaque);

    /**环境指针。*/
    void *opaque;

} abcdk_sqlite_backup_param;

/**
 * 备份。
 * 
 * @return SQLITE_OK(0) 成功，!SQLITE_OK(0) 失败。
 * 
 */
int abcdk_sqlite_backup(abcdk_sqlite_backup_param *param);

/**
 * 关闭数据库句柄。
 * 
 * @return SQLITE_OK(0) 成功，!SQLITE_OK(0) 失败。
 * 
*/
int abcdk_sqlite_close(sqlite3 *ctx);

/**
 * 忙碌处理函数。
*/
int abcdk_sqlite_busy_melt(void *opaque, int count);

/**
 * 打开数据库文件。
 * 
 * @param name 数据库文件名的指针。
 * 
 * @return !NULL(0) 成功(句柄)，NULL(0) 失败。
 * 
*/
sqlite3 *abcdk_sqlite_open(const char *name);

/**
 * 打开内存数据库。
*/
sqlite3* abcdk_sqlite_memopen();

/**
 * 启动事物。
*/
int abcdk_sqlite_tran_begin(sqlite3 *ctx);

/**
 * 提交事物。
*/
int abcdk_sqlite_tran_commit(sqlite3 *ctx);

/**
 * 回滚事物。
*/
int abcdk_sqlite_tran_rollback(sqlite3 *ctx);

/**
 * 回收空间。
*/
int abcdk_sqlite_tran_vacuum(sqlite3 *ctx);

/** 
 * 设置页大小。
 * 
 * @return SQLITE_OK(0) 成功，!SQLITE_OK(0) 失败。
 * 
*/
int abcdk_sqlite_pagesize(sqlite3 *ctx, int size);

/** 
 * 设置日志模式。
 * 
 * @return SQLITE_OK(0) 成功，!SQLITE_OK(0) 失败。
 * 
*/
int abcdk_sqlite_journal_mode(sqlite3 *ctx, int mode);

/**
 * 准备SQL语句。
 * 
 * @return !NULL(0) 成功(数据集指针)，NULL(0) 失败。
 * 
*/
sqlite3_stmt* abcdk_sqlite_prepare(sqlite3 *ctx,const char *sql);

/** 
 * 提交语句，或在数据集中移动游标到下一行。
 * 
 * @return > 0 有数据返回，= 0 无数返回(或末尾)。< 0 出错。
 * 
*/
int abcdk_sqlite_step(sqlite3_stmt *stmt);

/**
 * 关闭数据集。
 * 
 * @return SQLITE_OK(0) 成功，!SQLITE_OK(0) 失败。
 * 
*/
int abcdk_sqlite_finalize(sqlite3_stmt *stmt);

/**
 * 直接执行SQL语句。
 * 
 * @note 不能用于返回数据集。
 * 
 * @return >= 0 成功。< 0 出错。
 * 
*/
int abcdk_sqlite_exec_direct(sqlite3 *ctx,const char *sql);

/**
 * 获取数据集中指定字段的值。
 *
 * @param [in] max 缓存区最大长度，值超过这个长度的则会被截断。
 *
 * @return >= 0 成功(数据长度)， < 0 失败(未找到)。
 */
int abcdk_sqlite_get_data(sqlite3_stmt *ctx, int column, int type, void *buf, int max);

/**
 * 在数据集中查找字段的索引。
 * 
 * @return >= 0 成功(索引)，< 0 失败(未找到)。
 * 
*/
int abcdk_sqlite_name2index(sqlite3_stmt *stmt, const char *name);


__END_DECLS

#endif //ABCDK_SQLITE_UTIL_H