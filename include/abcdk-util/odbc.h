/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_ODBC_H
#define ABCDK_UTIL_ODBC_H

#include "abcdk-util/general.h"
#include "abcdk-util/allocator.h"

#ifdef FREEIMAGE_H
#error "There is a conflict in the header files of FreeImage and unixODBC, so we can't use 'include' reference directly."
#endif //FREEIMAGE_H

#ifdef HAVE_UNIXODBC
#include <sql.h>
#include <sqlext.h>
#endif //HAVE_UNIXODBC

__BEGIN_DECLS

#if defined(__SQL_H) && defined(__SQLEXT_H)

/**
 * ODBC接口。
*/
typedef struct _abcdk_odbc
{
    /** 环境。 */
    SQLHENV env;

    /** 连接。 */
    SQLHDBC dbc;

    /** 数据集。 */
    SQLHSTMT stmt;

    /** 
     * 数据集属性。
     * 
     * @note 尽量不要直接修改。
     */
    abcdk_allocator_t *attr;

} abcdk_odbc_t;

/**
 * 释放数据集属性。
*/
void abcdk_odbc_free_attr(abcdk_odbc_t *ctx);

/**
 * 创建数据集属性。
 * 
 * @warning 同一份数据集只会创建一次。
*/
SQLRETURN abcdk_odbc_alloc_attr(abcdk_odbc_t *ctx);

/**
 * 断开连接。
*/
SQLRETURN abcdk_odbc_disconnect(abcdk_odbc_t *ctx);

/**
 * 连接数据库。
 * 
 * @warning 默认：禁用自动提交。
 * 
 * @param timeout 超时(秒)，仅登录时有效。
 * @param tracefile 跟踪文件，NULL(0) 忽略。
*/
SQLRETURN abcdk_odbc_connect(abcdk_odbc_t *ctx, const char *uri,time_t timeout,const char *tracefile);

/**
 * 连接数据库。
 * 
 * @warning 默认：使用TCP连接。
 * 
 * @param product 产品名称的指针。支持：DB2/MYSQL/ORACLE/SQLSERVER/POSTGRESQL
 * @param driver 驱动名称的指针。见：/etc/odbcinst.ini
 * @param host 主机地址的指针。
 * @param port 主机端口。
 * @param db 数据库名称的指针。
 * @param user 登录用名称的指针。
 * @param pwd 登录密码的指针。
 * 
*/
SQLRETURN abcdk_odbc_connect2(abcdk_odbc_t *ctx, const char *product, const char *driver,
                             const char *host, uint16_t port, const char *db,
                             const char *user, const char *pwd,
                             time_t timeout, const char *tracefile);

/**
 * 启用或禁用自动提交。
 * 
 * @param enable !0 启用，0 禁用。
*/
SQLRETURN abcdk_odbc_autocommit(abcdk_odbc_t *ctx,int enable);

/**
 * 开启事务(关闭自动提交)。
*/
#define abcdk_odbc_tran_begin(ctx)   abcdk_odbc_autocommit(ctx,0)

/**
 * 结束事务。
*/
SQLRETURN abcdk_odbc_tran_end(abcdk_odbc_t *ctx, SQLSMALLINT type);

/**
 * 提交事务。
*/
#define abcdk_odbc_tran_commit(ctx) abcdk_odbc_tran_end(ctx, SQL_COMMIT)

/**
 * 回滚事务。
*/
#define abcdk_odbc_tran_rollback(ctx) abcdk_odbc_tran_end(ctx, SQL_ROLLBACK)

/**
 * 准备SQL语句。
 * 
 * @warning 默认：启用静态游标。
*/
SQLRETURN abcdk_odbc_prepare(abcdk_odbc_t *ctx, const char *sql);

/**
 * 执行SQL语句。
 * 
 * @return SQL_SUCCESS(0) 成功，SQL_NO_DATA(100) 无数据，< 0 失败。
*/
SQLRETURN abcdk_odbc_execute(abcdk_odbc_t *ctx);

/**
 * 关闭数据集。
 * 
 * @note 在abcdk_odbc_execute之前执行有意义，其它情况可以不必执行，数据集允许复用。
*/
SQLRETURN abcdk_odbc_finalize(abcdk_odbc_t *ctx);

/**
 * 直接执行SQL语句。
 * 
 * @see abcdk_odbc_prepare
 * @see abcdk_odbc_execute
 * 
*/
SQLRETURN abcdk_odbc_exec_direct(abcdk_odbc_t *ctx,const char *sql);

/**
 * 返回影响的行数。
*/
SQLRETURN abcdk_odbc_affect(abcdk_odbc_t *ctx,SQLLEN *rows);

/**
 * 在数据集中移动游标。
 * 
 * @return SQL_SUCCESS(0) 成功，SQL_NO_DATA(100) 无数据(游标已经在数据集的首或尾)，< 0 失败。
*/
SQLRETURN abcdk_odbc_fetch(abcdk_odbc_t *ctx, SQLSMALLINT direction, SQLLEN offset);

/**
 * 在数据集中移动游标到首行。
*/
#define abcdk_odbc_fetch_first(ctx) abcdk_odbc_fetch(ctx, SQL_FETCH_FIRST, 0)

/**
 * 在数据集中向下移动游标。
*/
#define abcdk_odbc_fetch_next(ctx) abcdk_odbc_fetch(ctx, SQL_FETCH_NEXT, 0)


/** 
 * 获取数据集中指定字段的值。
 * 
 * @param max 缓存区最大长度，值超过这个长度的则会被截断。
 * @param len 字段值长度的指针，返回前填充，NULL(0)忽略。
 * 
*/
SQLRETURN abcdk_odbc_get_data(abcdk_odbc_t *ctx, SQLSMALLINT column, SQLSMALLINT type,
                             SQLPOINTER buf, SQLULEN max, SQLULEN *len);

/**
 * 在数据集中查找字段的索引。
 * 
 * @return >= 0 成功(索引)，< 0 失败(未找到)。
*/
SQLSMALLINT abcdk_odbc_name2index(abcdk_odbc_t *ctx, const char *name);

#endif //defined(__SQL_H) && defined(__SQLEXT_H)

__END_DECLS

#endif //ABCDK_UTIL_ODBC_H