/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_DATABASE_ODBCPOOL_H
#define ABCDK_DATABASE_ODBCPOOL_H

#include "abcdk/util/general.h"
#include "abcdk/util/time.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/pool.h"
#include "abcdk/database/odbc.h"


__BEGIN_DECLS


#if defined(__SQL_H) && defined(__SQLEXT_H)

/** ODBC连接池。*/
typedef struct _abcdk_odbcpool abcdk_odbcpool_t;

/** 
 * 连接数据库回调函数。
 * 
 * @return 0 成功，!0 失败。
*/
typedef int (*abcdk_odbcpool_connect_cb)(abcdk_odbc_t *ctx, void *opaque);

/**
 * 销毁连接池。
 */
void abcdk_odbcpool_destroy(abcdk_odbcpool_t **ctx);

/**
 * 创建连接池。
 *
 * @param size 池大小。
 * @param connect_cb 连接函数指针。
 * @param opaque 环境指针。
 *
 * @return !NULL(0) 成功(连接池对象的指针)，NULL(0) 失败(系统错误或内存不足)。
 */
abcdk_odbcpool_t *abcdk_odbcpool_create(size_t size, abcdk_odbcpool_connect_cb connect_cb, void *opaque);

/**
 * 弹出一个连接。
 * 
 * @param [in] timeout 超时(毫秒)。
 * 
 * @return !NULL(0) 成功(连接对象的指针)，NULL(0) 失败(数据库无法连接)。
*/
abcdk_odbc_t *abcdk_odbcpool_pop(abcdk_odbcpool_t *ctx,time_t timeout);

/**
 * 回收一个连接。
 * 
 * @param [in out] odbc 连接对象指针的指针。
*/
void abcdk_odbcpool_push(abcdk_odbcpool_t *ctx, abcdk_odbc_t **odbc);

#endif // defined(__SQL_H) && defined(__SQLEXT_H)

__END_DECLS

#endif //ABCDK_DATABASE_ODBCPOOL_H
