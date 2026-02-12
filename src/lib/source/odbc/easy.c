/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/odbc/easy.h"

/**简单的ODBC接口.*/
struct _abcdk_odbc
{
    /** 连接池ID.*/
    uint64_t pool;

    /** 环境. */
    SQLHENV env;

    /** 连接. */
    SQLHDBC dbc;

    /** 数据集. */
    SQLHSTMT stmt;

    /** 数据集属性.*/
    abcdk_object_t *attr;

};// abcdk_odbc_t;

#ifdef HAVE_UNIXODBC

SQLRETURN _abcdk_odbc_check_return(SQLRETURN ret)
{
    SQLRETURN chk = SQL_ERROR;

    if (ret == SQL_SUCCESS || ret == SQL_SUCCESS_WITH_INFO)
        chk = SQL_SUCCESS;
    else
        chk = ret;

    return chk;
}

void _abcdk_odbc_free_attr_destroy(abcdk_object_t *alloc, void *opaque)
{
    /*只有这个单独申请的.*/
    void *p = alloc->pptrs[6];

    abcdk_heap_freep(&p);
}

void _abcdk_odbc_free_attr(abcdk_odbc_t *ctx)
{
    abcdk_object_t *p;

    if (ctx->attr)
    {
        for (size_t i = 0; i < ctx->attr->numbers; i++)
        {
            p = (abcdk_object_t *)ctx->attr->pptrs[i];

            abcdk_object_unref((abcdk_object_t **)&p);
        }

        abcdk_object_unref(&ctx->attr);
    }
}

SQLRETURN _abcdk_odbc_alloc_attr(abcdk_odbc_t *ctx)
{
    SQLSMALLINT columns;
    abcdk_object_t *p;
    SQLRETURN chk;

    assert(ctx != NULL);

    /*同一份数据集只需创建一次.*/
    if (ctx->attr != NULL)
        return SQL_SUCCESS;

    chk = SQLNumResultCols(ctx->stmt, &columns);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    ctx->attr = abcdk_object_alloc(NULL, columns, 0);
    if (!ctx->attr)
    {
        chk = SQL_ERROR;
        goto final_error;
    }

    size_t sizes[7] = {NAME_MAX, sizeof(SQLSMALLINT), sizeof(SQLSMALLINT), sizeof(SQLULEN), sizeof(SQLSMALLINT), sizeof(SQLSMALLINT), 0};

    for (size_t i = 0; i < ctx->attr->numbers; i++)
    {
        p = abcdk_object_alloc(sizes, ABCDK_ARRAY_SIZE(sizes), 0);
        if (!p)
        {
            chk = SQL_ERROR;
            goto final_error;
        }

        ctx->attr->pptrs[i] = (uint8_t *)p;

        chk = SQLDescribeCol(ctx->stmt, (SQLSMALLINT)(i + 1), p->pptrs[0], p->sizes[0],
                             ABCDK_PTR2PTR(SQLSMALLINT, p->pptrs[1], 0), ABCDK_PTR2PTR(SQLSMALLINT, p->pptrs[2], 0),
                             ABCDK_PTR2PTR(SQLULEN, p->pptrs[3], 0), ABCDK_PTR2PTR(SQLSMALLINT, p->pptrs[4], 0),
                             ABCDK_PTR2PTR(SQLSMALLINT, p->pptrs[5], 0));

        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        /*申请字段值的缓存区.*/
        p->pptrs[6] = abcdk_heap_alloc(ABCDK_PTR2OBJ(SQLULEN, p->pptrs[3], 0) + 1);
        if (!p->pptrs[6])
        {
            chk = SQL_ERROR;
            goto final_error;
        }

        /*记录可用长度.*/
        p->sizes[6] = ABCDK_PTR2OBJ(SQLULEN, p->pptrs[3], 0) + 1;

        /*注册清理函数.*/
        abcdk_object_atfree(p, _abcdk_odbc_free_attr_destroy, NULL);
    }

    return SQL_SUCCESS;

final_error:

    _abcdk_odbc_free_attr(ctx);

    return chk;
}

#endif //#ifdef HAVE_UNIXODBC

void abcdk_odbc_free(abcdk_odbc_t **ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return;
#else //#ifndef HAVE_UNIXODBC

    abcdk_odbc_t *p = NULL;

    if(!ctx || !*ctx)
        return;

    p = *ctx;
    *ctx = NULL;

    abcdk_odbc_disconnect(p);
    abcdk_heap_free(p);

#endif //#ifndef HAVE_UNIXODBC
}

abcdk_odbc_t *abcdk_odbc_alloc(uint64_t pool)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return NULL;
#else //#ifndef HAVE_UNIXODBC
    abcdk_odbc_t *odbc = (abcdk_odbc_t *)abcdk_heap_alloc(sizeof(abcdk_odbc_t));
    if(!odbc)
        return NULL;
    
    odbc->attr = NULL;
    odbc->dbc = NULL;
    odbc->env = NULL;
    odbc->stmt = NULL;
    odbc->pool = pool;

    return odbc;
#endif //#ifndef HAVE_UNIXODBC
}

uint64_t abcdk_odbc_get_pool(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return 0;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return ctx->pool;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_disconnect(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL);

    /*清理数据集属性.*/
    _abcdk_odbc_free_attr(ctx);

    if (ctx->stmt)
    {
        chk = SQLFreeHandle(SQL_HANDLE_STMT, ctx->stmt);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        ctx->stmt = NULL;
    }

    if (ctx->dbc)
    {
        chk = SQLDisconnect(ctx->dbc);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        chk = SQLFreeHandle(SQL_HANDLE_DBC, ctx->dbc);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        ctx->dbc = NULL;
    }

    if (ctx->env)
    {
        chk = SQLFreeHandle(SQL_HANDLE_ENV, ctx->env);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        ctx->env = NULL;
    }

    return SQL_SUCCESS;

final_error:

    return chk;

#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_connect(abcdk_odbc_t *ctx, const char *uri, time_t timeout, const char *tracefile)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL && uri != NULL && timeout >= 0);

    if (!ctx->env)
    {
        chk = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &ctx->env);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        chk = SQLSetEnvAttr(ctx->env, SQL_ATTR_ODBC_VERSION, (void *)SQL_OV_ODBC3, 0);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;
    }

    if (!ctx->dbc)
    {
        chk = SQLAllocHandle(SQL_HANDLE_DBC, ctx->env, &ctx->dbc);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        chk = SQLSetConnectAttr(ctx->dbc, SQL_ATTR_LOGIN_TIMEOUT, (SQLPOINTER)timeout, 0);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        if (tracefile)
        {
            chk = SQLSetConnectAttr(ctx->dbc, SQL_ATTR_TRACE, (SQLPOINTER)1, 0);
            if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
                goto final_error;

            chk = SQLSetConnectAttr(ctx->dbc, SQL_ATTR_TRACEFILE, (SQLPOINTER)tracefile, strlen(tracefile));
            if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
                goto final_error;
        }
    }

    chk = SQLDriverConnect(ctx->dbc, (SQLHWND)NULL, (SQLCHAR *)uri, SQL_NTS, NULL, 0, NULL, SQL_DRIVER_NOPROMPT);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_connect2(abcdk_odbc_t *ctx, const char *product, const char *driver,
                              const char *host, uint16_t port, const char *db,
                              const char *user, const char *pwd,
                              time_t timeout, const char *tracefile)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    char uri[NAME_MAX] = {0};

    assert(product != NULL && driver != NULL && host != NULL && port > 0 &&
           db != NULL && user != NULL && pwd != NULL);

    if (abcdk_strcmp(product, "db2", 0) == 0)
    {
        snprintf(uri, NAME_MAX, "DRIVER=%s;HOSTNAME=%s;PORT=%hu;DATABASE=%s;UID=%s;PWD=%s;PROTOCOL=TCPIP;CONNECTTYPE=1;",
                 driver, host, port, db, user, pwd);
    }
    else if (abcdk_strcmp(product, "oracle", 0) == 0)
    {
        snprintf(uri, NAME_MAX, "DRIVER={%s};DBQ=%s:%hu/%s;UID=%s;PWD=%s;",
                 driver, host, port, db, user, pwd);
    }
    else if (abcdk_strcmp(product, "mysql", 0) == 0)
    {
        /*CHARSET=UTF8;*/
        snprintf(uri, NAME_MAX, "DRIVER=%s;SERVER=%s;TCP_PORT=%hu;DATABASE=%s;UID=%s;PWD=%s;",
                 driver, host, port, db, user, pwd);
    }
    else if (abcdk_strcmp(product, "sqlserver", 0) == 0)
    {
        snprintf(uri, NAME_MAX, "Driver={%s};Server=%s;TCP_PORT=%hu;Database=%s;UID=%s;PWD=%s;",
                 driver, host, port, db, user, pwd);
    }
    else if (abcdk_strcmp(product, "postgresql", 0) == 0)
    {
        /*CHARSET=UTF8;*/
        snprintf(uri, NAME_MAX, "DRIVER=%s;SERVER=%s;TCP_PORT=%hu;DATABASE=%s;UID=%s;PWD=%s;",
                 driver, host, port, db, user, pwd);
    }

    return abcdk_odbc_connect(ctx, uri, timeout, tracefile);

#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_autocommit(abcdk_odbc_t *ctx, int enable)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    long flag;
    SQLRETURN chk;

    assert(ctx != NULL);

    flag = (enable ? SQL_AUTOCOMMIT_ON : SQL_AUTOCOMMIT_OFF);

    chk = SQLSetConnectAttr(ctx->dbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)flag, 0);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;

#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_tran_begin(abcdk_odbc_t *ctx) 
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return abcdk_odbc_autocommit(ctx, 0);

#endif //#ifndef HAVE_UNIXODBC
}


SQLRETURN abcdk_odbc_tran_end(abcdk_odbc_t *ctx, SQLSMALLINT type)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL);

    chk = SQLEndTran(SQL_HANDLE_DBC, ctx->dbc, type);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_tran_commit(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return abcdk_odbc_tran_end(ctx, SQL_COMMIT);
#endif //#ifndef HAVE_UNIXODBC
}


SQLRETURN abcdk_odbc_tran_rollback(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return abcdk_odbc_tran_end(ctx, SQL_ROLLBACK);
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_prepare(abcdk_odbc_t *ctx, const char *sql)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL && sql != NULL);

    /*清理旧的数据集属性.*/
    _abcdk_odbc_free_attr(ctx);

    if (!ctx->stmt)
    {
        chk = SQLAllocHandle(SQL_HANDLE_STMT, ctx->dbc, &ctx->stmt);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        chk = SQLSetStmtAttr(ctx->stmt, SQL_ATTR_CURSOR_TYPE, (SQLPOINTER)SQL_CURSOR_STATIC, 0);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;
    }

    chk = SQLPrepare(ctx->stmt, (SQLCHAR *)sql, SQL_NTS);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_bind_parameter(abcdk_odbc_t *ctx, SQLUSMALLINT ipar, SQLSMALLINT fParamType,
                                    SQLSMALLINT fCType, SQLSMALLINT fSqlType, SQLULEN cbColDef,
                                    SQLSMALLINT ibScale, SQLPOINTER rgbValue, SQLLEN cbValueMax)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL && ipar > 0 && rgbValue != NULL && cbValueMax > 0);

    chk = SQLBindParameter(ctx->stmt, ipar, fParamType, fCType, fSqlType, cbColDef, ibScale, rgbValue, cbValueMax, NULL);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_execute(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL);

    chk = SQLExecute(ctx->stmt);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_finalize(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL);

    /*清理数据集属性.*/
    _abcdk_odbc_free_attr(ctx);

    if (ctx->stmt)
    {
        chk = SQLFreeHandle(SQL_HANDLE_STMT, ctx->stmt);
        if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
            goto final_error;

        ctx->stmt = NULL;
    }

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_exec_direct(abcdk_odbc_t *ctx, const char *sql)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    chk = abcdk_odbc_prepare(ctx, sql);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    chk = abcdk_odbc_execute(ctx);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_affect(abcdk_odbc_t *ctx, SQLLEN *rows)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL && rows != NULL);

    chk = SQLRowCount(ctx->stmt, rows);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_fetch(abcdk_odbc_t *ctx, SQLSMALLINT direction, SQLLEN offset)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL && (direction >= 1 && direction <= 6));

    chk = SQLFetchScroll(ctx->stmt, direction, offset);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_fetch_first(abcdk_odbc_t *ctx)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return abcdk_odbc_fetch(ctx, SQL_FETCH_FIRST, 0);
#endif //#ifndef HAVE_UNIXODBC
}


SQLRETURN abcdk_odbc_fetch_next(abcdk_odbc_t *ctx) 
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    assert(ctx != NULL);

    return abcdk_odbc_fetch(ctx, SQL_FETCH_NEXT, 0);
#endif //#ifndef HAVE_UNIXODBC
}

SQLULEN abcdk_odbc_get_data(abcdk_odbc_t *ctx, SQLSMALLINT column, SQLSMALLINT type,
                              SQLPOINTER buf, SQLULEN max)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return -1;
#else //#ifndef HAVE_UNIXODBC
    SQLLEN StrLen_or_Ind = 0;
    SQLLEN real_len = 0;
    abcdk_object_t *p = NULL;
    SQLRETURN chk;

    assert(ctx != NULL && column >= 0 && buf != NULL && max > 0);

    /*创建数据集属性.*/
    chk = _abcdk_odbc_alloc_attr(ctx);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        return -1;

    p = (abcdk_object_t *)ctx->attr->pptrs[column];
    if (!p)
        return -1;

    /*清除无效的值.*/
    memset(p->pptrs[6], 0, p->sizes[6]);

    chk = SQLGetData(ctx->stmt, (SQLUSMALLINT)(column + 1), type, p->pptrs[6], p->sizes[6], &StrLen_or_Ind);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        return -1;

    if (StrLen_or_Ind > 0)
        real_len = ABCDK_MIN(StrLen_or_Ind, max);
    else
        real_len = ABCDK_MIN(p->sizes[3], max);

    /*复制数据.*/
    memcpy(buf, p->pptrs[6], real_len);

    /*返回长度.*/
    return real_len;
#endif //#ifndef HAVE_UNIXODBC
}

SQLSMALLINT abcdk_odbc_name2index(abcdk_odbc_t *ctx, const char *name)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return -1;
#else //#ifndef HAVE_UNIXODBC
    SQLSMALLINT columns;
    abcdk_object_t *p;
    SQLSMALLINT index;
    SQLRETURN chk;

    assert(ctx != NULL && name != NULL);

    /*创建数据集属性.*/
    chk = _abcdk_odbc_alloc_attr(ctx);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    for (index = ctx->attr->numbers - 1; index >= 0; index--)
    {
        p = (abcdk_object_t *)ctx->attr->pptrs[index];
        if (!p)
        {
            chk = SQL_ERROR;
            goto final_error;
        }

        if (abcdk_strcmp((char *)p->pptrs[0], name, 0) == 0)
            break;
    }

    return index;

final_error:

    return -1;
#endif //#ifndef HAVE_UNIXODBC
}

SQLRETURN abcdk_odbc_error_info(abcdk_odbc_t *ctx, SQLCHAR *Sqlstate, SQLINTEGER *NativeError,
                                SQLCHAR *MessageText, SQLSMALLINT BufferLength)
{
#ifndef HAVE_UNIXODBC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含unixODBC工具."));
    return SQL_ERROR;
#else //#ifndef HAVE_UNIXODBC
    SQLRETURN chk;

    assert(ctx != NULL);

    chk = SQLError(ctx->env, ctx->dbc, ctx->stmt, Sqlstate, NativeError, MessageText, BufferLength, NULL);
    if (_abcdk_odbc_check_return(chk) != SQL_SUCCESS)
        goto final_error;

    return SQL_SUCCESS;

final_error:

    return chk;
#endif //#ifndef HAVE_UNIXODBC
}
