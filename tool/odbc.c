/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/database/odbc.h"
#include "entry.h"

#if defined(__SQL_H) && defined(__SQLEXT_H)

typedef struct _abcdk_odbc
{
    int errcode;
    abcdk_option_t *args;

}abcdk_odbc_t;


void _abcdk_odbc_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的ODBC连接测试工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--product < NAME >\n");
    fprintf(stderr, "\t\t产品名称。支持: DB2/MYSQL/ORACLE/SQLSERVER/POSTGRESQL\n");

    fprintf(stderr, "\n\t--driver\n");
    fprintf(stderr, "\t\t驱动名称。见：/etc/odbcinst.ini\n");

    fprintf(stderr, "\n\t--server < ADDRESS >\n");
    fprintf(stderr, "\t\t服务地址(域名或IP)。默认: localhost\n");

    fprintf(stderr, "\n\t--port < NUMBER >\n");
    fprintf(stderr, "\t\t服务端口。默认: 50000(DB2)/3306(MYSQL)/1521(ORACLE)/1433(SQLSERVER)/5432(POSTGRESQL)\n");

    fprintf(stderr, "\n\t--db < NAME >\n");
    fprintf(stderr, "\t\t数据库名称。 \n");

    fprintf(stderr, "\n\t--uid < NAME >\n");
    fprintf(stderr, "\t\t数据库账户。\n");

    fprintf(stderr, "\n\t--pwd < PASSWROD >\n");
    fprintf(stderr, "\t\t数据库账户密码。 \n");

    fprintf(stderr, "\n\t--uri < STRING >\n");
    fprintf(stderr, "\t\t自定义连接字符串。\n");

    fprintf(stderr, "\n\t--timeout < SECONDS >\n");
    fprintf(stderr, "\t\t超时(秒)。默认: 30\n");

    fprintf(stderr, "\n\t--trace-file < FILE >\n");
    fprintf(stderr, "\t\t指定根踪文件。\n");
    
    ABCDK_ERRNO_AND_RETURN0(0);
}

void _abcdk_odbc_work(abcdk_odbc_t *ctx)
{
    abcdk_odbc_t *odbc = abcdk_odbc_alloc(0);
    const char *product = NULL;
    const char *driver = NULL;
    const char *server = NULL;
    uint16_t port = 0;
    const char *db = NULL;
    const char *uid = NULL;
    const char *pwd = NULL;
    const char *uri = NULL;
    time_t timeout = 0;
    const char *tracefile = NULL;
    SQLRETURN chk = SQL_ERROR;

    product = abcdk_option_get(ctx->args, "--product", 0, NULL);
    driver = abcdk_option_get(ctx->args, "--driver", 0, NULL);
    server = abcdk_option_get(ctx->args, "--server", 0, "localhost");
    port = abcdk_option_get_int(ctx->args, "--port", 0, 0);
    db = abcdk_option_get(ctx->args, "--db", 0, NULL);
    uid = abcdk_option_get(ctx->args, "--uid", 0, NULL);
    pwd = abcdk_option_get(ctx->args, "--pwd", 0, "");
    uri = abcdk_option_get(ctx->args, "--uri", 0, NULL);
    timeout = abcdk_option_get_long(ctx->args, "--timeout", 0, 30);
    tracefile = abcdk_option_get(ctx->args, "--trace-file", 0, NULL);

    /*优先检查自定义是否可用。*/
    if (uri && *uri)
    {
        chk = abcdk_odbc_connect(odbc, uri, timeout, tracefile);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (!product || !*product)
    {
        fprintf(stderr, "'--product NAME' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (abcdk_strcmp(product, "DB2", 0) != 0 &&
        abcdk_strcmp(product, "MYSQL", 0) != 0 &&
        abcdk_strcmp(product, "ORACLE", 0) != 0 &&
        abcdk_strcmp(product, "SQLSERVER", 0) != 0 &&
        abcdk_strcmp(product, "POSTGRESQL", 0) != 0)
    {
        fprintf(stderr, "'--product NAME' 仅支持DB2，MYSQL，ORACLE，SQLSERVER，POSTGRESQL。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (!driver || !*driver)
    {
        fprintf(stderr, "'--driver NAME' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (!server || !*server)
    {
        fprintf(stderr, "'--server ADDRESS' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (port == 0 || port == 65536)
    {
        if(abcdk_strcmp(product,"DB2",0)==0)
            port = 50000;
        else if(abcdk_strcmp(product,"MYSQL",0)==0)
            port = 3306;
        else if(abcdk_strcmp(product,"ORACLE",0)==0)
            port = 1521;
        else if(abcdk_strcmp(product,"SQLSERVER",0)==0)
            port = 1433;
        else if(abcdk_strcmp(product,"POSTGRESQL",0)==0)
            port = 5432;
    }

    if (port == 0 || port == 65536)
    {
        fprintf(stderr, "'--port NUMBER' 范围在1~65535之间(包含)。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (!db || !*db)
    {
        fprintf(stderr, "'--db NAME' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (!uid || !*uid)
    {
        fprintf(stderr, "'--uid NAME' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    chk = abcdk_odbc_connect2(odbc,product,driver,server,port,db,uid,pwd,timeout,tracefile);
    if(chk != SQL_SUCCESS)
    {
        fprintf(stderr, "连接失败，超时或参数错误。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

final:

    abcdk_odbc_disconnect(odbc);
    abcdk_odbc_free(&odbc);
}

#endif //defined(__SQL_H) && defined(__SQLEXT_H)


int abcdk_tool_odbc(abcdk_option_t *args)
{
#if defined(__SQL_H) && defined(__SQLEXT_H)

    abcdk_odbc_t ctx = {0};
    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_odbc_print_usage(ctx.args);
    }
    else
    {
        _abcdk_odbc_work(&ctx);
    }

    return ctx.errcode;

#else //defined(__SQL_H) && defined(__SQLEXT_H)

    fprintf(stderr, "当前构建版本未包含此工具。\n");
    return EPERM;

#endif //defined(__SQL_H) && defined(__SQLEXT_H)

}