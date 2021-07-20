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
#include "abcdkutil/general.h"
#include "abcdkutil/getargs.h"
#include "abcdkutil/odbc.h"

#if defined(__SQL_H) && defined(__SQLEXT_H)

void _abcdkodbc_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    /*Clear errno.*/
    errno = 0;

    abcdk_proc_basename(name);

#ifdef BUILD_VERSION_DATETIME
    fprintf(stderr, "\n%s Build %s\n", name, BUILD_VERSION_DATETIME);
#endif //BUILD_VERSION_DATETIME

    fprintf(stderr, "\n%s Version %d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR);

    if (only_version)
        return;

    fprintf(stderr, "\nSYNOPSIS:\n");

    fprintf(stderr, "\n%s [ --product < NAME > ] [ OPTIONS ]\n",name);

    fprintf(stderr, "\n%s [ --uri < STRING > ]\n",name);

    fprintf(stderr, "\nOPTIONS:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--verbose\n");
    fprintf(stderr, "\t\tPrint details. default: No\n");

    fprintf(stderr, "\n\t--product < NAME >\n");
    fprintf(stderr, "\t\tProduct name. support: DB2/MYSQL/ORACLE/SQLSERVER/POSTGRESQL\n");

    fprintf(stderr, "\n\t--driver\n");
    fprintf(stderr, "\t\tDriver name. see /etc/odbcinst.ini\n");

    fprintf(stderr, "\n\t--server < ADDRESS >\n");
    fprintf(stderr, "\t\tServer address(IP | Domain). default: localhost\n");

    fprintf(stderr, "\n\t--port < NUMBER >\n");
    fprintf(stderr, "\t\tServer port. default: 50000(DB2)/3306(MYSQL)/1521(ORACLE)/1433(SQLSERVER)/5432(POSTGRESQL)\n");

    fprintf(stderr, "\n\t--db < NAME >\n");
    fprintf(stderr, "\t\tDatabase name. \n");

    fprintf(stderr, "\n\t--uid < NAME >\n");
    fprintf(stderr, "\t\tUser name. \n");

    fprintf(stderr, "\n\t--pwd < PASSWROD >\n");
    fprintf(stderr, "\t\tPassword. \n");

    fprintf(stderr, "\n\t--uri < STRING >\n");
    fprintf(stderr, "\t\tCustom connection string.\n");

    fprintf(stderr, "\n\t--timeout < SECONDS >\n");
    fprintf(stderr, "\t\tTimeout(seconds). default: 30\n");

    fprintf(stderr, "\n\t--trace-file < FILE >\n");
    fprintf(stderr, "\t\tTrace file.\n");
}

void _abcdkodbc_work(abcdk_tree_t *args)
{
    abcdk_odbc_t ctx = {0};
    int verbose = 0;
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

    verbose = (abcdk_option_exist(args,"--verbose")?1:0);
    product = abcdk_option_get(args, "--product", 0, NULL);
    driver = abcdk_option_get(args, "--driver", 0, NULL);
    server = abcdk_option_get(args, "--server", 0, "localhost");
    port = abcdk_option_get_int(args, "--port", 0, 0);
    db = abcdk_option_get(args, "--db", 0, NULL);
    uid = abcdk_option_get(args, "--uid", 0, NULL);
    pwd = abcdk_option_get(args, "--pwd", 0, "");
    uri = abcdk_option_get(args, "--uri", 0, NULL);
    timeout = abcdk_option_get_long(args, "--timeout", 0, 30);
    tracefile = abcdk_option_get(args, "--trace-file", 0, NULL);

    /*Clear errno.*/
    errno = 0;

    if (uri && *uri)
    {
        chk = abcdk_odbc_connect(&ctx, uri, timeout, tracefile);
        goto final;
    }

    if (!product || !*product)
    {
        if(verbose)
            syslog(LOG_ERR, "'--product NAME' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (abcdk_strcmp(product, "DB2", 0) != 0 &&
        abcdk_strcmp(product, "MYSQL", 0) != 0 &&
        abcdk_strcmp(product, "ORACLE", 0) != 0 &&
        abcdk_strcmp(product, "SQLSERVER", 0) != 0 &&
        abcdk_strcmp(product, "POSTGRESQL", 0) != 0)
    {
        if(verbose)
            syslog(LOG_ERR, "'--product NAME' is one of them in DB2 MYSQL ORACLE SQLSERVER POSTGRESQL.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (!driver || !*driver)
    {
        if(verbose)
            syslog(LOG_ERR, "'--driver NAME' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (!server || !*server)
    {
        if(verbose)
            syslog(LOG_ERR, "'--server ADDRESS' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
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
        if(verbose)
            syslog(LOG_ERR, "'--port NUMBER' range is 1~65535.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (!db || !*db)
    {
        if(verbose)
            syslog(LOG_ERR, "'--db NAME' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    if (!uid || !*uid)
    {
        if(verbose)
            syslog(LOG_ERR, "'--uid NAME' cannot be omitted.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL, final);
    }

    chk = abcdk_odbc_connect2(&ctx,product,driver,server,port,db,uid,pwd,timeout,tracefile);

final:

    abcdk_odbc_disconnect(&ctx);

    fprintf(stderr,"%d\n",((chk == SQL_SUCCESS)?0:1));

}

#endif //defined(__SQL_H) && defined(__SQLEXT_H)


int main(int argc, char **argv)
{
#if defined(__SQL_H) && defined(__SQLEXT_H)

    abcdk_tree_t *args;

    setlocale(LC_ALL,"");

    args = abcdk_tree_alloc3(1);
    if (!args)
        goto final;

    abcdk_getargs(args, argc, argv, "--");
    //abcdk_option_fprintf(stderr, args);

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkodbc_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkodbc_print_usage(args, 1);
    }
    else
    {
        _abcdkodbc_work(args);
    }

final:

    abcdk_tree_free(&args);

#else
    
    fprintf(stderr, "The toolkit needs to be recompiled to support ODBC.\n");
    errno = EPERM;

#endif //defined(__SQL_H) && defined(__SQLEXT_H)

    return errno;
}