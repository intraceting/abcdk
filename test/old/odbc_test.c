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
#include "abcdk/getargs.h"
#include "abcdk/odbc.h"

#ifdef __SQLEXT_H


void test_insert(abcdk_odbc_t *o)
{


    assert(abcdk_odbc_tran_begin(o) == SQL_SUCCESS);

    time_t begin = time(NULL);

    for(time_t i = begin+1;i<begin+1000;i++)
    {
        char sql[200] = {0};
        sprintf(sql,"insert into LTR_TASK_TRACE(ID,TID,SID) VALUES('%ld','我们','他们');",i);
        SQLRETURN chk = abcdk_odbc_exec_direct(o,sql);

        assert(chk==SQL_SUCCESS);
    }

    assert(abcdk_odbc_tran_commit(o) == SQL_SUCCESS);
}

void test_select(abcdk_odbc_t *o)
{
    assert(abcdk_odbc_autocommit(o,1) == SQL_SUCCESS);

    SQLRETURN chk = abcdk_odbc_prepare(o,"select * from LTR_TASK_TRACE where CAST(id AS SIGNED) > ?;");

  //  SQLRETURN chk = abcdk_odbc_prepare(o,"select * from LTR_TASK_TRACE;");

    assert(chk == SQL_SUCCESS);

    size_t id = 1000;

    chk = SQLBindParameter(o->stmt, 1, SQL_PARAM_INPUT, SQL_C_SBIGINT, SQL_NUMERIC, 0, 0,
                                     (SQLPOINTER)&id, sizeof(size_t), NULL);

    assert(chk == SQL_SUCCESS);

    chk = abcdk_odbc_execute(o);

    assert(chk == SQL_SUCCESS || chk == SQL_NO_DATA);

    chk = abcdk_odbc_fetch_first(o);

    assert(chk == SQL_SUCCESS || chk == SQL_NO_DATA);

    while(chk == SQL_SUCCESS)
    {
        char id[100]={0};
        char tid[100]={0};
        char sid[100]={0};
        char ctime[100]={0};

        abcdk_odbc_get_data(o,abcdk_odbc_name2index(o,"id"),SQL_C_CHAR,id,100,NULL);

        SQLCHAR state=0;
        SQLINTEGER native =0;
        char msg[100]={0};
        SQLError(o->env,o->dbc,o->stmt,&state,&native,msg,100,NULL);

        printf("state=%hhu,native=%d,msg=%s\n",state,native,msg);

        abcdk_odbc_get_data(o,abcdk_odbc_name2index(o,"tid"),SQL_C_CHAR,tid,100,NULL);
        abcdk_odbc_get_data(o,abcdk_odbc_name2index(o,"sid"),SQL_C_CHAR,sid,100,NULL);
        abcdk_odbc_get_data(o,abcdk_odbc_name2index(o,"ctime"),SQL_C_CHAR,ctime,100,NULL);

        printf("%s,%s,%s,%s\n",id,tid,sid,ctime);

        chk = abcdk_odbc_fetch_next(o);
    }

}


int main(int argc, char **argv)
{
    setlocale(LC_ALL,"");

    abcdk_tree_t *t = abcdk_tree_alloc(NULL);

    abcdk_getargs(t,argc,argv,"--");

    abcdk_odbc_t o = {0};

    const char *tracefile = "/tmp/mysql-trace.log";

 //   assert(abcdk_odbc_connect(&o,argv[1],30,tracefile) == SQL_SUCCESS);

    assert(abcdk_odbc_connect2(&o,
                              abcdk_option_get(t, "--product", 0, ""),
                              abcdk_option_get(t, "--driver", 0, ""),
                              abcdk_option_get(t, "--host", 0, ""),
                              abcdk_option_get_int(t, "--port", 0, 1234),
                              abcdk_option_get(t, "--db", 0, ""),
                              abcdk_option_get(t, "--user", 0, ""),
                              abcdk_option_get(t, "--pwd", 0, ""),
                              30, 
                              tracefile) == SQL_SUCCESS);

    test_select(&o);

    test_insert(&o);

    test_select(&o);

    abcdk_odbc_finalize(&o);


    assert(abcdk_odbc_disconnect(&o) == SQL_SUCCESS);

    abcdk_tree_free(&t);

    return 0;
}

#else

int main(int argc, char **argv)
{
    
 
    return 0;
}

#endif //__SQLEXT_H