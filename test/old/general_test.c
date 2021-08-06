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
#include "abcdk-util/general.h"

void test_str()
{
#if 0 
    char buf[] = {"wqerqwer/qewr/q/rq/qwe/wreq//sg/d/dfsf/da"};
    char* buf2 = abcdk_strrep(buf,"qw","CCDD",0);
#elif 0

    char buf[] = {"wqerqwer/qewr/q/rq/qwe/wreq//sg/d/dfsf/da"};
    char* buf2 = abcdk_strrep(buf,"qw","",0);
#else 

    char buf[] = {"wqerqwer/qewr/q/rq/qwe/wreq//sg/d/dfsf/da"};
    char* buf2 = abcdk_strrep(buf,"//","/",0);

#endif 

    printf("%s\n",buf2);

    abcdk_heap_free(buf2);
}


int main(int argc, char **argv)
{

    test_str();
    

    return 0;
}