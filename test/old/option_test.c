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
#include "abcdk/util/option.h"
#include "abcdk/util/getargs.h"


int dump2(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if(deep==0)
        abcdk_tree_fprintf(stderr,deep,node,"OPT\n");
    if(deep==1)
        abcdk_tree_fprintf(stderr,deep,node,"%s\n",node->alloc->pptrs[ABCDK_OPTION_KEY]);
    if(deep==2)
        abcdk_tree_fprintf(stderr,deep,node,"%s\n",node->alloc->pptrs[ABCDK_OPTION_VALUE]);

    return 1;
}

void traversal(abcdk_tree_t *root)
{
    printf("\n-------------------------------------\n");

    abcdk_tree_iterator_t it = {0,dump2,NULL};
    abcdk_tree_scan(root,&it);

    printf("\n-------------------------------------\n");
}

void test1(int argc, char **argv)
{
    abcdk_tree_t *t = abcdk_tree_alloc(NULL);

    abcdk_option_set(t,"-","bbb");
    abcdk_option_set(t,"-","ccc");
    abcdk_option_set(t,"-","fff");
    abcdk_option_set(t,"-","eee");
    abcdk_option_set(t,"-","www");

    assert(abcdk_option_count(t,"-")==5);

    abcdk_option_set(t,"-bbb","123");
    abcdk_option_set(t,"-bbb","456");
    abcdk_option_set(t,"-bbb","789");
    abcdk_option_set(t,"-bbb","543");
    abcdk_option_set(t,"-bbb","854");

    assert(abcdk_option_count(t,"-bbb")==5);

    abcdk_option_set(t,"-ddd",NULL);

    assert(abcdk_option_exist(t,"-ddd"));

    assert(!abcdk_option_exist(t,"-ccc"));

    traversal(t);

    const char* p = abcdk_option_get(t,"-",0,NULL);

    printf("p=%s\n",p);

    const char* p1 = abcdk_option_get(t,"-bbb",1,NULL);

    printf("p1=%s\n",p1);

    const char* p2 = abcdk_option_get(t,"-ccc",1,NULL);

    assert(p2==NULL);

    p2 = abcdk_option_get(t,"-ccc",1,"f");

    assert(p2[0]=='f');

    int s = abcdk_option_fprintf(stderr,t);

    char buf[100] = {0};

    abcdk_option_snprintf(buf,100,t);


    abcdk_getargs(t,argc,argv,"--");

    printf("\n--------------------------------------\n");
    abcdk_option_fprintf(stderr,t);
    printf("\n--------------------------------------\n");
    
    abcdk_getargs_file(t,abcdk_option_get(t,"--test-import",0,NULL),'\n','#',"test-import","--");


    printf("\n--------------------------------------\n");
    abcdk_option_fprintf(stderr,t);
    printf("\n--------------------------------------\n");

    abcdk_option_remove(t,"-bbb");


    printf("\n--------------------------------------\n");
    abcdk_option_fprintf(stderr,t);
    printf("\n--------------------------------------\n");


    abcdk_tree_free(&t);
 
}

void test2(int argc, char **argv)
{
    abcdk_tree_t *t = abcdk_tree_alloc(NULL);

    abcdk_getargs_file(t,"/etc/os-release",'\n',0,NULL,NULL);

   printf("\n--------------------------------------\n");
    abcdk_option_fprintf(stderr,t);
    printf("\n--------------------------------------\n");

    abcdk_tree_free(&t);
}

int main(int argc, char **argv)
{

    test1(argc,argv);

    test2(argc,argv);




    return 0;
}