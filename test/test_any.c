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
#include "entry.h"

int abcdk_test_any(abcdk_option_t *args)
{
#if 0
    size_t b = 100;
    int a = ABCDK_CLAMP(0,-(ssize_t)b,(ssize_t)b);

    //int a = ABCDK_MAX((ssize_t)c,(ssize_t)b);

    printf("a=%d\n",a);
#elif 0

    char url[]={"http://asdfasfdasdf.asdfasdf.asdfasdfasf/a/////b/cccc/../ccccc/./././eeeee/?dsdfadf=bbbb&ddsafsd=rrrr#dddd"};

    abcdk_url_abspath(url,0);

    printf("%s\n",url);

    abcdk_object_t *p=NULL;


    p = abcdk_url_fixpath("/bbbb","http://aaaa.com");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/cccc");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com/cccc");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("../bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("../../bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("../../bbbb/../","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("://bbbb","https://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("http://bbbb/?ddddd","http://aaaa.com/cccc?ddsdssdd=ddd&ewerqer=rrrr#ffsfg");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("http://bbbb","/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("/bbbb","/aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","aaaa/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","/aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","aaaa/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);
#elif 0

    abcdk_object_t *p = NULL;

    p = abcdk_url_split("/aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb#");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb#ssss");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("//aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("://aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa:80/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://[22::1]:80/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@aaa.aaa:80/bbb?asfsfd=bbbb&cccc=dddd");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@[22::1]:80/bbb?");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user@aaa.aaa/bbb?asfsfd=bbbb&cccc=dddd#");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("://user@aaa.aaa/bbb?asfsfd=bbbb&cccc=dddd#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
    
    p = abcdk_url_split("://user@aaa.aaa/bbb#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
    
    p = abcdk_url_split("://user@aaa.aaa/#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
        
    p = abcdk_url_split("://user@aaa.aaa#又c");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
#else

    uint32_t a = 0x00123456;

    printf("%08x\n",a);
    printf("%08x\n",abcdk_endian_h_to_b32(a));

#endif 
}