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

    char url[]={"http://asdfasfdasdf.asdfasdf.asdfasdfasf/a/////b/cccc/../ccccc/./././eeeee/"};

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

    p = abcdk_url_fixpath("bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("http://bbbb","http://aaaa.com/cccc");
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
#elif 1

    abcdk_object_t *p = NULL;

    p = abcdk_url_split("/aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("//aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("://aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa:80/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://[22::1]:80/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://ccc:ddd@aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://ccc:ddd@aaa.aaa:80/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://ccc:ddd@[22::1]:80/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://ccc@aaa.aaa/bbb");
    for (int i = ABCDK_URL_SCHEME; i <= ABCDK_URL_AUTH; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
    

#endif 
}