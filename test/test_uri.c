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

int abcdk_test_uri(abcdk_tree_t *args)
{
    char src[] = {"http://localhoat:1234/中 文?aaa=bbb&ccc=dddd#aaaa"};
    char dst[1000] = {0};

    size_t slen = strlen(src);
    size_t dlen = 1000;

    ssize_t rs = abcdk_uri_encode(src,slen,dst,&dlen,1);

    printf("%s\n",dst);

    char dst2[1000] = {0};

    size_t dlen2 = 1000;

    ssize_t rs2 = abcdk_uri_decode(dst,dlen,dst2,&dlen2);

    printf("%s\n",dst2);
}