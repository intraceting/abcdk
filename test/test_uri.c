/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_uri(abcdk_option_t *args)
{
    char src[] = {"http://localhoat:1234/考试1?aaa=考试2&ccc=考试3#aaaa"};
    char dst[1000] = {0};

    size_t slen = strlen(src);
    size_t dlen = 1000;

    ssize_t rs = abcdk_url_encode(src,slen,dst,&dlen,1);

    printf("%s\n",dst);

    char dst2[1000] = {0};

    size_t dlen2 = 1000;

    ssize_t rs2 = abcdk_url_decode(dst,dlen,dst2,&dlen2,1);

    printf("%s\n",dst2);
    
    return 0;
}
