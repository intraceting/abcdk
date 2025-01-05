/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/video/h2645.h"

ssize_t abcdk_h2645_find_start_code(const void *b, const void *e, int *ksize)
{
    const void *p = NULL;
    size_t i = 0, k = 0;

    assert(b != NULL && e != NULL);
    assert(b <= e);

    p = (uint8_t *)b;

    while (p <= e)
    {
        if (e - p >= 4)
        {
            if (memcmp(p, "\x0\x0\x0\x1", k = 4) == 0)
                break;
        }

        if (e - p >= 3)
        {
            if (memcmp(p, "\x0\x0\x1", k = 3) == 0)
                break;
        }

        p += 1;
        k = 0;
    }

    /*可能未找到。*/
    if (p > e)
        return -1;

    if (ksize && k != 0)
        *ksize = k;

    return p - b;
}

const void *abcdk_h2645_packet_split(void **next, const void *e)
{
    void *p = NULL;
    ssize_t i1 = -1,i2 = -1;
    int ksize = 0;

    assert(next != NULL && e != NULL);
    assert(*next != NULL && *next <= e);

    p = *next;

    i1 = abcdk_h2645_find_start_code(p, e,&ksize);
    if (i1 < 0)
        return NULL;

    i2 = abcdk_h2645_find_start_code(p + i1 + ksize, e, &ksize);
    if(i2 < 0)
        *next = (void*)(e+1);
    else 
        *next = (void*)(p + i1 + ksize + i2);

    return p + i1 + ksize;
}

void abcdk_h2645_mp4toannexb(void *data, size_t size, int len_size)
{
    abcdk_bit_t buf = {0};
    int bits_count;
    uint64_t n;

    assert(data != NULL && size > 0 && len_size > 0 && len_size <= 8);

    /*to bits count.*/
    bits_count = len_size * 8;

    buf.data = (void *)data;
    buf.size = size;
    buf.pos = 0;

next_nal:

    n = abcdk_bit_read(&buf, bits_count);
    if (n <= 0)
        return;

    /*回滚游标。*/
    abcdk_bit_seek(&buf, -bits_count);

    /*替换长度为字流节头标识('01','001','0001',...,'00000001')。*/
    abcdk_bit_write(&buf, bits_count - 8, 0);
    abcdk_bit_write(&buf, 8, 1);

    /*游标跳过实体数据。*/
    abcdk_bit_seek(&buf, (n * 8));

    /*未到末尾，继续下一个NAL。*/
    if (!abcdk_bit_eof(&buf))
        goto next_nal;

    return;
}