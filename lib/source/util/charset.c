/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/util/charset.h"

ssize_t abcdk_verify_utf8(const void *data, size_t max)
{
    uint8_t *p = NULL;
    ssize_t cnt = 0;

    assert(data != NULL && max > 0);

    for (size_t i = 0; i < max;)
    {
        p = ABCDK_PTR2U8PTR(data, i);

        if(*p == '\0')
        {
            break;
        }
        else if ((*p & 0x80) == 0x00) // 0xxxxxxx
        {
            cnt += 1;
            i += 1;
        }
        else if ((*p & 0xE0) == 0xC0) // 110xxxxx 10xxxxxx
        {
            if ((i + 2) > max)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            cnt += 2;
            i += 2;
        }
        else if ((*p & 0xF0) == 0xE0) // 1110xxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 3) > max)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            if ((*(p + 2) & 0xc0) != 0x80)
                break;

            cnt += 3;
            i += 3;
        }
        else if ((*p & 0xF8) == 0xF0) // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 4) > max)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            if ((*(p + 2) & 0xc0) != 0x80)
                break;

            if ((*(p + 3) & 0xc0) != 0x80)
                break;

            cnt += 4;
            i += 4;
        }
        else if ((*p & 0xFC) == 0xF8) // 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 5) > max)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            if ((*(p + 2) & 0xc0) != 0x80)
                break;

            if ((*(p + 3) & 0xc0) != 0x80)
                break;

            if ((*(p + 4) & 0xc0) != 0x80)
                break;

            cnt += 5;
            i += 5;
        }
        else if ((*p & 0xFE) == 0xFC) // 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 6) > max)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            if ((*(p + 2) & 0xc0) != 0x80)
                break;

            if ((*(p + 3) & 0xc0) != 0x80)
                break;

            if ((*(p + 4) & 0xc0) != 0x80)
                break;

            if ((*(p + 5) & 0xc0) != 0x80)
                break;

            cnt += 6;
            i += 6;
        }
        else
        {
            break;
        }
    }

    return cnt;
}
