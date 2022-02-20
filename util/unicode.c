/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "util/unicode.h"

ssize_t abcdk_verify_utf8(const void *data, size_t len)
{
    uint8_t *p = NULL;
    ssize_t cnt = 0;

    assert(data != NULL && len > 0);

    for (size_t i = 0; i < len;)
    {
        p = ABCDK_PTR2U8PTR(data, i);

        if ((*p & 0x80) == 0x00) // 0xxxxxxx
        {
            cnt += 1;
            i += 1;
        }
        else if ((*p & 0xE0) == 0xC0) // 110xxxxx 10xxxxxx
        {
            if ((i + 2) > len)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            cnt += 2;
            i += 2;
        }
        else if ((*p & 0xF0) == 0xE0) // 1110xxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 3) > len)
                break;

            if ((*(p + 1) & 0xc0) != 0x80)
                break;

            if ((*(p + 2) & 0xc0) != 0x80)
                break;

            cnt += 3;
            i += 3;
        }
        else if ((*p & 0xF8) == 0xf0) // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        {
            if ((i + 4) > len)
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
        else
        {
            break;
        }
    }

    return cnt;
}

ssize_t abcdk_verify_gbk(const void *data,size_t len)
{
    uint8_t *p = NULL;
    ssize_t cnt = 0;

    assert(data != NULL && len > 0);

    for (size_t i = 0; i < len;)
    {
        p = ABCDK_PTR2U8PTR(data, i);

        if ((*p & 0x80) == 0x00) // 0x00–-x7F
        {
            cnt += 1;
            i += 1;
        }
        else if (*p >= 0x81 && *p <= 0xFE) //0x81-0xFE
        {
             if ((i + 2) > len)
                break;

            if (*(p + 1) < 0x40 || *(p + 1) > 0xFE || *(p + 1) == 0x7F) //0x40-0xFE(No 0x7F)
                break;

            cnt += 2;
            i += 2;
        }
        else
        {
            break;
        }
    }

    return cnt;
}
