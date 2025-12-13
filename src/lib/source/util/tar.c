/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/tar.h"

int abcdk_tar_num2char(uintmax_t val, char *buf, size_t len)
{
    char *tmpbuf;
    size_t tmplen;
    uintmax_t tmpval;

    assert(buf != NULL && len > 0);

    tmpbuf = buf;
    tmplen = len - 1; // 预留结束字符位置.
    tmpval = val;

    /*尝试8进制格式化输出.*/
    do
    {
        tmpbuf[--tmplen] = '0' + (char)(tmpval & 7);
        tmpval >>= 3;

    } while (tmplen);

    /*有余数时表示空间不足, 尝试base-256编码输出.*/
    if (tmpval)
    {
        tmpbuf = buf;
        tmplen = len; // 不需要保留结束字符位置.
        tmpval = val;

        memset(tmpbuf, 0, tmplen);

        do
        {
            tmpbuf[--tmplen] = (unsigned char)(tmpval & 0xFF);
            tmpval >>= 8;

        } while (tmplen);

        /*有余数时表示空间不足, 返回失败.*/
        if (tmpval)
            return -1;

        /*如果标志位如果被占用, 返回失败.*/
        if (*tmpbuf & '\x80')
            return -1;

        /*设置base-256编码标志.*/
        *tmpbuf |= '\x80';
    }

    return 0;
}

int abcdk_tar_char2num(const char *buf, size_t len, uintmax_t *val)
{
    const char *tmpbuf;
    size_t tmplen;
    uintmax_t *tmpval;
    size_t i;

    assert(buf != NULL && len > 0 && val != NULL);

    tmpbuf = buf;
    tmplen = len;
    tmpval = val;

    /*检测是否为base-256编码.*/
    if (*tmpbuf & '\x80')
    {
        /*解码非标志位的数值.*/
        *tmpval = (tmpbuf[i = 0] & '\x3F');

        /*解码其它数据.*/
        for (i += 1; i < len; i++)
        {
            /*检查是否发生数值溢出.*/
            if (*tmpval > (UINTMAX_MAX >> 8))
                return -1;

            *tmpval <<= 8;
            *tmpval |= (unsigned char)(tmpbuf[i]);
        }
    }
    else
    {
        /*跳过不是8进制的数字字符.*/
        for (i = 0; i < len; i++)
        {
            if (abcdk_isodigit(tmpbuf[i]))
                break;
        }

        /*字符转数值.*/
        for (; i < len; i++)
        {
            /*遇到非8进制的数字符时, 提前终止.*/
            if (!abcdk_isodigit(tmpbuf[i]))
                break;

            /*检查是否发生数值溢出.*/
            if (*tmpval > (UINTMAX_MAX >> 3))
                return -1;

            *tmpval <<= 3;
            *tmpval |= tmpbuf[i] - '0';
        }

        /*如果提前终止, 返回失败.*/
        if (i < len && tmpbuf[i] != '\0')
            return -1;
    }

    return 0;
}

uint32_t abcdk_tar_calc_checksum(abcdk_tar_hdr *hdr)
{
    uint32_t sum = 0;
    int i = 0;

    assert(hdr != NULL);

    for (i = 0; i < 148; ++i)
    {
        sum += ABCDK_PTR2OBJ(uint8_t, hdr, i);
    }

    /*-----跳过checksum(8bytes)字段------*/

    for (i += 8; i < 512; ++i) //...........
    {
        sum += ABCDK_PTR2OBJ(uint8_t, hdr, i);
    }

    sum += 256;

    return sum;
}

uint32_t abcdk_tar_get_checksum(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    /*较验和的字段长度8个字节, 但只有6个数字, 跟着一个NULL(0), 最后一个是空格.*/
    if (abcdk_tar_char2num(hdr->posix.chksum, 7, &val) != 0)
        return -1;

    return val;
}