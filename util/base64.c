/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/base64.h"

static char _abcdk_base64_encode_table(int n)
{
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "abcdefghijklmnopqrstuvwxyz"
           "0123456789"
           "+/="[n];
}

static int _abcdk_base64_decode_table(char c)
{
    if (c == '+')
        return 62;
    else if (c == '/')
        return 63;
    else if (c <= '9')
        return (int)(c - '0' + 52);
    else if (c == '=')
        return 64;
    else if (c <= 'Z')
        return (int)(c - 'A');
    else if (c <= 'z')
        return (int)(c - 'a' + 26);
    return 64;
}

#define ABCDK_BASE64_ENCODE_LEN(L) (((L) / 3 * 4 + (((L) % 3 == 0) ? 0 : 4)))

#define ABCDK_BASE64_DECODE_LEN(L, C2, C1) ((((L)-4) / 4 * 3) + (((C2) == '=' ? 1 : ((C1) == '=' ? 2 : 3))))

ssize_t abcdk_base64_encode(const uint8_t *src, size_t slen, char *dst, size_t dmaxlen)
{
    size_t remain = 0;
    size_t formal = 0;
    uint32_t mark = 0;
    size_t dlen = 0;

    assert(src != NULL && slen > 0);

    if (dst == NULL)
        return ABCDK_BASE64_ENCODE_LEN(slen);

    assert(ABCDK_BASE64_ENCODE_LEN(slen) >= dmaxlen);

    /*原文长度除以3的余数*/
    remain = slen % 3;

    /*原文长度减去除以3的余数。*/
    formal = slen - remain;

    mark = 0;
    dlen = 0;

    /*1:for FORMAL*/
    for (size_t n = 0; n < formal; n += 3)
    {
        /*
         * three bytes
         * 10101010 01010101 00001111
         * 
         */

        /*1: 00000000 00000000 00000000 10101010 */
        mark = src[n];
        /*2: 00000000 00000000 10101010 01010101*/
        mark = (mark << 8 | src[n + 1]);
        /*3: 00000000 10101010 01010101 00001111*/
        mark = (mark << 8 | src[n + 2]);

        /*six bits*/
        /*1: 00000000 '101010'10 01010101 00001111 & 00FC0000 = 101010 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x00FC0000) >> 18);
        /*2: 00000000 101010'10 0101'0101 00001111 & 0003F000 = 100101 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x0003F000) >> 12);
        /*3: 00000000 10101010 0101'0101 00'001111 & 00000FC0 = 0101 00 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x00000FC0) >> 6);
        /*4: 00000000 10101010 01010101 00'001111' & 0000003F = 001111 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0X0000003F) >> 0);
    }

    /*2:for REMAIN*/

    /*Remain two bytes*/
    if (remain == 2)
    {
        /* 
         * only two bytes
         * 10101010 01010101 
         * 
         */

        /*1: 00000000 00000000 00000000 10101010 */
        mark = src[formal];
        /*2: 00000000 00000000 10101010 01010101 */
        mark = (mark << 8 | src[formal + 1]);
        /*3: 00000000 10101010 01010101 00000000 (fill 0)*/
        mark = (mark << 8 | 0);

        /*six bits*/
        /*1: 00000000 '101010'10 01010101 00000000 & 00FC0000 = 101010 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x00FC0000) >> 18);
        /*2: 00000000 101010'10 0101'0101 00000000 & 0003F000 = 10 0101 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x0003F000) >> 12);
        /*3: 00000000 10101010 0101'0101 00'000000  & 00000FC0 = 0101 00 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x00000FC0) >> 6);
        /*4: TAIL to fill '=' */
        dst[dlen++] = '=';
    }

    /*Remain one byte*/
    if (remain == 1)
    {
        /*
         * one byte
         * 10101010
         * 
         */

        /*1:10101010 */
        mark = src[formal];
        /*2: 00000000 00000000 10101010 00000000 (fill 0) */
        mark = (mark << 8 | 0);
        /*3: 00000000 10101010 00000000 00000000 (fill 0 0) */
        mark = (mark << 8 | 0);

        /*six bits*/
        /*1: 00000000 '101010'10 00000000 00000000  & 00FC0000 = 101010 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x00FC0000) >> 18);
        /*2: 00000000 101010'10 0000'0000 00000000  & 0000000F = 10 0000 */
        dst[dlen++] = _abcdk_base64_encode_table((mark & 0x0003F000) >> 12);
        /*3: TAIL to fill '=' */
        dst[dlen++] = '=';
        /*4: TAIL to fill '=' */
        dst[dlen++] = '=';
    }

    return dlen;
}

ssize_t abcdk_base64_decode(const char *src, size_t slen, uint8_t *dst, size_t dmaxlen)
{
    size_t formal = 0;
    uint32_t mark = 0;
    size_t dlen = 0;

    assert(src != NULL && slen >= 4 && (slen % 4) == 0);

    if (dst == NULL)
        return ABCDK_BASE64_DECODE_LEN(slen, src[slen - 2], src[slen - 1]);

    assert(ABCDK_BASE64_DECODE_LEN(slen, src[slen - 2], src[slen - 1]) >= dmaxlen);

    /*原文长度减去4。*/
    formal = slen - 4;

    mark = 0;
    dlen = 0;

    for (long n = 0; n < formal; n += 4)
    {

        /* 
         * four bytes
         * a = 26,b = 27,c = 28,d = 28  (in decode table)
         * 
        */

        /*1: a  */
        mark = _abcdk_base64_decode_table(src[n]);
        /*2: a b */
        mark = (mark << 6 | _abcdk_base64_decode_table(src[n + 1]));
        /*3: a b c */
        mark = (mark << 6 | _abcdk_base64_decode_table(src[n + 2]));
        /*4: a b c d */
        mark = (mark << 6 | _abcdk_base64_decode_table(src[n + 3]));

        //three bytes
        /*1: */
        dst[dlen++] = (uint8_t)(mark >> 16 & 0x000000FF);
        /*2: */
        dst[dlen++] = (uint8_t)(mark >> 8 & 0x000000FF);
        /*3: */
        dst[dlen++] = (uint8_t)(mark >> 0 & 0x000000FF);
    }

    /*
     * TAIL four bytes
     * a b c = ; a b = = 
    */

    /*1:   */
    mark = _abcdk_base64_decode_table(src[formal]);
    /*2:  */
    mark = (mark << 6 | _abcdk_base64_decode_table(src[formal + 1]));
    /*3:   */
    if (src[formal + 2] != '=')
        mark = (mark << 6 | _abcdk_base64_decode_table(src[formal + 2]));
    else
        mark = (mark << 6);
    //4:
    if (src[formal + 3] != '=')
        mark = (mark << 6 | _abcdk_base64_decode_table(src[formal + 3]));
    else
        mark = (mark << 6);

    /*
     * only valid bytes
    */

    /*1: */
    dst[dlen++] = (uint8_t)(mark >> 16 & 0x000000FF);
    /*2: */
    if (src[formal + 2] != '=')
        dst[dlen++] = (uint8_t)(mark >> 8 & 0x000000FF);
    /*3: */
    if (src[formal + 3] != '=')
        dst[dlen++] = (uint8_t)(mark >> 0 & 0x000000FF);

    return dlen;
}