/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/rtp/aac.h"

int abcdk_rtp_aac_revert(const void *data, size_t size, abcdk_comm_queue_t *q, int size_length, ...)
{
    abcdk_comm_message_t *msg;
    int fsize_len[8] = {0},hfsize_len = 0;
    int hlen,flen[100][8] = {0};
    const void *p;
    int chk;

    assert(data != NULL && size > 0 && q != NULL && size_length > 0);

    /*复制AAC数据长度。*/
    fsize_len[0] = size_length;

    /*遍历其它字段长度。*/
    va_list vaptr;
    va_start(vaptr, size_length);
    
    for (int i = 1; i < 7; i++)
    {
        fsize_len[i] = va_arg(vaptr, int);
        if (fsize_len[i] < 0)
            break;

        hfsize_len += fsize_len[i];
    }

    va_end(vaptr);

    /*
     * AU Header Section.
     * 
     * AU Header lengths(2Bytes) + AU Header(Nbits) + AU Header(Nbits) + AU Header(Nbits) + Pidding(bits)
     * 
     * AU Header lengths 不包括自身。
    */
    hlen = abcdk_bloom_read_number(data, size, 0, 16);

    /*最大支持100个封包。*/
    for (int j = 0, pos = 16; j < 100; j++)
    {
        /*不能完成表达一个封包头部，表示结束。*/
        if (pos + hfsize_len > hlen)
            break;

        for (int i = 0; i < 8; i++)
        {
            if (fsize_len[i] < 0)
                break;

            if (fsize_len[i] > 0)
                flen[j][i] = abcdk_bloom_read_number(data, size, pos, fsize_len[i]);

            pos += fsize_len[i];
        }
    }

    p = ABCDK_PTR2VPTR(data, 2 + hlen / 8);

    for (int j = 0; j < 100; j++)
    {
        if (flen[j][0] <= 0)
            break;

        msg = abcdk_comm_message_copy(p, flen[j][0]);
        if (!msg)
            return -1;

        chk = abcdk_comm_queue_push(q, msg, 0);
        if (chk != 0)
        {
            /*加入队列失败，删除消息。*/
            abcdk_comm_message_unref(&msg);
            return -1;
        }

        p = ABCDK_PTR2VPTR(p, flen[j][0]);
    }

    return 1;
}