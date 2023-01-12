/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/rtp/aac.h"


int abcdk_rtp_aac_revert(const void *data, size_t size, abcdk_queue_t *q, int size_bits, int index_bits)
{
    abcdk_receiver_t *msg;
    int au_len,au_size;
    int hlen,flen[200][2] = {0};
    const void *p;
    size_t remain;
    int chk;

    assert(data != NULL && size > 0 && q != NULL && size_bits > 0 && index_bits >=0);
   

    /*
     * AU Header Section.
     * 
     * AU Header lengths(2Bytes) + AU Header(Nbits) + AU Header(Nbits) + AU Header(Nbits) + Pidding(bits)
     * 
     * AU Header lengths 不包括自身。
    */

    /*所有AU分包的头部的总长度(bits)。*/
    au_len = abcdk_bloom_read_number(data, size, 0, 16);

    /*单个包头部的长度(bits)。*/
    au_size = size_bits + index_bits;

    /*仅支持Size Length和Index Length两个可变头部组合。*/
    if (au_len <= 0 || au_len % au_size != 0)
        return -2;

    /*最大支持100个封包。*/
    for (int j = 0, pos = 16; j < 200; j++)
    {
        /*不能完成表达一个封包头部，表示结束。*/
        if (pos + au_size > au_len)
            break;

        if (size_bits > 0)
            flen[j][0] = abcdk_bloom_read_number(data, size, pos, size_bits);

        if (index_bits > 0)
            flen[j][1] = abcdk_bloom_read_number(data, size, pos, index_bits);

        pos += au_size;
    }

    p = ABCDK_PTR2VPTR(data, 2 + abcdk_align(au_len,8) / 8);

    for (int j = 0; j < 200; j++)
    {
        if (flen[j][0] <= 0)
            break;

        msg = abcdk_receiver_alloc(NULL);
        if (!msg)
            return -1;

        /*模拟接收。*/
        abcdk_receiver_append(msg,p,flen[j][0],&remain);

        chk = abcdk_queue_push(q, msg, 0);
        if (chk != 0)
        {
            /*加入队列失败，删除消息。*/
            abcdk_receiver_unref(&msg);
            return -1;
        }

        p = ABCDK_PTR2VPTR(p, flen[j][0]);
    }

    return 1;
}