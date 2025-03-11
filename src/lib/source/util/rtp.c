/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/rtp.h"

void abcdk_rtp_header_serialize(const abcdk_rtp_header_t *hdr, void *data, size_t size)
{
    assert(hdr != NULL && data != NULL && size > 0);
    assert(hdr->csrc_len * 4 + 12 <= size);

    abcdk_bloom_write_number(data, size, 0, 2, hdr->version);
    abcdk_bloom_write_number(data, size, 2, 1, hdr->padding);
    abcdk_bloom_write_number(data, size, 3, 1, hdr->extension);
    abcdk_bloom_write_number(data, size, 4, 4, hdr->csrc_len);
    abcdk_bloom_write_number(data, size, 8, 1, hdr->marker);
    abcdk_bloom_write_number(data, size, 9, 7, hdr->payload);
    abcdk_bloom_write_number(data, size, 16, 16, hdr->seq_no);
    abcdk_bloom_write_number(data, size, 32, 32, hdr->timestamp);
    abcdk_bloom_write_number(data, size, 64, 32, hdr->ssrc);

    for (int i = 0; i < hdr->csrc_len; i++)
        abcdk_bloom_write_number(data, size, 96 + (i * 32), 32, hdr->csrc[i]);
}

void abcdk_rtp_header_deserialize(const void *data, size_t size, abcdk_rtp_header_t *hdr)
{
    assert(hdr != NULL && data != NULL && size >= 12);

    hdr->version = abcdk_bloom_read_number(data, size, 0, 2);
    hdr->padding = abcdk_bloom_read_number(data, size, 2, 1);
    hdr->extension = abcdk_bloom_read_number(data, size, 3, 1);
    hdr->csrc_len = abcdk_bloom_read_number(data, size, 4, 4);
    hdr->marker = abcdk_bloom_read_number(data, size, 8, 1);
    hdr->payload = abcdk_bloom_read_number(data, size, 9, 7);
    hdr->seq_no = abcdk_bloom_read_number(data, size, 16, 16);
    hdr->timestamp = abcdk_bloom_read_number(data, size, 32, 32);
    hdr->ssrc = abcdk_bloom_read_number(data, size, 64, 32);

    assert(hdr->csrc_len * 4 + 12 <= size);

    for (int i = 0; i < hdr->csrc_len; i++)
        hdr->csrc[i] = abcdk_bloom_read_number(data, size, 96 + (i * 32), 32);
}


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

        msg = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM,UINT32_MAX,NULL);
        if (!msg)
            return -1;

        /*模拟接收。*/
        abcdk_receiver_append(msg,p,flen[j][0],&remain);

        chk = abcdk_queue_push(q,msg);
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


int abcdk_rtp_h264_revert(const void *data, size_t size, abcdk_queue_t *q)
{
    abcdk_receiver_t *msg;
    int f, nri, type, len;
    int s, e, r, type2;
    const void *p;
    size_t size2,remain;
    int chk;

    assert(data != NULL && size > 0 && q != NULL);

    /*
     * NAL Header，or FU indicator.
     * 
     *  0 1 2 3 4 5 6 7
     * +-+-+-+-+-+-+-+-+
     * |F|NRI|  Type   |
     * +---------------+
    */
    f = abcdk_bloom_read_number(data, size, 0, 1);
    nri = abcdk_bloom_read_number(data, size, 1, 2);
    type = abcdk_bloom_read_number(data, size, 3, 5);

    if(type >= 1 && type <= 12)
    {
        /* 
        * 其它，单个NAL包。
        *
        * NAL Header + data(Nbytes)
        */
        msg = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM,UINT32_MAX,NULL);
        if (!msg)
            return -1;
        
        /*模拟接收数据。*/
        abcdk_receiver_append(msg,data,size,&remain);

        chk = abcdk_queue_push(q,msg);
        if (chk != 0)
        {       
            /*加入队列失败，删除消息。*/
            abcdk_receiver_unref(&msg);
            return -1;
        }
        
        return 1;
    }
    else if (type == 24)
    {
        /* 
        * 24，单时间聚合包(STAP-A)。
        *
        * NAL Header + len(2byte) + data(Nbytes) + ... + len(2byte) + data(Nbytes)
        */

        size2 = size - 1;
        p = ABCDK_PTR2VPTR(data, 1);
        
        while (1)
        {
            /*单个NAL长度，16bits。不包含字段本身。*/
            len = abcdk_bloom_read_number(p, size2, 0, 16);

            /*
            * 递归解包。
            * 
            * 因为RTP设计不支嵌套，这里递归解包是可行的。
            * 但是如果遇到伪造的嵌套数据包，递归深度过高很可能造成异常。
            */
            chk = abcdk_rtp_h264_revert(ABCDK_PTR2VPTR(p, 2), len, q);
            if (chk != 1)
                return -1;

            /*Next*/
            size2 = size2 - (len + 2);
            p = ABCDK_PTR2VPTR(p, len + 2);

            /*没有更多的数据包。*/
            if (size2 <= 0)
                return 1;
        }
    }
    else if (type == 28)
    {
        /*
        * 28，分片单元包(FU-A)。
        * FU indicator + FU Header + data(Ntypes)
        */
        size2 = size - 1;
        p = ABCDK_PTR2VPTR(data, 1);

        /*
         * FU Header.
         *
         *  0 1 2 3 4 5 6 7
         * +-+-+-+-+-+-+-+-+
         * |S|E|R|  Type   |
         * +---------------+
        */
        s = abcdk_bloom_read_number(p, size2, 0, 1);// !0 起始包，0 非起始包。
        e = abcdk_bloom_read_number(p, size2, 1, 1);// !0 结束包，0 非结束包。
        r = abcdk_bloom_read_number(p, size2, 2, 1);
        type2 = abcdk_bloom_read_number(p, size2, 3, 5);

        if (s)
        {
            msg = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM,UINT32_MAX,NULL);
            if (!msg)
                return -1;

            /*模拟接收数据。*/
            abcdk_receiver_append(msg, data, 1, &remain);
            abcdk_receiver_append(msg, ABCDK_PTR2VPTR(p, 1), size2 - 1, &remain);

            /* 还原NAL Header。分片时，原始头type被放在FU Header中。。*/
            p = abcdk_receiver_data(msg,0);
            abcdk_bloom_write_number(ABCDK_PTR2U8PTR(p,0), 1, 3, 5, type2);

            chk = abcdk_queue_push(q, msg);
            if (chk != 0)
            {
                /*加入队列失败，删除消息。*/
                abcdk_receiver_unref(&msg);
                return -1;
            }

            return 0;
        }
        else
        {
            msg = (abcdk_receiver_t *)abcdk_queue_pop(q);
            if (!msg)
                return -1;

            /*拼接数据包。跑过分片包的FU indicator和FU Header。*/
            abcdk_receiver_append(msg, ABCDK_PTR2VPTR(p, 1), size2 - 1, &remain);

            chk = abcdk_queue_push(q,msg);
            if (chk != 0)
            {    
                /*加入队列失败，删除消息。*/
                abcdk_receiver_unref(&msg);
                return -1;
            }

            if (e)
                return 1;
            else
                return 0;
        }
    }
    

    return -2;
}


int abcdk_rtp_hevc_revert(const void *data, size_t size, abcdk_queue_t *q)
{
    abcdk_receiver_t *msg;
    int f, type, lid, tid;
    int s, e, type2;
    const void *p;
    size_t size2,remain;
    int chk;

    assert(data != NULL && size > 0 && q != NULL);

    /*
     * NAL Header，or FU indicator.
     * 
     *  0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
     * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
     * |F| Type      | LayerID   | TID | 
     * +---------------+-+-+-+-+-+-+-+-+
    */

    f = abcdk_bloom_read_number(data, size, 0, 1);
    type = abcdk_bloom_read_number(data, size, 1, 6);
    lid = abcdk_bloom_read_number(data, size, 7, 6);
    tid = abcdk_bloom_read_number(data, size, 13, 3);

    if(type >= 1 && type < 48)
    {
        /* 
        * 其它，单个NAL包。
        *
        * NAL Header + data(Nbytes)
        */
        msg = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM,UINT32_MAX,NULL);
        if (!msg)
            return -1;
        
        /*模拟接收数据。*/
        abcdk_receiver_append(msg, data, size, &remain);

        chk = abcdk_queue_push(q, msg);
        if (chk != 0)
        {       
            /*加入队列失败，删除消息。*/
            abcdk_receiver_unref(&msg);
            return -1;
        }
        
        return 1;
    }
    else if(type == 49)
    {
        /*
        * 49，分片单元包(FU-A)。
        * FU indicator + FU Header + data(Ntypes)
        */

        /*
         * FU Header.
         *
         *  0 1 2 3 4 5 6 7
         * +-+-+-+-+-+-+-+-+
         * |S|E|    Type   |
         * +---------------+
        */
        s = abcdk_bloom_read_number(ABCDK_PTR2VPTR(data, 2), 1, 0, 1);// !0 起始包，0 非起始包。
        e = abcdk_bloom_read_number(ABCDK_PTR2VPTR(data, 2), 1, 1, 1);// !0 结束包，0 非结束包。
        type2 = abcdk_bloom_read_number(ABCDK_PTR2VPTR(data, 2), 1, 2, 6);

        if (s)
        {
            msg = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM,UINT32_MAX,NULL);
            if (!msg)
                return -1;

            /*模拟接收数据。*/
            abcdk_receiver_append(msg, data, 2, &remain);
            abcdk_receiver_append(msg, ABCDK_PTR2VPTR(data, 3), size - 3, &remain);

            /* 还原NAL Header。分片时，原始头type被放在FU Header中。*/
            p = abcdk_receiver_data(msg,0);
            abcdk_bloom_write_number(ABCDK_PTR2U8PTR(p,0), 1, 1, 6, type2);

            chk = abcdk_queue_push(q,msg);
            if (chk != 0)
            {
                /*加入队列失败，删除消息。*/
                abcdk_receiver_unref(&msg);
                return -1;
            }

            return 0;
        }
        else
        {
            msg = (abcdk_receiver_t *)abcdk_queue_pop(q);
            if (!msg)
                return -1;

            /*拼接数据包。跑过分片包的FU indicator和FU Header。*/
            abcdk_receiver_append(msg, ABCDK_PTR2VPTR(data, 3), size - 3, &remain);

            chk = abcdk_queue_push(q,msg);
            if (chk != 0)
            {    
                /*加入队列失败，删除消息。*/
                abcdk_receiver_unref(&msg);
                return -1;
            }

            if (e)
                return 1;
            else
                return 0;
        }
    }
    

    return -2;
}