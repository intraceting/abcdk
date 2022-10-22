/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/rtp/h264.h"

int abcdk_rtp_h264_revert(const void *data, size_t size, abcdk_comm_queue_t *q)
{
    abcdk_comm_message_t *msg;
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

    
    if (type <= 23)
    {
        /* 
        * 1~23，单个NAL包。
        *
        * NAL Header + data(Nbytes)
        */
        msg = abcdk_comm_message_copy(data, size);
        if (!msg)
            return -1;
        
        /*模拟接收数据。*/
        abcdk_comm_message_reset(msg,size);

        chk = abcdk_comm_queue_push(q, msg, 0);
        if (chk == 0)
            return 1;

        /*加入队列失败，删除消息。*/
        abcdk_comm_message_unref(&msg);
        return -1;
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
            if (s <= 0)
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
            msg = abcdk_comm_message_alloc(size2);
            if (!msg)
                return -1;

            /*模拟接收数据。*/
            abcdk_comm_message_recv2(msg, p, size2, &remain);

            /* 还原NAL Header。分片时，原始头被拆成两部分(0~3,4~7)。*/
            p = abcdk_comm_message_data(msg);
            abcdk_bloom_write_number((uint8_t*)p, 1, 0, 1, f);
            abcdk_bloom_write_number((uint8_t*)p, 1, 1, 2, nri);

            chk = abcdk_comm_queue_push(q, msg, 0);
            if (chk == 0)
                return 0;

            /*加入队列失败，删除消息。*/
            abcdk_comm_message_unref(&msg);
            return -1;
        }
        else
        {
            msg = abcdk_comm_queue_pop(q, 0);
            if (!msg)
                return -1;

            /*增量扩展缓存。*/
            abcdk_comm_message_expand(msg, ABCDK_MAX(1500, size2 - 1));

            /*
             * 拼接数据包。跑过分片包的FU indicator和FU Header。
            */
            abcdk_comm_message_recv2(msg, ABCDK_PTR2VPTR(p, 1), size2 - 1, &remain);

            chk = abcdk_comm_queue_push(q, msg, 0);
            if (chk == 0)
                return 0;

            /*加入队列失败，删除消息。*/
            abcdk_comm_message_unref(&msg);
            return -1;

            if (e)
                return 1;
            else
                return 0;
        }
    }

    return -2;
}