/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/rtp/hevc.h"

int abcdk_rtp_hevc_revert(const void *data, size_t size, abcdk_queue_t *q)
{
    abcdk_message_t *msg;
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
        msg = abcdk_message_copy(data, size);
        if (!msg)
            return -1;
        
        /*模拟接收数据。*/
        abcdk_message_reset(msg,size);

        chk = abcdk_queue_push(q, msg, 0);
        if (chk != 0)
        {       
            /*加入队列失败，删除消息。*/
            abcdk_message_unref(&msg);
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
            msg = abcdk_message_alloc(size-1);
            if (!msg)
                return -1;

            /*模拟接收数据。*/
            abcdk_message_recv(msg, data, 2, &remain);
            abcdk_message_recv(msg, ABCDK_PTR2VPTR(data, 3), size - 3, &remain);

            /* 还原NAL Header。分片时，原始头type被放在FU Header中。*/
            p = abcdk_message_data(msg);
            abcdk_bloom_write_number(ABCDK_PTR2U8PTR(p,0), 1, 1, 6, type2);

            chk = abcdk_queue_push(q, msg, 0);
            if (chk != 0)
            {
                /*加入队列失败，删除消息。*/
                abcdk_message_unref(&msg);
                return -1;
            }

            return 0;
        }
        else
        {
            msg = (abcdk_message_t *)abcdk_queue_pop(q, 0);
            if (!msg)
                return -1;

            /*增量扩展缓存。*/
            abcdk_message_expand(msg, size - 3);

            /*拼接数据包。跑过分片包的FU indicator和FU Header。*/
            abcdk_message_recv(msg, ABCDK_PTR2VPTR(data, 3), size - 3, &remain);

            chk = abcdk_queue_push(q, msg, 0);
            if (chk != 0)
            {    
                /*加入队列失败，删除消息。*/
                abcdk_message_unref(&msg);
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