/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/http/reply.h"

int abcdk_http_reply_chunked(abcdk_comm_node_t *node, abcdk_object_t *data)
{
    int chk;

    assert(node != NULL && data != NULL);

    chk = abcdk_comm_post_format(node, 20, "%x\r\n", data->sizes[0]);
    if (chk != 0)
        return -1;

    chk = abcdk_comm_post(node,data);
    if (chk != 0)
        return -1;

    chk = abcdk_comm_post_buffer(node, "\r\n", 2);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_http_reply_chunked_buffer(abcdk_comm_node_t *node, const void *data, size_t size)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_post_format(node, 20, "%x\r\n", size);
    if (chk != 0)
        return -1;

    if (data != NULL && size > 0)
    {
        chk = abcdk_comm_post_buffer(node, data, size);
        if (chk != 0)
            return -1;
    }

    chk = abcdk_comm_post_buffer(node, "\r\n", 2);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_http_reply_chunked_vformat(abcdk_comm_node_t *node, int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && fmt != NULL && max > 0);

    obj = abcdk_object_alloc2(max);
    if (!obj)
        return -1;

    chk = vsnprintf(obj->pstrs[0], max, fmt, ap);
    if (chk <= 0)
        goto final_error;

    /*修正格式化后的数据长度。*/
    obj->sizes[0] = chk;

    chk = abcdk_http_reply_chunked(node, obj);
    if (chk == 0)
        return 0;

final_error:

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return -1;
}

int abcdk_http_reply_chunked_format(abcdk_comm_node_t *node, int max, const char *fmt, ...)
{
    int chk;

    assert(node != NULL && fmt != NULL && max > 0);


    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_http_reply_chunked_vformat(node, max, fmt, ap);
    va_end(ap);

    return chk;
}
