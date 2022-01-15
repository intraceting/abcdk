/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "util/blockio.h"


ssize_t abcdk_block_read(int fd, void *data, size_t size,abcdk_buffer_t *buf)
{
    ssize_t rsize = 0;
    size_t rsize2 = 0;

    assert(fd >= 0 && data != NULL && size > 0);

    if (buf)
    {
        assert(buf->data != NULL && buf->size > 0);

        while (rsize < size)
        {
            if (buf->wsize > 0)
            {
                /*缓存有数据，先从缓存读取。*/
                rsize2 = abcdk_buffer_read(buf, ABCDK_PTR2PTR(void, data, rsize), size - rsize);
                if (rsize2 <= 0)
                    break;

                /*累加读取长度。*/
                rsize += rsize2;

                /*吸收已经读取的缓存数据。*/
                abcdk_buffer_drain(buf);
            }
            else if (buf->wsize == 0 || (size - rsize) < buf->size)
            {
                /*
                 * 满足以下两个条件之一，则先读取到缓存空间。
                 * 1：缓存无数据。
                 * 2：待读数据小于缓存空间。
                */
                rsize2 = abcdk_buffer_import_atmost(buf, fd, buf->size);
                if (rsize2 <= 0)
                    break;
            }
            else
            {
                assert((size - rsize) >= buf->size);

                /*
                 * 同时满足以下两个条件，把数据直接从文件中读取。
                 * 1：缓存无数据。
                 * 2：待读取数据大于缓存空间。
                */
                rsize2 = abcdk_read(fd, ABCDK_PTR2PTR(void, data, rsize), buf->size);
                if (rsize2 <= 0)
                    break;

                /*累加读取长度。*/
                rsize += rsize2;

                /*带缓存的读取，每次必须读取相同的大小。*/
                if (rsize2 != buf->size)
                    break;
            }
        }
    }
    else
    {
        /*无缓存空间，直接从文件读取。*/
        rsize = abcdk_read(fd, data, size);
    }

    return rsize;
}

ssize_t abcdk_block_write(int fd, const void *data, size_t size,abcdk_buffer_t *buf)
{
    ssize_t wsize = 0;
    ssize_t wsize2 = 0;

    assert(fd >= 0 && data != NULL && size > 0);
 
    if (buf)
    {
        assert(buf->data != NULL && buf->size > 0);

        while (wsize < size)
        {
            if (buf->wsize > 0 && buf->wsize == buf->size)
            {
                /*缓存空间已满，先把缓存数据导出到文件。*/
                wsize2 = abcdk_buffer_export_atmost(buf, fd, buf->size);
                if (wsize2 <= 0)
                    break;

                /*吸收已经导出(已经写入到文件)的缓存数据。*/
                abcdk_buffer_drain(buf);
            }
            else if (buf->wsize > 0 || (size - wsize) < buf->size)
            {
                /* 
                 * 满足以下两个条件之一，则先把数据写进缓存空间。
                 * 1：缓存有数据，但未满。
                 * 2：缓存无数据，但是待写入数据小于缓存空间。
                 */
                wsize2 = abcdk_buffer_write(buf, ABCDK_PTR2PTR(void, data, wsize), size - wsize);
                if (wsize2 <= 0)
                    break;

                /*累加写入长度。*/
                wsize += wsize2;
            }
            else
            {
                assert((size - wsize) >= buf->size);

                /*
                 * 同时满足以下两个条件，把数据直接写入到文件。
                 * 1：缓存无数据。
                 * 2：待写入数据大于缓存空间。
                */
                wsize2 = abcdk_write(fd, ABCDK_PTR2PTR(void, data, wsize), buf->size);
                if (wsize2 <= 0)
                    break;

                /*累加写入长度。*/
                wsize += wsize2;
            }
        }
    }
    else
    {
        /*无缓存空间，直接写入文件。*/
        wsize = abcdk_write(fd, data, size);
    }

    return wsize;
}

int abcdk_block_write_trailer(int fd, uint8_t stuffing,abcdk_buffer_t *buf)
{
    ssize_t wsize2 = 0;

    assert(fd >= 0);

    /*无缓存。*/
    if (!buf)
        return 0;

    assert(buf->data != NULL && buf->size > 0);

    /*缓存无数据。*/
    if (buf->wsize == 0)
        return 0;

    /*缓存有数据，先用填充物填满缓存空间。*/
    abcdk_buffer_fill(buf, stuffing);

    /*把缓存数据导出到文件。*/
    wsize2 = abcdk_buffer_export_atmost(buf, fd, buf->size);
    if (wsize2 <= 0)
        return -1;

    /*吸收已经导出(已经写入到文件)的缓存数据。*/
    abcdk_buffer_drain(buf);

    /*检查是否有数据未导出。*/
    if (buf->wsize == 0)
        return 0;

    return -1;
}
