/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/qrcode/util.h"


abcdk_object_t *abcdk_qrcode_encode(const char *data, size_t size, int level, int scale, int margin)
{
#ifndef HAVE_QRENCODE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含QRencode工具."));
    return NULL; 
#else //#ifndef HAVE_QRENCODE
    abcdk_object_t *dst = NULL;
    QRcode *src = NULL;
    int qr_width,qr_height;
    int y8_width,y8_height;

    assert(data != NULL && size > 0 && (level >= QR_ECLEVEL_L && level <= QR_ECLEVEL_H) && scale > 0 && margin > 0);

    src = QRcode_encodeData(size, data, 0, level);
    if(!src)
        return NULL;

    qr_height = src->width;
    qr_width = src->width;

    y8_height = (qr_height + margin * 2) * scale;
    y8_width = (qr_width + margin * 2) * scale;

    dst = abcdk_object_alloc3(y8_width,y8_height);
    if (!dst)
        goto ERR;

    /*
     *QR: 0x01=black, 0x00=white.
     *Y8: 0x00=black, 0xFF=white.
     */

    /*fill 0xFF(white).*/
    for (int dst_y = 0; dst_y < y8_height; dst_y++)
        memset(dst->pptrs[dst_y], 0xFF, y8_width);

    for (int src_y = 0; src_y < qr_height; src_y++)
    {
        for (int src_x = 0; src_x < qr_width; src_x++)
        {
            uint8_t *src_p = ABCDK_PTR2U8PTR(src->data, src_y * qr_width + src_x);

            for (int dst_y = 0; dst_y < scale; dst_y++)
            {
                for (int dst_x = 0; dst_x < scale; dst_x++)
                {
                    uint8_t *dst_p = ABCDK_PTR2U8PTR(dst->pptrs[(src_y + margin) * scale + dst_y], (src_x + margin) * scale + dst_x);

                    *dst_p = (*src_p & 0x01 ? 0x00 : 0xFF);
                }
            }
        }
    }

    QRcode_free(src);//free.

    return dst;

ERR:

    if(src)
        QRcode_free(src);
    
    abcdk_object_unref(&dst);

    return NULL;
#endif //#ifndef HAVE_QRENCODE
}

int abcdk_qrcode_encode_save(const char *dst, const char *data, size_t size, int level, int scale, int margin)
{
#ifndef HAVE_QRENCODE
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含QRencode工具."));
    return -1; 
#else //#ifndef HAVE_QRENCODE
    abcdk_object_t *src = NULL;
    abcdk_object_t *src_rgb = NULL;
    int width,height;
    int chk;

    assert(dst != NULL);

    if (access(dst, F_OK) == 0)
    {
        chk = truncate(dst, 0);
        if (chk != 0)
            return -1;
    }

    src = abcdk_qrcode_encode(data, size, level, scale, margin);
    if (!src)
        return -1;

    height = src->numbers;
    width = src->sizes[0];
    
    src_rgb = abcdk_object_alloc2(height * width * 3);
    if(!src_rgb)
        goto ERR;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t *src_p = ABCDK_PTR2U8PTR(src->pptrs[y], x);
            uint8_t *rgb_p = ABCDK_PTR2U8PTR(src_rgb->pptrs[0], y * width * 3 + x * 3);

            /*
             *Y8: 0x00=black, 0xFF=white.
             *RGB: 0x00,0x00,0x00=black, 0xFF,0xFF,0xFF=white.
             */
            rgb_p[0] = rgb_p[1] = rgb_p[2] = src_p[0];
        }
    }

    /*高度设置为负, 使图像镜像存储(屏幕显示坐标系).*/
    chk = abcdk_bmp_save_file(dst, src_rgb->pptrs[0], width * 3, width, -height, 24);

    abcdk_object_unref(&src);
    abcdk_object_unref(&src_rgb);

    if(chk != 0)
        return -1;

    return 0;

ERR:

    abcdk_object_unref(&src);
    abcdk_object_unref(&src_rgb);

    return -1;
#endif //#ifndef HAVE_QRENCODE
}
