/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/image/bmp.h"

int abcdk_bmp_save_fd(int fd, const uint8_t *data, uint32_t stride, uint32_t width, int32_t height, uint8_t bits)
{
    abcdk_bmp_file_hdr fhdr = {0};
    abcdk_bmp_info_hdr ihdr = {0};
    uint32_t bmp_stride = 0;
    uint32_t bmp_xbytes = 0;
    uint8_t bmp_xpad[4] = {0};
    uint32_t bmp_padsize = 0;
    const uint8_t *tmp = NULL;
        
    assert(fd >= 0 && data != NULL && stride > 0 && width > 0 && height != 0 && bits > 0);

    assert(bits == 24 || bits == 32);

    if (bits == 24)
        assert(stride >= width * 3);
    if (bits == 32)
        assert(stride >= width * 4);

    bmp_xbytes = width * (bits / 8);

    /*The width must be a multiple of 4.*/
    bmp_stride = abcdk_align(bmp_xbytes, 4);

    if (bmp_xbytes < bmp_stride)
        bmp_padsize = bmp_stride - bmp_xbytes;

    ihdr.size = abcdk_endian_h_to_l32(sizeof(ihdr));
    ihdr.width = abcdk_endian_h_to_l32(width);
    ihdr.height = abcdk_endian_h_to_l32(height);
    ihdr.planes = abcdk_endian_h_to_l16(1);
    ihdr.bitcount = abcdk_endian_h_to_l16(bits);
    ihdr.compression = 0;
    ihdr.size_image = abcdk_endian_h_to_l32(bmp_stride * height);
    ihdr.x_meter = 0;
    ihdr.y_meter = 0;
    ihdr.color_used = 0;
    ihdr.color_important = 0;

    fhdr.type = abcdk_endian_h_to_l16(0x4D42);
    fhdr.size = abcdk_endian_h_to_l32(sizeof(fhdr) + ihdr.size + ihdr.size_image);
    fhdr.reserved1 = fhdr.reserved2 = 0;
    fhdr.offset = abcdk_endian_h_to_l32(sizeof(fhdr) + sizeof(ihdr));

    if(abcdk_write(fd,&fhdr,sizeof(fhdr)) !=sizeof(fhdr))
        return -1; 
    
    if(abcdk_write(fd,&ihdr,sizeof(ihdr)) !=sizeof(ihdr))
        return -1;

    /*Copy data pointer.*/
    tmp = data;

    for (int32_t i = 0; i < abs(height); i++)
    {
        if (abcdk_write(fd, tmp, bmp_xbytes) != bmp_xbytes)
            return -1;

        if (bmp_padsize > 0)
        {
            if (abcdk_write(fd, bmp_xpad, bmp_padsize) != bmp_padsize)
                return -1;
        }

        /*Next line.*/
        tmp += stride;
    }

    return 0;
}

int abcdk_bmp_save_file(const char *file, const uint8_t *data, uint32_t stride, uint32_t width, int32_t height, uint8_t bits)
{
    int fd = -1;
    int chk;

    assert(file != NULL && data != NULL && stride > 0 && width > 0 && height != 0 && bits > 0);

    fd = abcdk_open(file,1,0,1);
    if(fd<0)
        return -1;

    chk = abcdk_bmp_save(fd,data,stride,width,height,bits);

    abcdk_closep(&fd);

    return chk;   
}

int abcdk_bmp_load(int fd, uint8_t *buf, size_t size, uint32_t align,
                   uint32_t *stride, uint32_t *width, int32_t *height, uint8_t *bits)
{
    abcdk_bmp_file_hdr fhdr = {0};
    abcdk_bmp_info_hdr ihdr = {0};
    uint32_t bmp_stride = 0;
    uint32_t bmp_xbytes = 0;
    uint8_t bmp_xpad[4] = {0};
    uint32_t bmp_padsize = 0;
    uint32_t img_stride = 0;
    uint8_t *tmp = NULL;

    assert(fd >= 0);

    if (abcdk_read(fd, &fhdr, sizeof(fhdr)) != sizeof(fhdr))
        return -1;

    /*to host endian*/
    fhdr.type = abcdk_endian_l_to_h16(fhdr.type);
    fhdr.size = abcdk_endian_l_to_h32(fhdr.size);
    fhdr.offset = abcdk_endian_l_to_h32(fhdr.offset);

    /*Check magic(BM).*/
    if(fhdr.type != 0x4D42)
        ABCDK_ERRNO_AND_RETURN1(EINVAL,-1);

    if (abcdk_read(fd, &ihdr, sizeof(ihdr)) != sizeof(ihdr))
        return -1;

    /*to host endian*/
    ihdr.size = abcdk_endian_l_to_h32(ihdr.size);
    ihdr.width = abcdk_endian_l_to_h32(ihdr.width);
    ihdr.height = abcdk_endian_l_to_h32(ihdr.height);
    ihdr.planes = abcdk_endian_l_to_h16(ihdr.planes);
    ihdr.bitcount = abcdk_endian_l_to_h16(ihdr.bitcount);
    ihdr.size_image = abcdk_endian_l_to_h32(ihdr.size_image);
    ihdr.compression = abcdk_endian_l_to_h32(ihdr.compression);
    ihdr.x_meter = abcdk_endian_l_to_h32(ihdr.x_meter);
    ihdr.y_meter = abcdk_endian_l_to_h32(ihdr.y_meter);
    ihdr.color_used = abcdk_endian_l_to_h32(ihdr.color_used);
    ihdr.color_important = abcdk_endian_l_to_h32(ihdr.color_important);

    /*Check pixel format,just BI_RGB.*/
    if (ihdr.compression != 0)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, EINTR);

    /*Check 24 or 32.*/
    if (ihdr.bitcount != 24 && ihdr.bitcount != 32)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, EINTR);

    /**/
    bmp_xbytes = ihdr.width * (ihdr.bitcount / 8);

    /*The width must be a multiple of 4.*/
    bmp_stride = abcdk_align(bmp_xbytes, 4);

    /*Check image size.*/
    if (bmp_stride * ihdr.height  != ihdr.size_image)
        ABCDK_ERRNO_AND_RETURN1(EINVAL, -1);

    if (bmp_xbytes < bmp_stride)
        bmp_padsize = bmp_stride - bmp_xbytes;

    img_stride = abcdk_align(bmp_xbytes,align);

    /*May be check image info.*/
    if (buf == NULL || size <= 0)
        goto final;

    /*Check buffer size.*/
    if (img_stride * ihdr.height > size)
        ABCDK_ERRNO_AND_RETURN1(ENOSPC, -1);

    /*Seek file POS.*/
    if (lseek(fd, fhdr.offset, SEEK_SET) < 0)
        return -1;

    /*Copy buffer pointer.*/
    tmp = buf;

    for (int32_t i = 0; i < abs(ihdr.height); i++)
    {
        if (abcdk_read(fd, tmp, bmp_xbytes) != bmp_xbytes)
            return -1;

        if (bmp_padsize > 0)
        {
            if (abcdk_read(fd, bmp_xpad, bmp_padsize) != bmp_padsize)
                return -1;
        }

        /*Next line.*/
        tmp += img_stride;
    }

final:

    if(stride)
        *stride = img_stride;
    if(width)
        *width = ihdr.width;
    if(height)
        *height = ihdr.height;
    if(bits)
        *bits = ihdr.bitcount;

    return 0;
}

int abcdk_bmp_load2(const char *file, uint8_t *buf, size_t size, uint32_t align,
                    uint32_t *stride, uint32_t *width, int32_t *height, uint8_t *bits)
{
    int fd = -1;
    int chk;

    assert(file != NULL);

    fd = abcdk_open(file, 0, 0, 1);
    if (fd < 0)
        return -1;

    chk = abcdk_bmp_load(fd, buf, size, align, stride, width, height, bits);

    abcdk_closep(&fd);

    return chk;
}

