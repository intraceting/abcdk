/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/image/freeimage.h"

#ifdef FREEIMAGE_H

/* 环境初始化计数器*/
static volatile int64_t _abcdk_fi_init_count = 0;

void abcdk_fi_uninit()
{
    int64_t chk = abcdk_atomic_fetch_and_add(&_abcdk_fi_init_count, -1);

    assert(chk >= 1);

    if (chk == 1)
        FreeImage_DeInitialise();
}

void abcdk_fi_init(int load_local_plugins_only)
{
    int64_t chk = abcdk_atomic_fetch_and_add(&_abcdk_fi_init_count, 1);

    assert(chk >= 0);

    if (chk == 0)
        FreeImage_Initialise(load_local_plugins_only);
}

unsigned _abcdk_fi_read_cb(void *buffer, unsigned size, unsigned count, fi_handle handle)
{
    int fd = ABCDK_PTR2OBJ(int, handle, 0);
    ssize_t ret;

    ret = abcdk_read(fd, buffer, size * count);

    return ret / size;
}

unsigned _abcdk_fi_write_cb(void *buffer, unsigned size, unsigned count, fi_handle handle)
{
    int fd = ABCDK_PTR2OBJ(int, handle, 0);
    ssize_t ret;

    ret = abcdk_write(fd, buffer, size * count);

    return ret / size;
}

int _abcdk_fi_seek_cb(fi_handle handle, long offset, int origin)
{
    int fd = ABCDK_PTR2OBJ(int, handle, 0);

    return lseek(fd, offset, origin);
}

long _abcdk_fi_tell_cb(fi_handle handle)
{
    int fd = ABCDK_PTR2OBJ(int, handle, 0);

    return lseek(fd, 0, SEEK_SET);
}

int abcdk_fi_save(FREE_IMAGE_FORMAT fifmt, int fiflag, int fd, const uint8_t *data,
                  uint32_t stride, uint32_t width, uint32_t height, uint8_t bits)
{
    FreeImageIO io = {0};
    FIBITMAP *dib = NULL;
    uint8_t *dib_tmp = NULL;
    uint32_t dib_stride = 0;
    uint32_t dib_xbytes = 0;
    const uint8_t *tmp = NULL;
    int chk = -1;

#if ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JXR);
#else 
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JP2);
#endif //ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)

    assert(fd >= 0 && data != NULL && stride > 0 && width > 0 && height > 0 && bits > 0);

    assert(bits == 24 || bits == 32);

    if (bits == 24)
        assert(stride >= width * 3);
    if (bits == 32)
        assert(stride >= width * 4);

    dib = FreeImage_Allocate(width,height,bits,0,0,0);
    if(!dib)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM,-1);

    dib_tmp = FreeImage_GetBits(dib);
    if(!dib_tmp)
        ABCDK_ERRNO_AND_GOTO1(EPERM,final);

    dib_stride = FreeImage_GetPitch(dib);
    dib_xbytes = FreeImage_GetLine(dib);

    /*Copy data pointer.*/
    tmp = data;

    for (int32_t i = 0; i < height; i++)
    {
        memcpy(dib_tmp,tmp,dib_xbytes);

        /*Next line.*/
        tmp += stride;
        dib_tmp += dib_stride;
    }

    io.read_proc = _abcdk_fi_read_cb;
    io.seek_proc = _abcdk_fi_seek_cb;
    io.tell_proc = _abcdk_fi_tell_cb;
    io.write_proc = _abcdk_fi_write_cb;

    if(!FreeImage_SaveToHandle(fifmt,dib,&io,(fi_handle)&fd,fiflag))
        ABCDK_ERRNO_AND_GOTO1(EPERM,final);

    /*No error.*/
    chk = 0;

final:

    if(dib)
        FreeImage_Unload(dib);

    return chk;
}

int abcdk_fi_save2(FREE_IMAGE_FORMAT fifmt, int fiflag, const char *file, const uint8_t *data,
                   uint32_t stride, uint32_t width, uint32_t height, uint8_t bits)
{
    int fd = -1;
    int chk;

#if ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JXR);
#else 
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JP2);
#endif //ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)

    assert(file != NULL && data != NULL && stride > 0 && width > 0 && height != 0 && bits > 0);

    fd = abcdk_open(file,1,0,1);
    if(fd<0)
        return -1;

    chk = abcdk_fi_save(fifmt,fiflag,fd,data,stride,width,height,bits);

    abcdk_closep(&fd);

    return chk;
}

FIBITMAP *abcdk_fi_load(FREE_IMAGE_FORMAT fifmt,int fiflag,int fd)
{
    FreeImageIO io = {0};
    FIBITMAP *dib = NULL;
    
#if ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JXR);
#else 
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JP2);
#endif //ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)

    assert(fd>=0);

    io.read_proc = _abcdk_fi_read_cb;
    io.seek_proc = _abcdk_fi_seek_cb;
    io.tell_proc = _abcdk_fi_tell_cb;
    io.write_proc = _abcdk_fi_write_cb;

    dib = FreeImage_LoadFromHandle(fifmt,&io,(fi_handle)&fd,fiflag);
    if(!dib)
        ABCDK_ERRNO_AND_RETURN1(EPERM,NULL);

    return dib;
}

FIBITMAP *abcdk_fi_load2(FREE_IMAGE_FORMAT fifmt,int fiflag,const char *file)
{
    int fd = -1;
    FIBITMAP *dib = NULL;

#if ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JXR);
#else 
    assert(fifmt >= FIF_BMP && fifmt <= FIF_JP2);
#endif //ABCDK_VERSION_AT_LEAST(FREEIMAGE_MAJOR_VERSION,FREEIMAGE_MINOR_VERSION,3,17)

    assert(file != NULL);

    fd = abcdk_open(file,0,0,1);
    if(fd<0)
        return NULL;

    dib = abcdk_fi_load(fifmt,fiflag,fd);

    abcdk_closep(&fd);

    return dib;
}

#endif //FREEIMAGE_H