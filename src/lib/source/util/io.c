/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/io.h"

int abcdk_poll(int fd, int event, time_t timeout)
{
    struct pollfd arr = {0};
    int chk = 0;
    int ret = 0;

    assert(fd >= 0 && (event & 0x03));

    arr.fd = fd;
    arr.events = 0;

    if ((event & 0x01))
        arr.events |= POLLIN;
    if ((event & 0x02))
        arr.events |= POLLOUT;

    arr.events |= POLLERR;
    arr.events |= POLLHUP;
    arr.events |= POLLNVAL;

    chk = poll(&arr, 1, (timeout >= INT32_MAX ? -1 : timeout));
    if (chk <= 0)
        return chk;

    if (chk & POLLERR)
        return -1;
    if (chk & POLLHUP)
        return -1;
    if (chk & POLLNVAL)
        return -1;
    if (chk & POLLIN)
        ret |= 0x01;
    if (chk & POLLOUT)
        ret |= 0x02;

    return ret;
}

ssize_t abcdk_write(int fd, const void *data, size_t size)
{
    ssize_t wsize = 0;
    ssize_t wsize2 = 0;

    assert(fd >= 0 && data && size > 0);

    wsize = write(fd, data, size);
    if (wsize > 0)
    {
        if (wsize < size)
        {
            /*有的系统超过2GB，需要分段落盘。*/
            wsize2 = abcdk_write(fd, ABCDK_PTR2PTR(void, data, wsize), size - wsize);
            if (wsize2 > 0)
                wsize += wsize2;
        }
    }

    return wsize;
}

ssize_t abcdk_read(int fd, void *data, size_t size)
{
    ssize_t rsize = 0;
    ssize_t rsize2 = 0;

    assert(fd >= 0 && data && size > 0);

    rsize = read(fd, data, size);
    if (rsize > 0)
    {
        if (rsize < size)
        {
            /*有的系统超过2GB，需要分段读取。*/
            rsize2 = abcdk_read(fd, ABCDK_PTR2PTR(char, data, rsize), size - rsize);
            if (rsize2 > 0)
                rsize += rsize2;
        }
    }

    return rsize;
}

void abcdk_closep(int *fd)
{
    int fd_cp;

    if (!fd)
        return;

    fd_cp = *fd;
    *fd = -1;

    if (fd_cp >= 0)
        close(fd_cp);
}

int abcdk_open(const char *file, int rw, int nonblock, int create)
{
    int flag = O_RDONLY;
    mode_t mode = S_IRUSR | S_IWUSR;

    assert(file);

    if (rw)
        flag = O_RDWR;

    if (nonblock)
        flag |= O_NONBLOCK;

    if (rw && create)
        flag |= O_CREAT;

    flag |= O_LARGEFILE;
    flag |= O_CLOEXEC;

    return open(file, flag, mode);
}

int abcdk_reopen(int fd2, const char *file, int rw, int nonblock, int create)
{
    int fd = -1;
    int fd3 = -1;

    assert(fd2 >= 0);

    fd = abcdk_open(file, rw, nonblock, create);
    if (fd < 0)
        return -1;

    fd3 = dup2(fd, fd2);

    /*必须要关闭，不然句柄就会丢失，造成资源泄露。*/
    abcdk_closep(&fd);

    return fd3;
}

int abcdk_fflag_set(int fd, int flag)
{
    assert(fd >= 0 && flag != 0);

    return fcntl(fd, F_SETFL, flag);
}

int abcdk_fflag_get(int fd)
{
    assert(fd >= 0);

    return fcntl(fd, F_GETFL, 0);
}

int abcdk_fflag_add(int fd, int flag)
{
    int old;
    int opt;

    assert(fd >= 0 && flag != 0);

    old = abcdk_fflag_get(fd);
    if (old == -1)
        return -1;

    if ((~old) & flag)
    {
        opt = old | flag;
        return abcdk_fflag_set(fd, opt);
    }

    return 0;
}

int abcdk_fflag_del(int fd, int flag)
{
    int old;
    int opt;

    assert(fd >= 0 && flag != 0);

    old = abcdk_fflag_get(fd);
    if (old == -1)
        return -1;

    opt = old & ~flag;

    return abcdk_fflag_set(fd, opt);
}


ssize_t abcdk_load(const char *file, void *buf, size_t size, size_t offset)
{
    int fd = -1;
    ssize_t rlen = 0;
    off_t off;
    int chk;

    fd = abcdk_open(file, 0, 0, 0);
    if (fd < 0)
        return -1;

    off = lseek(fd, offset, SEEK_SET);
    if (off != offset)
        goto END;

    rlen = abcdk_read(fd, buf, size);

END:

    abcdk_closep(&fd);
    
    return rlen;
}

ssize_t abcdk_save(const char *file, const void *buf, size_t size, size_t offset)
{
    int fd = -1;
    ssize_t wlen = 0;
    off_t off;
    int chk;

    fd = abcdk_open(file, 1, 0, 1);
    if (fd < 0)
        return -1;

    off = lseek(fd, offset, SEEK_SET);
    if (off != offset)
        return -1;

    wlen = abcdk_write(fd, buf, size);

final:

    abcdk_closep(&fd);
    
    return wlen;
}

ssize_t abcdk_save2temp(char *file, const void *buf, size_t size, size_t offset)
{
    int fd = -1;
    ssize_t wlen = 0;
    off_t off;
    int chk;

    assert(file != NULL && buf != NULL && size > 0);

    fd = mkstemp(file);
    if (fd < 0)
        return -1;

    off = lseek(fd, offset, SEEK_SET);
    if (off != offset)
        return -1;

    wlen = abcdk_write(fd, buf, size);

final:

    abcdk_closep(&fd);

    return wlen;
}

ssize_t abcdk_fgetline(FILE *fp, char **line, size_t *len, uint8_t delim, char note)
{
    char *line_p = NULL;
    ssize_t rlen = -1;

    assert(fp != NULL && line != NULL);

    while ((rlen = getdelim(line, len, delim, fp)) != -1)
    {
        line_p = *line;

        if (*line_p == '\0' || *line_p == note)
            continue;
        else
            break;
    }

    return rlen;
}

void abcdk_fclosep(FILE **fp)
{
    FILE *fp_p = NULL;

    if(!fp || !*fp)
        return;

    fp_p = *fp;
    *fp = NULL;

    fclose(fp_p);
}

int64_t abcdk_fsize(FILE *fp)
{
    int64_t size = -1;
    int64_t pos = -1;

    assert(fp != NULL);

    pos = ftell(fp);
    if (pos >= 0)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fseek(fp, pos, SEEK_SET);
    }

    return size;
}

int abcdk_futimens(int fd, const struct timespec *atime, const struct timespec *mtime)
{
    struct timespec times[2] = {0};
    int chk;

    if (atime && atime->tv_sec > 0)
        times[0] = *atime;
    else
        clock_gettime(CLOCK_REALTIME, &times[0]);

    if (mtime && mtime->tv_sec > 0)
        times[1] = *mtime;
    else
        clock_gettime(CLOCK_REALTIME, &times[1]);

    chk = futimens(fd, times);
    if (chk != 0)
        return -1;

    return 0;
}

ssize_t abcdk_transfer(int fd, void *data, size_t size, int direction, time_t timeout,
                       const void *magic, size_t mglen)
{
    time_t time_end;
    time_t time_span;
    ssize_t len = 0, all = 0;
    int fd_flag = 0;
    int chk;

    assert(fd >= 0 && data != NULL && size > 0 && direction != 0 && timeout > 0);
    assert(direction == 1 || direction == 2);

    /*计算过期时间。*/
    time_end = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3) + timeout;
    time_span = timeout;

    /*获取句柄标志。*/
    fd_flag = abcdk_fflag_get(fd);

    /*仅支持带有异步标志的句柄。*/
    if (!(fd_flag & O_NONBLOCK))
        return -2;

    while (all < size)
    {
        if(direction == 2)
            len = write(fd, ABCDK_PTR2VPTR(data, all), size - all);
        else if(direction == 1)
            len = read(fd, ABCDK_PTR2VPTR(data, all), size - all);
        else
            break;

        if (len == -1)
        {
            if (errno != EAGAIN && errno != EINTR)
                break;

            if (direction == 2)
                chk = abcdk_poll(fd, 0x02, time_span);
            else if (direction == 1)
                chk = abcdk_poll(fd, 0x01, time_span);
            else
                break;

            if (chk > 0)
                continue;
            else
                break;
        }
        else if (len == 0)
            break;
        else
            all += len;


        /*输出数据时不需要检查起始码。*/
        if(direction != 1)
            continue;

        /*未定义起始码，忽略。*/
        if (!magic || mglen <= 0)
            continue;

        /*已读取数据长度不能小于起始码长度。*/
        if(all < mglen)
            continue;
        
        len = 0;
        for (size_t i = 0; i < all - mglen; i++)
        {
            if (memcmp(ABCDK_PTR2VPTR(data, i), magic, mglen) == 0)
                break;

            /*逐个字节查找。*/
            len += 1;
        }

        /*根据起始码位移动数据，并重新计算数据长度。*/
        if (len > 0)
        {
            memmove(data, ABCDK_PTR2VPTR(data, len), all - len);
            all -= len;

            /*起始码不符合，计算剩余超时时长。*/
            time_span = time_end - abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3);
            if (time_span <= 0)
                break;
        }
        else
        {
            /*数据不足，继续按原超时等待。*/
            time_span = timeout;
        }
    }

    return all;
}
