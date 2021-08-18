/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-mp4/file.h"

int abcdk_mp4_size(int fd, uint64_t *size)
{
    struct stat attr = {0};

    if (fstat(fd, &attr) != 0)
        return -1;

    *size = attr.st_size;

    return 0;
}

int abcdk_mp4_read(int fd, void *data, size_t size)
{
    ssize_t rsize = abcdk_read(fd, data, size);
    if (rsize <= 0)
        return -2;
    else if (rsize != size)
        return -1;

    return 0;
}

int abcdk_mp4_read_u16(int fd, uint16_t *data)
{
    if (abcdk_mp4_read(fd, data, sizeof(uint16_t)))
        return -1;

    *data = abcdk_endian_b_to_h16(*data);

    return 0;
}

int abcdk_mp4_read_u24(int fd, uint8_t *data)
{
    if (abcdk_mp4_read(fd, data, sizeof(uint8_t) * 3))
        return -1;

    abcdk_endian_b_to_h(data, 3);

    return 0;
}

int abcdk_mp4_read_u32(int fd, uint32_t *data)
{
    if (abcdk_mp4_read(fd, data, sizeof(uint32_t)))
        return -1;

    *data = abcdk_endian_b_to_h32(*data);

    return 0;
}

int abcdk_mp4_read_u64(int fd, uint64_t *data)
{
    if (abcdk_mp4_read(fd, data, sizeof(uint64_t)))
        return -1;

    *data = abcdk_endian_b_to_h64(*data);

    return 0;
}