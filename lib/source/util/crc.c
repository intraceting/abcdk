/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/crc.h"

int _abcdk_crc32_init(void *opaque)
{
    uint32_t *table = (uint32_t *)opaque;

    unsigned int c;
    unsigned int i, j;

    for (i = 0; i < 256; i++)
    {
        c = (unsigned int)i;
        for (j = 0; j < 8; j++)
        {
            if (c & 1)
                c = 0xEDB88320L ^ (c >> 1);
            else
                c = c >> 1;
        }

        table[i] = c;
    }

    return 0;
}

uint32_t abcdk_crc32(const void *data, size_t size, ...)
{
    uint32_t sum = ~0;
    static volatile int init = 0;
    static uint32_t table[256] = {0};
    int chk;

    assert(data != NULL && size > 0);

    chk = abcdk_once(&init, _abcdk_crc32_init, table);
    assert(chk >= 0);

    for (size_t i = 0; i < size; i++)
    {
        sum = table[(sum ^ ABCDK_PTR2OBJ(uint8_t, data, i)) & 0xFF] ^ (sum >> 8);
    }

    return ~sum;
}

const uint16_t crctalbeabs[] = {
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400};

uint16_t abcdk_crc16(const void *data, size_t size, ...)
{
    uint8_t *ptr = (uint8_t *)data;
    uint16_t len = size;

    uint16_t crc = 0xffff;
    uint16_t i;
    uint8_t ch;
    for (i = 0; i < len; i++)
    {
        ch = *ptr++;
        crc = crctalbeabs[(ch ^ crc) & 15] ^ (crc >> 4);
        crc = crctalbeabs[((ch >> 4) ^ crc) & 15] ^ (crc >> 4);
    }
    crc = ((crc & 0x00FF) << 8) | ((crc & 0xFF00) >> 8); //??????
    return crc;
}