/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_com_ultrasound(abcdk_tree_t *args)
{
    int fd = abcdk_open("/dev/ttyUSB0", 1, 0, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);

    abcdk_serialport_t *ctx = abcdk_serialport_create(fd);

    char sendmsg[8] = {0};
    char recvmsg[70] = {0};

    abcdk_bloom_write_number(sendmsg, 8, 0, 8, 0x01);
    abcdk_bloom_write_number(sendmsg, 8, 8, 8, 0x03);
    abcdk_bloom_write_number(sendmsg, 8, 16, 16, 0x0101);
    abcdk_bloom_write_number(sendmsg, 8, 32, 16, 0x01);

    uint16_t crc = abcdk_crc16(sendmsg, 6);
    abcdk_bloom_write_number(sendmsg, 8, 48, 16, crc);

    int chk = abcdk_serialport_transfer(ctx, sendmsg, 8, recvmsg, 7, 100, NULL, 0);

    abcdk_serialport_destroy(&ctx);
}

uint32_t _abcdk_test_com_checksum(const void *data, size_t size)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < size; i++)
        sum += ABCDK_PTR2U8(data, i);
    return sum;
}

int abcdk_test_com_xyz(abcdk_tree_t *args)
{
    int chk;
    int fd = abcdk_open("/dev/ttyUSB0", 1, 0, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);

    abcdk_serialport_t *ctx = abcdk_serialport_create(fd);

    abcdk_hexdump_option_t opt = {0};

    opt.flag = ABCDK_HEXDEMP_SHOW_ADDR | ABCDK_HEXDEMP_SHOW_CHAR;

    uint8_t sendmsg[8] = {0};
    uint8_t sendmsg2[8] = {0};
    uint8_t recvmsg[70] = {0};

    /***************************************************/

    abcdk_bloom_write_number(sendmsg, 5, 0, 8, 0xA4);
    abcdk_bloom_write_number(sendmsg, 5, 8, 8, 0x03);
    abcdk_bloom_write_number(sendmsg, 5, 16, 8, 0x08);
    abcdk_bloom_write_number(sendmsg, 5, 24, 8, 18);
    abcdk_bloom_write_number(sendmsg, 5, 32, 8, _abcdk_test_com_checksum(sendmsg, 4));

    abcdk_hexdump(stderr, sendmsg, 5, 0, &opt);

    chk = abcdk_serialport_transfer(ctx, sendmsg, 5, NULL, 0, 10000000, NULL, 0);
    assert(chk == 0);

    /***************************************************/

    abcdk_bloom_write_number(sendmsg2, 5, 0, 8, 0xA4);
    abcdk_bloom_write_number(sendmsg2, 5, 8, 8, 0x06);
    abcdk_bloom_write_number(sendmsg2, 5, 16, 8, 0x03);
    abcdk_bloom_write_number(sendmsg2, 5, 24, 8, 0);
    abcdk_bloom_write_number(sendmsg2, 5, 32, 8, _abcdk_test_com_checksum(sendmsg2, 4));

    abcdk_hexdump(stderr, sendmsg2, 5, 0, &opt);

    chk = abcdk_serialport_transfer(ctx, sendmsg2, 5, NULL, 0, 1000000, NULL, 0);
    assert(chk == 0);

    //sleep(3);

    /***************************************************/

    for (int i = 0; i < 10000; i++)
    {
        chk = abcdk_serialport_transfer(ctx, NULL, 0, recvmsg, 23, 100, sendmsg, 3);
        assert(chk == 0);

        //   abcdk_hexdump(stderr, recvmsg, 23, 0, &opt);
        uint16_t crc = _abcdk_test_com_checksum(recvmsg, 22);

        if ((crc & 0xff) != recvmsg[22])
        {
            abcdk_hexdump(stderr, recvmsg, 23, 0, &opt);
        }
        else
        {
            for(int i = 0;i<9;i++)
            {
                uint16_t a = abcdk_bloom_read_number(recvmsg,23,4+i*16,16);
                printf("%hu,",a );
            }
            printf("\n");
        }
    }

    abcdk_serialport_destroy(&ctx);
}

int abcdk_test_com(abcdk_tree_t *args)
{
    // abcdk_test_com_ultrasound(args);
    abcdk_test_com_xyz(args);
}