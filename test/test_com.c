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

    abcdk_hexdump_option_t opt = {0};

    opt.flag = ABCDK_HEXDEMP_SHOW_ADDR | ABCDK_HEXDEMP_SHOW_CHAR;

    char sendmsg[8] = {0};
    char recvmsg[70] = {0};

#if 0
    
    char sendmsg[8] = {0};
    char recvmsg[70] = {0};

    abcdk_bloom_write_number(sendmsg, 8, 0, 8, 0x01);
    abcdk_bloom_write_number(sendmsg, 8, 8, 8, 0x06);
    abcdk_bloom_write_number(sendmsg, 8, 16, 16, 0x0200);
    abcdk_bloom_write_number(sendmsg, 8, 32, 16, 0x0002);
    abcdk_bloom_write_number(sendmsg, 8, 48, 16, abcdk_crc16(sendmsg, 6));

    int chk = abcdk_serialport_transfer(ctx, sendmsg, 8, recvmsg, 8, 1000, sendmsg, 2);
    assert(chk == 0);

    assert(memcmp(sendmsg,recvmsg,8)==0);
#else

 

    //uint8_t addrs[3] = {0x01,0x02,0x05};
    uint8_t addrs[3] = {0x02,0x02,0x02};
    uint16_t dists[3] = {0};

//#pragma omp parallel for num_threads(3)
    for (int i = 0; i < 1000; i++)
    {
        int id = i%3;
        int a = addrs[id];

        char sendmsg[8] = {0};
        char recvmsg[70] = {0};

        abcdk_bloom_write_number(sendmsg, 8, 0, 8, a);
        abcdk_bloom_write_number(sendmsg, 8, 8, 8, 0x03);
        abcdk_bloom_write_number(sendmsg, 8, 16, 16, 0x0101);
        abcdk_bloom_write_number(sendmsg, 8, 32, 16, 0x01);
        abcdk_bloom_write_number(sendmsg, 8, 48, 16, abcdk_crc16(sendmsg, 6));

       // usleep(1000*3);

      //  uint64_t s,d;
      //  d = abcdk_clock(s,&s);

        int chk = abcdk_serialport_transfer(ctx, sendmsg, 8, recvmsg, 7, 1000, sendmsg, 2);
        if(chk != 0)
        {
            uint64_t b;
            abcdk_serialport_get_option(ctx,ABCDK_SERIALPORT_OPT_INTERVAL,&b);
            abcdk_serialport_set_option(ctx,ABCDK_SERIALPORT_OPT_INTERVAL,b+1000);

            continue;
        }

     //   printf("d = %lu\n",d);

        uint16_t oldcrc = abcdk_bloom_read_number(recvmsg, 7, 40, 16);
        uint16_t newcrc = abcdk_crc16(recvmsg, 5);
        if (oldcrc != newcrc)
        {
            abcdk_hexdump(stderr, recvmsg, 7, 0, &opt);
        }
        else
        {
            dists[id] = abcdk_bloom_read_number(recvmsg, 7, 24, 16);

            printf("%d=[%06hu]\n",a, dists[id]);
        }
    }

#endif

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
    abcdk_bloom_write_number(sendmsg, 5, 24, 8, 27);
    abcdk_bloom_write_number(sendmsg, 5, 32, 8, _abcdk_test_com_checksum(sendmsg, 4));

    abcdk_hexdump(stderr, sendmsg, 5, 0, &opt);

    chk = abcdk_serialport_transfer(ctx, sendmsg, 5, NULL, 0, 10000000, NULL, 0);
    assert(chk == 0);

    sleep(3);

    /***************************************************/

    abcdk_bloom_write_number(sendmsg2, 5, 0, 8, 0xA4);
    abcdk_bloom_write_number(sendmsg2, 5, 8, 8, 0x06);
    abcdk_bloom_write_number(sendmsg2, 5, 16, 8, 0x03);
    abcdk_bloom_write_number(sendmsg2, 5, 24, 8, 0);
    abcdk_bloom_write_number(sendmsg2, 5, 32, 8, _abcdk_test_com_checksum(sendmsg2, 4));

    abcdk_hexdump(stderr, sendmsg2, 5, 0, &opt);

    chk = abcdk_serialport_transfer(ctx, sendmsg2, 5, NULL, 0, 1000000, NULL, 0);
    assert(chk == 0);

    sleep(3);

    /***************************************************/

    for (int i = 0; i < 100000; i++)
    {
        chk = abcdk_serialport_transfer(ctx, NULL, 0, recvmsg, 32, 10000, sendmsg, 4);
        assert(chk == 0);

        //   abcdk_hexdump(stderr, recvmsg, 32, 0, &opt);
        uint16_t crc = _abcdk_test_com_checksum(recvmsg, 31);

        if ((crc & 0xff) != recvmsg[31])
        {
            abcdk_hexdump(stderr, recvmsg, 32, 0, &opt);
        }
        else
        {
            abcdk_point_t acc,gyro,mag;
            acc.x = ABCDK_PTR2I16(recvmsg,4);
            acc.y = ABCDK_PTR2I16(recvmsg,6);
            acc.z = ABCDK_PTR2I16(recvmsg,8);
            gyro.x = ABCDK_PTR2I16(recvmsg,10);
            gyro.y = ABCDK_PTR2I16(recvmsg,12);
            gyro.z = ABCDK_PTR2I16(recvmsg,14);
            float roll = (float)ABCDK_PTR2I16(recvmsg,16)/100;
            float pitch = (float)ABCDK_PTR2I16(recvmsg,18)/100;
            float yaw = (float)ABCDK_PTR2I16(recvmsg,20)/100;
            int8_t level = ABCDK_PTR2I8(recvmsg,22);
            float temp = (float)ABCDK_PTR2I16(recvmsg,23)/100;
            mag.x = ABCDK_PTR2I16(recvmsg,25);
            mag.y = ABCDK_PTR2I16(recvmsg,27);
            mag.z = ABCDK_PTR2I16(recvmsg,29);
            
            printf("acc[%06hd,%06hd,%06hd],gyro[%06hd,%06hd,%06hd],roll[%06.2f],pitch[%06.2f],yaw[%06.2f],level[%hhd],temp[%03.02f],mag[%06hd,%06hd,%06hd] %10s\r",
                (uint16_t)acc.x,(uint16_t)acc.y,(uint16_t)acc.z,
                (uint16_t)gyro.x,(uint16_t)gyro.y,(uint16_t)gyro.z,
                roll,pitch,yaw,level,temp,
                (uint16_t)mag.x,(uint16_t)mag.y,(uint16_t)mag.z," ");
        }
    }

    abcdk_serialport_destroy(&ctx);
}

int abcdk_test_com(abcdk_tree_t *args)
{
     abcdk_test_com_ultrasound(args);
   // abcdk_test_com_xyz(args);
}