/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_com_ultrasound(abcdk_option_t *args)
{
    int fd = abcdk_open("/dev/ttyUSB0", 1, 1, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);


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

    int chk = abcdk_tcattr_transfer(fd, sendmsg, 8, recvmsg, 8, 1000, sendmsg, 2);
    assert(chk == 0);

    assert(memcmp(sendmsg,recvmsg,8)==0);
#else

  

    uint8_t addrs[3] = {0x35,0x33,0x34};
  //  uint8_t addrs[3] = {0x02,0x02,0x02};
    uint16_t dists[3] = {0};

   //uint8_t addrs[3] = {0x33,0x33,0x33};

    uint64_t s = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,6) ,t;

    for (int i = 0; i < 1000; i++)
    {
        int id = i%3;
        int a = addrs[id];
        int chk = -1;

        char sendmsg[8] = {0};
        char recvmsg[70] = {0};

        abcdk_bloom_write_number(sendmsg, 8, 0, 8, a);
        abcdk_bloom_write_number(sendmsg, 8, 8, 8, 0x03);
        abcdk_bloom_write_number(sendmsg, 8, 16, 16, 0x0101);
        abcdk_bloom_write_number(sendmsg, 8, 32, 16, 0x01);
        abcdk_bloom_write_number(sendmsg, 8, 48, 16, abcdk_crc16(sendmsg, 6));

       // usleep(1000*3);


        abcdk_clock(s,&s);
        

        chk = abcdk_tcattr_transfer(fd, sendmsg, 8, recvmsg, 7, 200, sendmsg, 2);

        t = abcdk_clock(s,&s);
    //    printf("[%02x] t = %lu\n",a,t);
        if(chk != 0)
        {
            printf("%02x timeout.\n",a);
            continue;
        }


        uint16_t oldcrc = abcdk_bloom_read_number(recvmsg, 7, 40, 16);
        uint16_t newcrc = abcdk_crc16(recvmsg, 5);
        if (oldcrc != newcrc)
        {
            printf("%02x crc[%04x,%04x].\n",a,oldcrc,newcrc);
            abcdk_hexdump(stderr, recvmsg, 7, 0, NULL);
        }
        else
        {
            dists[id] = abcdk_bloom_read_number(recvmsg, 7, 24, 16);

            printf("%02x=[%06hu]\n",a, dists[id]);
        }
    }

#endif

    abcdk_closep(&fd);

    return 0;
}

uint32_t _abcdk_test_com_checksum(const void *data, size_t size)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < size; i++)
        sum += ABCDK_PTR2U8(data, i);
    return sum;
}

int abcdk_test_com_xyz(abcdk_option_t *args)
{
    int chk;
    int fd = abcdk_open("/dev/ttyUSB0", 1, 1, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);


    uint8_t sendmsg[8] = {0};
    uint8_t sendmsg2[8] = {0};
    uint8_t recvmsg[70] = {0};

    /***************************************************/

    abcdk_bloom_write_number(sendmsg, 5, 0, 8, 0xA4);
    abcdk_bloom_write_number(sendmsg, 5, 8, 8, 0x03);
    abcdk_bloom_write_number(sendmsg, 5, 16, 8, 0x08);
    abcdk_bloom_write_number(sendmsg, 5, 24, 8, 27);
    abcdk_bloom_write_number(sendmsg, 5, 32, 8, _abcdk_test_com_checksum(sendmsg, 4));

    abcdk_hexdump(stderr, sendmsg, 5, 0, NULL);

    chk = abcdk_tcattr_transfer(fd, sendmsg, 5, NULL, 0, 10000000, NULL, 0);
    assert(chk == 0);

    sleep(3);

    /***************************************************/

    abcdk_bloom_write_number(sendmsg2, 5, 0, 8, 0xA4);
    abcdk_bloom_write_number(sendmsg2, 5, 8, 8, 0x06);
    abcdk_bloom_write_number(sendmsg2, 5, 16, 8, 0x03);
    abcdk_bloom_write_number(sendmsg2, 5, 24, 8, 0);
    abcdk_bloom_write_number(sendmsg2, 5, 32, 8, _abcdk_test_com_checksum(sendmsg2, 4));

    abcdk_hexdump(stderr, sendmsg2, 5, 0,NULL);

    chk = abcdk_tcattr_transfer(fd, sendmsg2, 5, NULL, 0, 1000000, NULL, 0);
    assert(chk == 0);

    sleep(3);

    /***************************************************/

    for (int i = 0; i < 100000; i++)
    {
        chk = abcdk_tcattr_transfer(fd, NULL, 0, recvmsg, 32, 10000, sendmsg, 4);
        assert(chk == 0);

        //   abcdk_hexdump(stderr, recvmsg, 32, 0, NULL);
        uint16_t crc = _abcdk_test_com_checksum(recvmsg, 31);

        if ((crc & 0xff) != recvmsg[31])
        {
            abcdk_hexdump(stderr, recvmsg, 32, 0,NULL);
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

    abcdk_closep(&fd);
    return 0;
}


int abcdk_test_com_driver(abcdk_option_t *args)
{
    int chk;
    int fd = abcdk_open("/dev/ttyUSB0", 1, 1, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);


    uint8_t sendmsg[80] = {0};
    uint8_t sendmsg2[80] = {0};
    uint8_t recvmsg[70] = {0};

    abcdk_bit_t wbits = {0,sendmsg,80};
    abcdk_bit_t rbits = {0,recvmsg,70};

    int addr = 0x01;

    /***************************************************/
    wbits.pos = 0;
    abcdk_bit_write(&wbits,8,addr);
    abcdk_bit_write(&wbits,8,0x06);
    abcdk_bit_write(&wbits,16,0x200E);
    abcdk_bit_write(&wbits,16,6);
    abcdk_bit_write(&wbits,16,abcdk_crc16(sendmsg, 6));


    chk = abcdk_tcattr_transfer(fd, sendmsg,8,NULL, 0, 10000, NULL, 0);
    assert(chk ==0);

   // abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

    printf("clear error.\n");

    /***************************************************/
    wbits.pos = 0;
    abcdk_bit_write(&wbits,8,addr);
    abcdk_bit_write(&wbits,8,0x06);
    abcdk_bit_write(&wbits,16,0x200D);
    abcdk_bit_write(&wbits,16,3);
    abcdk_bit_write(&wbits,16,abcdk_crc16(sendmsg, 6));


    chk = abcdk_tcattr_transfer(fd, sendmsg,8,NULL, 0,10000, NULL, 0);
    assert(chk ==0);

   // abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

    /***************************************************/
    wbits.pos = 0;
    abcdk_bit_write(&wbits,8, addr);
    abcdk_bit_write(&wbits,8, 0x10);
    abcdk_bit_write(&wbits,16, 0x2080);
    abcdk_bit_write(&wbits,16, 4);
    abcdk_bit_write(&wbits,8, 8);
    abcdk_bit_write(&wbits,16, 100);
    abcdk_bit_write(&wbits,16, 100);
    abcdk_bit_write(&wbits,16, 100);
    abcdk_bit_write(&wbits,16, 100);
    abcdk_bit_write(&wbits,16, abcdk_crc16(sendmsg, 15));


    chk = abcdk_tcattr_transfer(fd, sendmsg,17,NULL, 0,10000, NULL, 0);
    assert(chk ==0);

    //abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

    /***************************************************/
    wbits.pos = 0;
    abcdk_bit_write(&wbits,8,addr);
    abcdk_bit_write(&wbits,8,0x06);
    abcdk_bit_write(&wbits,16,0x200E);
    abcdk_bit_write(&wbits,16,8);
    abcdk_bit_write(&wbits,16,abcdk_crc16(sendmsg, 6));


    chk = abcdk_tcattr_transfer(fd, sendmsg,8,NULL, 0, 10000, NULL, 0);
    assert(chk ==0);

  //  abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

    /***************************************************/

    for (int i = 0; i < 10; i++)
    {
        int speed = i*10;
        wbits.pos = 0;
        abcdk_bit_write(&wbits, 8, addr);
        abcdk_bit_write(&wbits, 8, 0x10);
        abcdk_bit_write(&wbits, 16, 0x2088);
        abcdk_bit_write(&wbits, 16, 2);
        abcdk_bit_write(&wbits, 8, 4);
        abcdk_bit_write(&wbits, 16, speed);
        abcdk_bit_write(&wbits, 16, -speed);
        abcdk_bit_write(&wbits, 16, abcdk_crc16(sendmsg, 11));

        chk = abcdk_tcattr_transfer(fd, sendmsg, 13, NULL, 0, 10000, NULL, 0);
        assert(chk == 0);

       // abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

        sleep(10);
    }

     /***************************************************/
    wbits.pos = 0;
    abcdk_bit_write(&wbits,8,addr);
    abcdk_bit_write(&wbits,8,0x06);
    abcdk_bit_write(&wbits,16,0x200E);
    abcdk_bit_write(&wbits,16,7);
    abcdk_bit_write(&wbits,16,abcdk_crc16(sendmsg, 6));

    chk = abcdk_tcattr_transfer(fd, sendmsg,8,NULL, 0, 10000, NULL, 0);
    assert(chk ==0);

  //  abcdk_hexdump(stderr, recvmsg, 8, 0,NULL);

    printf("stop.\n");

    /***************************************************/

    abcdk_closep(&fd);
    
    return 0;
}

int abcdk_test_com_gripper(abcdk_option_t *args)
{
    int chk;
    int fd = abcdk_open("/dev/ttyUSB0", 1, 1, 0);

    abcdk_tcattr_serial(fd, 115200, 8, 0, 1, NULL);


    uint8_t sendmsg[80] = {0};
    uint8_t sendmsg2[80] = {0};
    uint8_t recvmsg[70] = {0};

    abcdk_bit_t wbits = {0,sendmsg,80};
    abcdk_bit_t rbits = {0,recvmsg,70};

    int addr = 0x01;

    for (int i = 0; i < 100; i++)
    {
        int a = rand()%70;

        wbits.pos = 0;
        abcdk_bit_write(&wbits, 16, 0xEB90);
        abcdk_bit_write(&wbits, 8, addr);
        abcdk_bit_write(&wbits, 8, 0x03);
        abcdk_bit_write(&wbits, 8, 0x54);
        //   abcdk_bit_write(&wbits,8,0x11);
        abcdk_bit_write(&wbits, 16, abcdk_endian_h_to_b16(1000 / 70 * a));
        abcdk_bit_write(&wbits, 8, _abcdk_test_com_checksum(sendmsg + 2, 5));

        chk = abcdk_tcattr_transfer(fd, sendmsg, wbits.pos / 8, recvmsg, 7, 10000, NULL, 0);
        assert(chk == 0);

        sleep(1);
    }

    abcdk_closep(&fd);
    
    return 0;
}

int abcdk_test_com(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, -1);

    if (cmd == 1)
        abcdk_test_com_ultrasound(args);
    if (cmd == 2)
        abcdk_test_com_xyz(args);
    if (cmd == 3)
        abcdk_test_com_driver(args);
    if (cmd == 4)
        abcdk_test_com_gripper(args);

    return 0;
}
