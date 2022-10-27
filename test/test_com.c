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


int abcdk_test_com(abcdk_tree_t *args)
{
  //  int fd = abcdk_open("/dev/ttyUSB0",1,0,0);

   // abcdk_tcattr_serial(fd,115200,8,0,1,NULL);

    char sendmsg[] = {0x01,0x03,0x01,0x01,0x00,0x01,0xD4,0x36};
   // char sendmsg[8] = {0};

  //  abcdk_bloom_write_number(sendmsg,8,0,8,0x01);
  //  abcdk_bloom_write_number(sendmsg,8,8,8,0x03);
  //  abcdk_bloom_write_number(sendmsg,8,16,16,0x0101);
  //  abcdk_bloom_write_number(sendmsg,8,32,16,0x01);
  //  abcdk_bloom_write_number(sendmsg,8,48,16,0xd436);

    uint16_t a = abcdk_crc16(sendmsg,6);
    uint16_t aa = ~a;
    uint32_t b = abcdk_crc32(sendmsg,6);

  //  abcdk_write(fd,sendmsg,sizeof(sendmsg));

  //  char recvmsg[100] = {0};

 //   int n = abcdk_read(fd,recvmsg,7);

  //  abcdk_closep(&fd);
}