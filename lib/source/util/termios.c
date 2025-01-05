/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
*/
#include "abcdk/util/termios.h"

int abcdk_tcattr_option(int fd, const struct termios *now, struct termios *old)
{
    assert(fd >=0 && (now != NULL || old != NULL));

    if (old)
    {
        if (tcgetattr(fd, old) != 0)
            goto final_error;
    }

    if (now)
    {
        tcflush(fd,TCIFLUSH);
        
        if (tcsetattr(fd,TCSANOW,now)!=0)
            goto final_error;
    }

    return 0;

final_error:

    return -1;
}

int abcdk_tcattr_cbreak(int fd, struct termios *old)
{
    struct termios now = {0};

    assert(fd >= 0);

    if (abcdk_tcattr_option(fd, NULL, &now) != 0)
        return -1;

    now.c_lflag &= (~ICANON);
    now.c_lflag &= (~ECHO);
    now.c_cc[VTIME] = 0;
    now.c_cc[VMIN] = 1;

    return abcdk_tcattr_option(fd,&now,old);
}

static struct _abcdk_tcattr_serial_baudrate_dict
{
    uint64_t code;
    speed_t speed;
} abcdk_tcattr_serial_baudrate_dict[] = {
    {50, B50},
    {75, B75},
    {110, B110},
    {134, B134},
    {150, B150},
    {200, B200},
    {300, B300},
    {600, B600},
    {1200, B1200},
    {1800, B1800},
    {2400, B2400},
    {4800, B4800},
    {9600, B9600},
    {19200, B19200},
    {38400, B38400},
    {57600, B57600},
    {115200, B115200},
    {230400, B230400},
    {230400, B230400},
    {460800, B460800},
    {500000, B500000},
    {576000, B576000},
    {921600, B921600},
    {1000000, B1000000},
    {1152000, B1152000},
    {1500000, B1500000},
    {2000000, B2000000},
    {2500000, B2500000},
    {3000000, B3000000},
    {3500000, B3500000},
    {4000000, B4000000}};

int abcdk_tcattr_serial(int fd, int baudrate, int bits, int parity, int stop,struct termios *old)
{
    struct termios now = {0};
    speed_t speed = B0;

    assert(fd >= 0);

    /*查找波特率配置。*/
    for (int i = 0; i < ABCDK_ARRAY_SIZE(abcdk_tcattr_serial_baudrate_dict); i++)
    {
        if(abcdk_tcattr_serial_baudrate_dict[i].code == baudrate)
        {
            speed = abcdk_tcattr_serial_baudrate_dict[i].speed;
            break;
        }
    }

    /*不支持的波特率。*/
    if(speed == B0)
        return -1;

    /*设置波特率。*/
    cfsetispeed(&now, speed);
    cfsetospeed(&now, speed);

    /**/
    now.c_cflag |= (CLOCAL | CREAD);

    /*清数据位标志*/
    now.c_cflag &= ~CSIZE; 

    /*禁用校验位*/
    now.c_cflag &= ~PARENB;

    /*设置数据位。*/
    if(bits == 5)
        now.c_cflag |= CS5;
    else if(bits == 6)
        now.c_cflag |= CS6;
    else if(bits == 7)
        now.c_cflag |= CS7;
    else /*if(bits == 8)*/
        now.c_cflag |= CS8;

    /*设置校验位。*/
    if (parity == 1)
    {
        /*奇。*/
        now.c_cflag |= PARENB;
        now.c_cflag |= PARODD;
    }
    else if (parity == 2)
    {
        /*偶。*/
        now.c_cflag |= PARENB;
        now.c_cflag &= ~PARODD;
    }


    /*设置停止位。*/
    if(stop == 2)
        now.c_cflag |= CSTOPB;
    else /*if(stop == 1)*/
        now.c_cflag &= ~CSTOPB;

    now.c_cc[VTIME] = 0;
    now.c_cc[VMIN] = 1;

    return abcdk_tcattr_option(fd,&now,old);
    
}

int abcdk_tcattr_transfer(int fd, const void *out, size_t outlen, void *in, size_t inlen,
                          time_t timeout, const void *magic, size_t mglen)
{
    ssize_t wlen, rlen;
    int chk;

    assert(fd >=0 && timeout > 0);

    /*如需要接收数据，则先清除缓存中未处理的数据。*/
    if (in != NULL && inlen > 0)
        tcflush(fd,TCIFLUSH);

    if (out != NULL && outlen > 0)
    {
        wlen = abcdk_transfer(fd, (void *)out, outlen, 2, timeout, NULL, 0);
        if (wlen != outlen)
            return -1;

        /*等待发送完成。*/
        chk = tcdrain(fd);
        if(chk != 0)
            return -1;
    }

    if (in != NULL && inlen > 0)
    {
        rlen = abcdk_transfer(fd, in, inlen, 1, timeout, magic, mglen);
        if (rlen != inlen)
            return -1;
    }

    return 0;
}