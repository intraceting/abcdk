/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/termios.h"

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

int abcdk_tcattr_serial(int fd, int baudrate, int bits, int parity, int stop,struct termios *old)
{
    struct termios now = {0};

    assert(fd >= 0);

    /*设置波特率。*/
    switch (baudrate)
    {
        case 2400:
            cfsetispeed(&now, B2400);
            cfsetospeed(&now, B2400);
            break;
        case 4800:
            cfsetispeed(&now, B4800);
            cfsetospeed(&now, B4800);
            break;
        case 19200:
            cfsetispeed(&now, B19200);
            cfsetospeed(&now, B19200);
            break;
        case 38400:
            cfsetispeed(&now, B38400);
            cfsetospeed(&now, B38400);
            break;
        case 57600:
            cfsetispeed(&now, B57600);
            cfsetospeed(&now, B57600);
            break;
        case 115200:
            cfsetispeed(&now, B115200);
            cfsetospeed(&now, B115200);
            break;
        case 230400:
            cfsetispeed(&now, B230400);
            cfsetospeed(&now, B230400);
            break;
        case 9600:
        default:
            cfsetispeed(&now, B9600);
            cfsetospeed(&now, B9600);
            break;
    }

    now.c_cflag |= (CLOCAL | CREAD);

    /*清数据位标志*/
    now.c_cflag &= ~CSIZE; 

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
        now.c_iflag |= (INPCK | ISTRIP);
    }
    else if (parity == 2)
    {
        /*偶。*/
        now.c_iflag |= (INPCK | ISTRIP);
        now.c_cflag |= PARENB;
        now.c_cflag &= ~PARODD;
    }
    else
    {
        /*无。*/
        now.c_cflag &= ~PARENB;
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