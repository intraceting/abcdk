/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_DEFS_H
#define ABCDK_UTIL_DEFS_H

/**/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif //_GNU_SOURCE

/**/
#include <endian.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <syslog.h>
#include <pthread.h>
#include <ctype.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <libgen.h>
#include <fnmatch.h>
#include <limits.h>
#include <dirent.h>
#include <poll.h>
#include <iconv.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <tar.h>
#include <termios.h>
#include <dlfcn.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>
#include <sys/mtio.h>
#include <sys/un.h>
#include <sys/time.h>
#include <sys/vfs.h>
#include <scsi/scsi.h>
#include <scsi/scsi_ioctl.h>
#include <scsi/sg.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <net/ethernet.h>

/** 转字符串。*/
#define ABCDK_STR_NOT_USE(s) #s
#define ABCDK_STR(s) ABCDK_STR_NOT_USE(s)

/**
 * 转换指针类型。
 * 
 * @note 支持针地址偏移(Bytes)。
 * 
*/
#define ABCDK_PTR2PTR(T, P, OF) ((T *)(((char *)(P)) + (OF)))

/**/
#define ABCDK_PTR2VPTR(P, OF) ABCDK_PTR2PTR(void, P, OF)
#define ABCDK_PTR2I8PTR(P, OF) ABCDK_PTR2PTR(int8_t, P, OF)
#define ABCDK_PTR2U8PTR(P, OF) ABCDK_PTR2PTR(uint8_t, P, OF)
#define ABCDK_PTR2I16PTR(P, OF) ABCDK_PTR2PTR(int16_t, P, OF)
#define ABCDK_PTR2U16PTR(P, OF) ABCDK_PTR2PTR(uint16_t, P, OF)
#define ABCDK_PTR2I32PTR(P, OF) ABCDK_PTR2PTR(int32_t, P, OF)
#define ABCDK_PTR2U32PTR(P, OF) ABCDK_PTR2PTR(uint32_t, P, OF)
#define ABCDK_PTR2I64PTR(P, OF) ABCDK_PTR2PTR(int64_t, P, OF)
#define ABCDK_PTR2U64PTR(P, OF) ABCDK_PTR2PTR(uint64_t, P, OF)
#define ABCDK_PTR2SIZEPTR(P, OF) ABCDK_PTR2PTR(ssize_t, P, OF)
#define ABCDK_PTR2USIZEPTR(P, OF) ABCDK_PTR2PTR(size_t, P, OF)

/**
 * 转换指针对象。
 * 
 * @note 支持针地址偏移(Bytes)。
 * 
*/
#define ABCDK_PTR2OBJ(T, P, OF) (*ABCDK_PTR2PTR(T, P, OF))

/**/
#define ABCDK_PTR2I8(P, OF) ABCDK_PTR2OBJ(int8_t, P, OF)
#define ABCDK_PTR2U8(P, OF) ABCDK_PTR2OBJ(uint8_t, P, OF)
#define ABCDK_PTR2I16(P, OF) ABCDK_PTR2OBJ(int16_t, P, OF)
#define ABCDK_PTR2U16(P, OF) ABCDK_PTR2OBJ(uint16_t, P, OF)
#define ABCDK_PTR2I32(P, OF) ABCDK_PTR2OBJ(int32_t, P, OF)
#define ABCDK_PTR2U32(P, OF) ABCDK_PTR2OBJ(uint32_t, P, OF)
#define ABCDK_PTR2I64(P, OF) ABCDK_PTR2OBJ(int64_t, P, OF)
#define ABCDK_PTR2U64(P, OF) ABCDK_PTR2OBJ(uint64_t, P, OF)
#define ABCDK_PTR2SIZE(P, OF) ABCDK_PTR2OBJ(ssize_t, P, OF)
#define ABCDK_PTR2USIZE(P, OF) ABCDK_PTR2OBJ(size_t, P, OF)

/** 数值比较，返回最大值。*/
#define ABCDK_MAX(A, B) (((A) > (B)) ? (A) : (B))

/** 数值比较，返回最小值。*/
#define ABCDK_MIN(A, B) (((A) < (B)) ? (A) : (B))

/** 规划数值到区间内(包括两端极值)。*/
#define ABCDK_CLAMP(V, A, B) \
    ABCDK_MIN(ABCDK_MAX((A), (B)), ABCDK_MAX(ABCDK_MIN((A), (B)), (V)))

/** 交换两个数值变量的值。*/
#define ABCDK_INTEGER_SWAP(A, B) ( \
    {                              \
        (A) ^= (B);                \
        (B) ^= (A);                \
        (A) ^= (B);                \
    })

/** 设置出错码，并返回。*/
#define ABCDK_ERRNO_AND_RETURN0(E) ( \
    {                                \
        errno = (E);                 \
        return;                      \
    })

/** 设置出错码，并返回值。*/
#define ABCDK_ERRNO_AND_RETURN1(E, V) ( \
    {                                   \
        errno = (E);                    \
        return (V);                     \
    })

/** 设置出错码，并跳转。*/
#define ABCDK_ERRNO_AND_GOTO1(E, M) ( \
    {                                 \
        errno = (E);                  \
        goto M;                       \
    })

/** 计算数组大小。*/
#define ABCDK_ARRAY_SIZE(V) (sizeof((V)) / sizeof((V)[0]))

/** 终端字符颜色设置。*/
#define ABCDK_ANSI_COLOR_RESET "\x1b[0m"
#define ABCDK_ANSI_COLOR_RED "\x1b[31m"
#define ABCDK_ANSI_COLOR_GREEN "\x1b[32m"
#define ABCDK_ANSI_COLOR_YELLOW "\x1b[33m"
#define ABCDK_ANSI_COLOR_BLUE "\x1b[34m"
#define ABCDK_ANSI_COLOR_MAGENTA "\x1b[35m"
#define ABCDK_ANSI_COLOR_CYAN "\x1b[36m"

/** 4字节TAG生成器(整型数值以大端字节序存储)。*/
#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ABCDK_FOURCC_MKTAG(a, b, c, d) ((a) | ((b) << 8) | ((c) << 16) | ((uint32_t)(d) << 24))
#else
#define ABCDK_FOURCC_MKTAG(a, b, c, d) ((d) | ((c) << 8) | ((b) << 16) | ((uint32_t)(a) << 24))
#endif

/** 断言提示。 */
#define ABCDK_ASSERT(expr, tips) \
    ((expr) ? (void)(0) : ({fprintf(stderr,"%s(%d): %s: %s\n",__FILE__, __LINE__,__FUNCTION__,#tips);abort(); }))

#endif //ABCDK_UTIL_DEFS_H