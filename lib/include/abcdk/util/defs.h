/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_DEFS_H
#define ABCDK_UTIL_DEFS_H

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
#include <syslog.h>
#include <pwd.h>
#include <locale.h>
#include <malloc.h>
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
#include <sys/sendfile.h>
#include <scsi/scsi.h>
#include <scsi/scsi_ioctl.h>
#include <scsi/sg.h>
#include <sys/syscall.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/ip_icmp.h>
#include <netinet/icmp6.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <net/ethernet.h>

#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP


/** 转字符串。*/
#define ABCDK_STR_NOT_USE(s) #s
#define ABCDK_STR(s) ABCDK_STR_NOT_USE(s)

/**
 * 指针转换类型。
 * 
 * @param T 类型。
 * @param P 指针。
 * @param F 偏移量(字节)。
 * 
*/
#define ABCDK_PTR2PTR(T, P, F) ((T *)(((char *)(P)) + (F)))

/**/
#define ABCDK_PTR2VPTR(P, F) ABCDK_PTR2PTR(void, P, F)
#define ABCDK_PTR2I8PTR(P, F) ABCDK_PTR2PTR(int8_t, P, F)
#define ABCDK_PTR2U8PTR(P, F) ABCDK_PTR2PTR(uint8_t, P, F)
#define ABCDK_PTR2I16PTR(P, F) ABCDK_PTR2PTR(int16_t, P, F)
#define ABCDK_PTR2U16PTR(P, F) ABCDK_PTR2PTR(uint16_t, P, F)
#define ABCDK_PTR2I32PTR(P, F) ABCDK_PTR2PTR(int32_t, P, F)
#define ABCDK_PTR2U32PTR(P, F) ABCDK_PTR2PTR(uint32_t, P, F)
#define ABCDK_PTR2I64PTR(P, F) ABCDK_PTR2PTR(int64_t, P, F)
#define ABCDK_PTR2U64PTR(P, F) ABCDK_PTR2PTR(uint64_t, P, F)
#define ABCDK_PTR2SIZEPTR(P, F) ABCDK_PTR2PTR(ssize_t, P, F)
#define ABCDK_PTR2USIZEPTR(P, F) ABCDK_PTR2PTR(size_t, P, F)

/**
 * 指针转换对象。
 * 
 * @param T 类型。
 * @param P 指针。
 * @param F 偏移量(字节)。
 * 
*/
#define ABCDK_PTR2OBJ(T, P, F) (*ABCDK_PTR2PTR(T, P, F))

/**/
#define ABCDK_PTR2I8(P, F) ABCDK_PTR2OBJ(int8_t, P, F)
#define ABCDK_PTR2U8(P, F) ABCDK_PTR2OBJ(uint8_t, P, F)
#define ABCDK_PTR2I16(P, F) ABCDK_PTR2OBJ(int16_t, P, F)
#define ABCDK_PTR2U16(P, F) ABCDK_PTR2OBJ(uint16_t, P, F)
#define ABCDK_PTR2I32(P, F) ABCDK_PTR2OBJ(int32_t, P, F)
#define ABCDK_PTR2U32(P, F) ABCDK_PTR2OBJ(uint32_t, P, F)
#define ABCDK_PTR2I64(P, F) ABCDK_PTR2OBJ(int64_t, P, F)
#define ABCDK_PTR2U64(P, F) ABCDK_PTR2OBJ(uint64_t, P, F)
#define ABCDK_PTR2SIZE(P, F) ABCDK_PTR2OBJ(ssize_t, P, F)
#define ABCDK_PTR2USIZE(P, F) ABCDK_PTR2OBJ(size_t, P, F)

/** 
 * 数值比较，返回最大值。
 * 
 * @note 不同类的数值，无法返回正确的结果。
*/
#define ABCDK_MAX(A, B) \
    ((A) > (B) ? (A) : (B))

/** 
 * 数值比较，返回最小值。
 * 
 * @note 不同类的数值，无法返回正确的结果。
*/
#define ABCDK_MIN(A, B) \
    ((A) < (B) ? (A) : (B))

/** 
 * 规划数值到区间内(包括两端极值)。
 * 
 * @warning 不同类型的数值，无法返回正确的结果。
*/
#define ABCDK_CLAMP(V, A, B) \
    ((V) < (A)?(A):((V) > (B)?(B):(V)))

/** 
 * 交换两个数值变量的值。
 * 
 * @warning 当相同的数值用下面方法交换时数值会变成零(0)，因此忽略相同的数值交换请求。
*/
#define ABCDK_INTEGER_SWAP(A, B) ( \
    {                              \
        if((A) != (B))             \
        {                          \
            (A) ^= (B);            \
            (B) ^= (A);            \
            (A) ^= (B);            \
        }                          \
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
    ((expr) ? (void)(0) : ({fprintf(stderr,"%s(%d): %s\n",__FUNCTION__, __LINE__,#tips);fflush(stderr);abort(); }))

/** 高版本。*/
#define ABCDK_VERSION_AT_LEAST(max, min, x, y) ((max) > (x) || (max) == (x) && (min) >= (y))

/** 低版本。*/
#define ABCDK_VERSION_AT_MOST(max, min, x, y) ((max) < (x) || (max) == (x) && (min) <= (y))

/** GCC版本。*/
#ifdef __GNUC__
#define ABCDK_GCC_VERSION_AT_LEAST(x, y) ABCDK_VERSION_AT_LEAST(__GNUC__, __GNUC_MINOR__, x, y)
#define ABCDK_GCC_VERSION_AT_MOST(x, y) ABCDK_VERSION_AT_MOST(__GNUC__, __GNUC_MINOR__, x, y)
#else
#define ABCDK_GCC_VERSION_AT_LEAST(x, y) 0
#define ABCDK_GCC_VERSION_AT_MOST(x, y) 0
#endif

#if ABCDK_GCC_VERSION_AT_LEAST(3, 1)
#define ABCDK_DEPRECATED __attribute__((deprecated))
#else
#define ABCDK_DEPRECATED
#endif

#endif //ABCDK_UTIL_DEFS_H