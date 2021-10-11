/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_GENERAL_H
#define ABCDK_UTIL_GENERAL_H

#include "abcdk-util/defs.h"
#include "abcdk-util/atomic.h"

#ifdef HAVE_OPENMP
#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP
#endif //HAVE_OPENMP

__BEGIN_DECLS

/*------------------------------------------------------------------------------------------------*/

/**
 * 数值对齐。
 * 
 * @param align 对齐量。0,1是等价的。
*/
size_t abcdk_align(size_t size,size_t align);

/**
 * 执行一次。
 * 
 * @param status 状态，一般是静态类型。必须初始化为0。
 * @param routine 执行函数。0 成功，!0 失败。
 * 
 * @return 0 第一次，1 第2~N次，-1 失败。
*/
int abcdk_once(volatile int* status,int (*routine)(void *opaque),void *opaque);

/*------------------------------------------------------------------------------------------------*/

/**
 * 内存申请。
 */
void* abcdk_heap_alloc(size_t size);

/**
 * 内存重新申请。
 */
void* abcdk_heap_realloc(void *buf,size_t size);

/**
 * 内存释放。
 * 
 * @param data 内存的指针。
 */
void abcdk_heap_free(void *data);

/**
 * 内存释放。
 * 
 * @param data 指针的指针。返回时赋值NULL(0)。
 */
void abcdk_heap_free2(void **data);

/**
 * 内存克隆。
 * 
 * @warning 申请内存大小为size+1。
*/
void *abcdk_heap_clone(const void *data, size_t size);

/*------------------------------------------------------------------------------------------------*/

/**
 * 时间戳整形化。
 * 
 * 当精度为纳秒时，公元2444年之前或者时长544年之内有效。
 * 
 * @param precision 精度。0～9。
*/
uint64_t abcdk_time_clock2kind(struct timespec* ts,uint8_t precision);

/**
 * 获取时间戳并整形化。
 * 
 * @param id CLOCK_* in time.h
*/
uint64_t abcdk_time_clock2kind_with(clockid_t id ,uint8_t precision);

/**
 * 本地时间转国际时间。
 * 
 * @param reverse 0 本地转国际，!0 国际转本地。
*/
struct tm* abcdk_time_local2utc(struct tm* dst,const struct tm* src,int reverse);

/**
 * 获取自然时间。
 * 
 * @param utc 0 获取本地，!0 获取国际。
*/
struct tm* abcdk_time_get(struct tm* tm,int utc);

/**
 * 秒转自然时间。
 * 
 * @param sec 秒。
 * @param utc 0 转本地，!0 转国际。
*/
struct tm* abcdk_time_sec2tm(struct tm* tm,time_t sec,int utc);

/*------------------------------------------------------------------------------------------------*/

/**
 * 检查字符是否为8进制数字字符。
 * 
 * @return !0 是，0 否。
*/
int abcdk_isodigit(int c);

/*------------------------------------------------------------------------------------------------*/

/**
 * 字符串克隆。
*/
#define abcdk_strdup(str) abcdk_heap_clone(str,strlen(str)+1);

/**
 * 字符串查找。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return !NULL(0) 匹配字符串的首地址， NULL(0) 未找到。
*/
const char* abcdk_strstr(const char *str, const char *sub,int caseAb);

/**
 * 字符串查找。
 * 
 * @return !NULL(0) 匹配字符串尾地址之后，NULL(0) 未找到。
*/
const char* abcdk_strstr_eod(const char *str, const char *sub,int caseAb);

/**
 * 字符串比较。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return 1: s1 > s2, 0: s1 = s2, -1: s1 < s2
*/
int abcdk_strcmp(const char *s1, const char *s2,int caseAb);

/**
 * 字符串比较。
 * 
 * @param caseAb 0 不区分大小写，!0 区分大小写。
 * 
 * @return 1: s1 > s2, 0: s1 = s2, -1: s1 < s2
*/
int abcdk_strncmp(const char *s1, const char *s2,size_t len,int caseAb);

/**
 * 字符串修剪。
 * 
 * @param isctype_cb 字符比较函数。返回值：!0 是，0 否。isctype等函数在ctype.h文件中。
 * @param where 0 右端，1 左端，2 两端。
 * 
*/
char* abcdk_strtrim(char* str,int (*isctype_cb)(int c),int where);

/**
 * 字符串分割。
 * 
 * @param str 待分割字符串的指针。可能会被修改。
 * @param delim 分割字符的串指针。全字匹配，并区分大小写。
 * @param saveptr 临时的指针。不支持访问。
 * 
 * @return !NULL(0) 分割后字符串的指针，NULL(0) 结束。
*/
char *abcdk_strtok(char *str, const char *delim, char **saveptr);

/**
 * 检测字符串中的字符类型。
 * 
 * @param isctype_cb 字符比较函数。返回值：!0 是，0 否。isctype等函数在ctype.h文件中。
 * 
 * @return !0 通过，0 未通过。
*/
int abcdk_strtype(const char* str,int (*isctype_cb)(int c));

/**
 * 字符串查找并替换。
 * 
 * @return  !NULL(0) 成功(指针需要用abcdk_heap_free去释放)， NULL(0) 失败。
*/
char* abcdk_strrep(const char* str,const char *src, const char *dst, int caseAb);

/*------------------------------------------------------------------------------------------------*/

/**
 * 字符串匹配。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_fnmatch(const char *str,const char *wildcard,int caseAb,int ispath);

/*------------------------------------------------------------------------------------------------*/

/**
 * BKDR32
 * 
*/
uint32_t abcdk_hash_bkdr(const void* data,size_t size);

/**
 * BKDR64
 * 
*/
uint64_t abcdk_hash_bkdr64(const void* data,size_t size);

/*------------------------------------------------------------------------------------------------*/

/**
 * 字节序检测
 * 
 * @param big 0 检测是否为小端字节序，!0 检测是否为大端字节序。
 * 
 * @return 0 否，!0 是。
 */
int abcdk_endian_check(int big);

/**
 * 字节序交换。
 * 
 * @return dst
*/
uint8_t* abcdk_endian_swap(uint8_t* dst,int len);

/**
 * 大端字节序转本地字节序。
 * 
 * 如果本地是大端字节序，会忽略。
*/
uint8_t* abcdk_endian_b_to_h(uint8_t* dst,int len);

/**
 * 16位整型数值，大端字节序转本地字节序。
*/
uint16_t abcdk_endian_b_to_h16(uint16_t src);

/**
 * 24位整型数值，大端字节序转本地字节序。
*/
uint32_t abcdk_endian_b_to_h24(const uint8_t* src);

/**
 * 32位整型数值，大端字节序转本地字节序。
*/
uint32_t abcdk_endian_b_to_h32(uint32_t src);

/**
 * 64位整型数值，大端字节序转本地字节序。
*/
uint64_t abcdk_endian_b_to_h64(uint64_t src);

/**
 * 本地字节序转大端字节序。
 * 
 * 如果本地是大端字节序，会忽略。
*/
uint8_t* abcdk_endian_h_to_b(uint8_t* dst,int len);

/**
 * 16位整型数值，本地字节序转大端字节序。
*/
uint16_t abcdk_endian_h_to_b16(uint16_t src);

/**
 * 24位整型数值，本地字节序转大端字节序。
*/
uint8_t* abcdk_endian_h_to_b24(uint8_t* dst,uint32_t src);

/**
 * 32位整型数值，本地字节序转大端字节序。
*/
uint32_t abcdk_endian_h_to_b32(uint32_t src);

/**
 * 64位整型数值，本地字节序转大端字节序。
*/
uint64_t abcdk_endian_h_to_b64(uint64_t src);

/**
 * 小端字节序转本地字节序。
 * 
 * 如果本地是小端字节序，会忽略。
*/
uint8_t* abcdk_endian_l_to_h(uint8_t* dst,int len);

/**
 * 16位整型数值，小端字节序转本地字节序。
*/
uint16_t abcdk_endian_l_to_h16(uint16_t src);

/**
 * 24位整型数值，小端字节序转本地字节序。
*/
uint32_t abcdk_endian_l_to_h24(uint8_t* src);

/**
 * 32位整型数值，小端字节序转本地字节序。
*/
uint32_t abcdk_endian_l_to_h32(uint32_t src);

/**
 * 64位整型数值，小端字节序转本地字节序。
*/
uint64_t abcdk_endian_l_to_h64(uint64_t src);

/**
 * 本地字节序转小端字节序。
 * 
 * 如果本地是小端字节序，会忽略。
*/
uint8_t* abcdk_endian_h_to_l(uint8_t* dst,int len);

/**
 * 16位整型数值，本地字节序转小端字节序。
*/
uint16_t abcdk_endian_h_to_l16(uint16_t src);

/**
 * 24位整型数值，本地字节序转小端字节序。
*/
uint8_t* abcdk_endian_h_to_l24(uint8_t* dst,uint32_t src);

/**
 * 32位整型数值，本地字节序转小端字节序。
*/
uint32_t abcdk_endian_h_to_l32(uint32_t src);

/**
 * 64位整型数值，本地字节序转小端字节序。
*/
uint64_t abcdk_endian_h_to_l64(uint64_t src);

/*------------------------------------------------------------------------------------------------*/

/** 
 * 布隆-插旗
 * 
 * @param size 池大小(Bytes)
 * @param number 编号。有效范围：0 ～ size*8-1。
 * 
 * @return 0 成功，1 成功（或重复操作）。
*/
int abcdk_bloom_mark(uint8_t* pool,size_t size,size_t number);

/** 
 * 布隆-拔旗
 * 
 * @param size 池大小(Bytes)
 * @param number 编号。有效范围：0 ～ size*8-1。
 * 
 * @return 0 成功，1 成功（或重复操作）。
 * 
*/
int abcdk_bloom_unset(uint8_t* pool,size_t size,size_t number);

/**
 * 布隆-过滤
 * 
 * @param size 池大小(Bytes)
 * @param number 编号。有效范围：0 ～ size*8-1。
 * 
 * @return 0 不存在，1 已存在。
*/
int abcdk_bloom_filter(uint8_t* pool,size_t size,size_t number);

/*------------------------------------------------------------------------------------------------*/

/**
 * 拼接目录。
 * 
 * 自动检查前后的'/'字符，接拼位置只保留一个'/'字符，或自动添加一个'/'字符。
 * 
 * @warning 要有足够的可用空间，不然会溢出。
*/
char *abcdk_dirdir(char *path,const char *suffix);

/**
 * 创建目录。
 * 
 * 支持创建多级目录。如果末尾不是'/'，则最后一级的名称会被当做文件名而忽略。
*/
void abcdk_mkdir(const char *path,mode_t mode);

/**
 * 截取目录。
 * 
 * 最后一级的名称会被裁剪，并且无论目录结构是否真存在都会截取。 
*/
char *abcdk_dirname(char *dst, const char *src);

/**
 * 截取目录或文件名称。
 * 
 * @note 最后一级的名称'/'(包括)之前的会被裁剪，并且无论目录结构是否真存在都会截取。 
*/
char *abcdk_basename(char *dst, const char *src);

/**
 * 美化目录。
 * 
 * @note 不会检测目录结构是否存在。
 * 
 * 例：/aaaa/bbbb/../ccc -> /aaaa/ccc
 * 例：/aaaa/bbbb/./ccc -> /aaaa/bbbb/ccc
*/
char *abcdk_dirnice(char *dst, const char *src);

/**
 * 修补文件或目录的绝对路径。
 * 
 * @note 不会检测目录结构是否存在。
 * 
 * @param file 文件或目录的指针。
 * @param path 路径的指针，NULL(0) 当前工作路径。
*/
char *abcdk_abspath(char *buf, const char *file, const char *path);

/*------------------------------------------------------------------------------------------------*/

/**
 * 获取当前程序的完整路径和文件名。
*/
char* abcdk_proc_pathfile(char* buf);

/**
 * 获取当前程序的完整路径。
 * 
 * @param append 拼接目录或文件名。NULL(0) 忽略。
 * 
*/
char* abcdk_proc_dirname(char* buf,const char* append);

/**
 * 获取当前程序的文件名。
*/
char* abcdk_proc_basename(char* buf);

/**
 * 单实例模式运行。
 * 
 * 文件句柄在退出前不要关闭，否则会使文件解除锁定状态。
 * 
 * 进程ID以十进制文本格式写入文件，例：2021 。
 * 
 * @param pid 正在运行的进程ID，当接口返回时填写。NULL(0) 忽略。
 * 
 * @return >= 0 成功(文件句柄，当前进程是唯一进程)，-1 失败(已有实例正在运行)。
*/
int abcdk_proc_singleton(const char* lockfile,int* pid);

/*------------------------------------------------------------------------------------------------*/

/**
 * 获取当前用户的运行路径。
 * 
 * 可能不存在，使用前最好检查一下。
 *
 * /var/run/user/$UID/
 * 
 * @param append 拼接目录或文件名。NULL(0) 忽略。
*/
char* abcdk_user_dirname(char* buf,const char* append);

/*------------------------------------------------------------------------------------------------*/

/**
 * 在描述符上等待事件。
 * 
 * @param event 事件。0x01 读，0x02 写，0x03读写。
 * @param timeout 超时(毫秒)。>= 0 有事件或时间过期，< 0 直到有事件或出错。
 * 
 * @return > 0 有事件，0 超时，< 0 出错。
*/
int abcdk_poll(int fd, int event,time_t timeout);

/**
 * 写数据。
 * 
 * @return > 0 成功(写入的长度)，<= 0 失败(空间不足或出错)。
*/
ssize_t abcdk_write(int fd, const void *data, size_t size);

/**
 * 读数据。
 * 
 * @return > 0 成功(读取的长度)，<= 0 失败(已到末尾或出错)。
*/
ssize_t abcdk_read(int fd, void *data, size_t size);

/**
 * 关闭文件句柄。
*/
void abcdk_closep(int *fd);

/**
 * 打开文件。
 * 
 * @return >= 0 成功(句柄)，-1 失败。
 * 
*/
int abcdk_open(const char *file, int rw, int nonblock, int create);

/**
 * 关联文件到已经打开的句柄。
 * 
 * 已打开的文件会被关闭，新打开的文件会绑定到fd2句柄。
 * 
 * @param fd2 已打开的句柄。
 * 
 * @return fd2 成功，-1 失败。
 * 
*/
int abcdk_reopen(int fd2,const char *file, int rw, int nonblock, int create);

/**
 * 获取标志。
 * 
 * @return !-1 成功(标志)，-1 失败。
*/
int abcdk_fflag_get(int fd);

/**
 * 添加标志。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_fflag_add(int fd,int flag);

/**
 * 删除标志。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_fflag_del(int fd,int flag);

/*------------------------------------------------------------------------------------------------*/

/**
 * 创建子进程，用于执行shell。
 *
 * @param cmd 命令行字符串指针。
 * @param envp 环境变量的组数指针，不影响父进程。{"KEY=VALUE","...",NULL}。
 * @param stdin_fd 输入句柄指针，NULL(0) 忽略。
 * @param stdout_fd 输出句柄指针，NULL(0) 忽略。
 * @param stderr_fd 出错句柄指针，NULL(0) 忽略。
 * 
 * @return 子进程ID 成功，-1 失败。
*/
pid_t abcdk_popen(const char* cmd,char * const envp[],int* stdin_fd, int* stdout_fd, int* stderr_fd);


/*------------------------------------------------------------------------------------------------*/

/**
 * 打开共享内存文件。
 *
 * 通常是在'/dev/shm/'目录内创建。
 * 
 * @return >= 0 句柄，-1 失败。
*/
int abcdk_shm_open(const char* name,int rw, int create);

/**
 * 删除共享内存文件。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_shm_unlink(const char* name);

/*------------------------------------------------------------------------------------------------*/

/**
 * 日志初始化。
 * 
 * 只能执行一次。
 * 
 * @param ident NULL(0) 进程名做为标识，!NULL(0) 自定义标识。
 * @param level 记录级别。LOG_*宏定义在syslog.h文件中。
 * @param copy2stderr 0 仅记录，!0 复制到stderr。
 * 
 */
void abcdk_openlog(const char *ident,int level,int copy2stderr);


/*------------------------------------------------------------------------------------------------*/

/**
 * 二进制转十六进制。
 * 
 * @param dst 十六进制数据的指针。可用空间至少是二进制数据长度的两倍。
 * @param src 二进制数的指针。
 * @param size 二进制数据的长度。
 * @param ABC 0 小写，!0 大写。
 * 
 * @return !NULL(0) 成功(十六进制数据的指针)，NULL(0) 失败。
*/
char *abcdk_bin2hex(char* dst,const void *src,size_t size,int ABC);

/**
 * 十六进制转二进制。
 * 
 * @param dst 二进制数据的指针。可用空间至少是十六进制数据长度的二分之一。
 * @param src 十六进制数的指针。
 * @param size 十六进制数据的长度。
 * 
 * @return !NULL(0) 成功(二进制数据的指针)，NULL(0) 失败。
*/
void *abcdk_hex2bin(void *dst,const char* src,size_t size);

/*------------------------------------------------------------------------------------------------*/

/**
 * 循环移位。
 *  
 * @param size 数据长度(节字)。
 * @param bits 移动位数。
 * @param direction 1 由低向高，2 由高向低。
*/
void *abcdk_cyclic_shift(void *data,size_t size,size_t bits, int direction);

/*------------------------------------------------------------------------------------------------*/

__END_DECLS

#endif //ABCDK_UTIL_GENERAL_H