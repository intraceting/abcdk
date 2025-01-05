/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_IO_H
#define ABCDK_UTIL_IO_H

#include "abcdk/util/defs.h"
#include "abcdk/util/clock.h"

__BEGIN_DECLS

/**
 * 在描述符上等待事件。
 * 
 * @param event 事件。0x01 读，0x02 写。可用“|”运算符同时等待。
 * @param timeout 超时(毫秒)。>= 0 有事件或时间过期，< 0 直到有事件或出错。
 * 
 * @return > 0 有事件(监听的事件)，0 超时，< 0 出错。
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
 * @note 已打开的文件会被关闭，新打开的文件会绑定到fd2句柄。
 * 
 * @param fd2 已打开的句柄。
 * 
 * @return >= 0 成功(fd2)，-1 失败。
 * 
*/
int abcdk_reopen(int fd2,const char *file, int rw, int nonblock, int create);

/**
 * 设置标志。
 * 
 * @note 会覆盖现存的。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_fflag_set(int fd,int flag);

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

/**
 * 从文件加载数据。
 * 
 * @return >= 0 成功(长度)，< 0 失败(文件不存在或没有权限访问)。
*/
ssize_t abcdk_load(const char *file, void *buf, size_t size, size_t offset);

/**
 * 向文件保存数据。
 * 
 * @return >= 0 成功(长度)，< 0 失败(文件不存在或没有权限访问)。
*/
ssize_t abcdk_save(const char *file, const void *buf, size_t size, size_t offset);

/**
 * 向临时文件保存数据。
 * 
 * @return >= 0 成功(长度)，< 0 失败(文件不存在或没有权限访问)。
*/
ssize_t abcdk_save2temp(char *file, const void *buf, size_t size, size_t offset);

/**
 * 从文件中读取一行。
 * 
 * @note 结束读取时line指针要用free释放。
 * @note 两次读取操作，line指针可能会指向不同地址。
 * 
 * @param line 指针的指针。
 * @param delim 分割字符。
 * @param note 注释字符。
 * 
 * @return >0 成功，-1 失败(错误或已到文件末尾)。
*/
ssize_t abcdk_fgetline(FILE *fp, char **line, size_t *len, uint8_t delim, char note);

/**
 * 关闭文件句柄。
*/
void abcdk_fclosep(FILE **fp);

/**
 * 获取文件大小。
*/
int64_t abcdk_fsize(FILE *fp);

/**
 * 设置文件的访问和修改时间。
 * 
 * @param atime 访问时间。
 * @param mtime 访问时间。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_futimens(int fd,const struct timespec *atime,const struct timespec *mtime);

/**
 * 读写数据。
 * 
 * @note 仅支持带有异步标志的句柄。
 * 
 * @param [in] direction 方向。1：输入，2：输出。
 * @param [in] timeout 超时(毫秒)。
 * @param [in] magic 起始码，NULL(0) 忽略。注：仅对输入有效。
 * @param [in] mglen 起始码长度，<= 0 忽略。注：仅对输入有效。
 * 
 * @return > 0 成功(读写的长度)，<= 0 失败(出错、空间不足或已到末尾、超时)。
*/
ssize_t abcdk_transfer(int fd, void *data, size_t size, int direction, time_t timeout,
                       const void *magic, size_t mglen);

__END_DECLS

#endif //ABCDK_UTIL_IO_H