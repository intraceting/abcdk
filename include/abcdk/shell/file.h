/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_FILE_H
#define ABCDK_SHELL_FILE_H

#include "abcdk/util/general.h"
#include "abcdk/util/popen.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"

__BEGIN_DECLS

/**
 * 查询哪个进程占用文件。
 * 
 * @note 因为要遍历系统运行环境，所以效率非常慢。
 * 
 * @param [in] file 全路径的文件名。
 * @param [in out] pids 保存占用文件的进程PID数组。
 * @param [in] max 进程PID数组容量。
 * 
 * @return >= 0 成功(占用文件的进程数量)，-1 失败(无法查询) ，-2 失败(无法解析数据)。
*/
int abcdk_file_wholockme(const char *file,int pids[],int max);

/**
 * 文件分段存储。
 * 
 * @note 在原文件所在的目录进行文件分段存储。
 * 
 * @code
 * //1
 * abcdk_file_segment("/aaa/bbb.log","/aaa/bbb.%d.log",10);
 * //2
 * abcdk_file_segment("/aaa/bbb.log","/aaa/bbb.log.%04d",10);
 * //3
 * abcdk_file_segment("/aaa/bbb.log","/aaa/%d.bbb.log",10);
 * //4
 * abcdk_file_segment("/aaa/bbb.log","/aaa/%04d.bbb.log",10);
 * @endcode
 * 
 * @param [in] src 源文件名(包括路径)。
 * @param [in] dst 目标文件名(包括路径)。注：文件名仅支持一个数值格式控制符。
 * @param [in] max 最大分段数量。
 * 
 * @return 0 成功，-1 失败(访问被拒绝)。
 * 
*/
int abcdk_file_segment(const char *src, const char *dst, int max);

__END_DECLS

#endif //ABCDK_SHELL_FILE_H