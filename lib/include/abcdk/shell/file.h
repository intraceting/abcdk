/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SHELL_FILE_H
#define ABCDK_SHELL_FILE_H

#include "abcdk/util/general.h"
#include "abcdk/util/popen.h"
#include "abcdk/util/io.h"
#include "abcdk/util/path.h"
#include "abcdk/util/dirent.h"

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
 * @code
 * //1 
 * uint64_t prev2next = 0;
 * abcdk_file_segment("/aaa/bbb.log","/aaa/bbb.%llu.log",1,100,&prev2next);
 * //2
 * uint64_t prev2next = 1;
 * abcdk_file_segment("/aaa/bbb.log","/aaa/bbb.log.%05llu",100,33,&prev2next);
 * //3
 * uint64_t prev2next = 1000;
 * abcdk_file_segment("/aaa/bbb.log","/aaa/%llu.bbb.log",123,44,&prev2next);
 * //4
 * uint64_t prev2next = 555;
 * abcdk_file_segment("/aaa/bbb.log","/aaa/%010llu.bbb.log",55,55,&prev2next);
 * @endcode
 * 
 * @param [in] src 源文件名(包括路径)，NULL(0) 忽略。
 * @param [in] dst 目标文件名(包括路径)。注：文件名仅支持一个数值格式控制符。
 * @param [in] winsize 冗余窗口。1~65535。
 * @param [in] start 起始编号。1~...
 * @param [in out] pos 游标。pos[0] > pos[1] 查找历史编号，接续生产。
 * 
 * @return 0 成功，-1 删除历史文件失败(无权限)，-2 分段重命名失败(无权限或不存在)。
 * 
*/
int abcdk_file_segment(const char *src,const char *dst, uint16_t winsize, uint64_t start, uint64_t pos[2]);


__END_DECLS

#endif //ABCDK_SHELL_FILE_H