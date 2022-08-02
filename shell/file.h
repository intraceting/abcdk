/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_FILE_H
#define ABCDK_SHELL_FILE_H

#include "util/general.h"

__BEGIN_DECLS

/**
 * 谁在占用文件。
 * 
 * @warning 因为要遍历系统运行环境，所以效率非常慢。
 * 
 * @param [in] file 全路径的文件名。
 * @param [in out] pids 保存占用文件的进程PID数组。
 * @param [in] max 进程PID数组容量。
 * 
 * @return >= 0 成功(占用文件的进程数量)，-1 失败(无法查询) ，-2 失败(无法解析数据)。
*/
int abcdk_file_wholockme(const char *file,int pids[],int max);

/**
 * 文件分段。
 * 
 * @note 在原文件所在的目录进行文件分段。
 * 
 * @param [in] file 文件名(包括路径)。
 * @param [in] fmt 分段的文件名格式。仅支持%d或%0Nd格式符，N为数字宽度。
 * @param [in] max 最大分段数量。
 * 
 * @return 0 成功，-1 失败(访问被拒绝)。
 * 
*/
int abcdk_file_subsection(const char *file, const char *fmt, int max);

/**
 * 文件备份。
*/
int abcdk_file_backup(const char *file,);

__END_DECLS

#endif //ABCDK_SHELL_FILE_H