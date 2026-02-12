/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_CURL_UTIL_H
#define ABCDK_CURL_UTIL_H

#include "abcdk/curl/curl.h"

__BEGIN_DECLS

/**
 * 下载文件.
 * 
 * @param [in] offset 偏移量.0 无效.
 * @param [in] count 长度.0 无效.
 * @param [in] ctimeout 连接超时.
 * @param [in] stimeout 读写超时.
 * 
 * @return 0 成功, -1 失败(超时, 不支持).
*/
int abcdk_curl_download_fd(int fd,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout);

/**
 * 下载文件.
*/
int abcdk_curl_download_filename(const char *file,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout);


__END_DECLS

#endif //ABCDK_CURL_UTIL_H