/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_CURL_CURL_H
#define ABCDK_CURL_CURL_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/io.h"
#include "abcdk/util/uri.h"
#include "abcdk/util/trace.h"

#ifdef HAVE_CURL
#include <curl/curl.h>
#endif // HAVE_CURL

__BEGIN_DECLS

#ifdef CURLINC_CURL_H

/**
 * 下载文件。
 * 
 * @param [in] offset 偏移量。0 无效。
 * @param [in] count 长度。0 无效。
 * @param [in] ctimeout 连接超时。
 * @param [in] stimeout 读写超时。
 * 
 * @return 0 成功，-1 失败(超时、不支持)。
*/
int abcdk_curl_download_fd(int fd,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout);

/**
 * 下载文件。
*/
int abcdk_curl_download_filename(const char *file,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout);

#endif //CURLINC_CURL_H

__END_DECLS

#endif //ABCDK_CURL_CURL_H