/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_HTTP_RESPONSE_H
#define ABCDK_HTTP_RESPONSE_H

#include "util/general.h"
#include "util/getargs.h"

/** 
 * 翻译状态码描述。
 * 
 * @return !NULL(0) 描述字符串指针，NULL(0) 状态码未找到。
*/
const char *abcdk_http_resp_status_desc(uint32_t code);

#endif //ABCDK_HTTP_RESPONSE_H