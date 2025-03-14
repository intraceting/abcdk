/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_CHARSET_H
#define ABCDK_UTIL_CHARSET_H

#include "abcdk/util/general.h"


__BEGIN_DECLS

/**
 * 检验UTF-8编码。
 * 
 * @note 不识别BOM头。
 * 
 * 1 bytes: 0xxxxxxx ;
 * 2 bytes: 110xxxxx 10xxxxxx ;
 * 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx ;
 * 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 5 bytes: 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 6 bytes: 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx ;
 * 
 * @return 符合的长度(字节)。
*/
ssize_t abcdk_verify_utf8(const void *data,size_t max);

__END_DECLS

#endif //ABCDK_UTIL_CHARSET_H