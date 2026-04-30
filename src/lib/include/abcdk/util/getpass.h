/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_GETPASS_H
#define ABCDK_UTIL_GETPASS_H

#include "abcdk/util/object.h"

__BEGIN_DECLS

/**
 * 从特定输入流读取密钥.
 * 
 * @param [in] prompt 密钥提示.
*/
abcdk_object_t *abcdk_getpass(FILE *fp, const char *prompt, ...);

/**
 * 读取密钥回调函数.
 *
 * @return > 0 密钥长度, <= 0 出错.
 */
typedef int (*abcdk_get_password_cb)(char *buf, int size, int enc_or_dec, void *opaque);

/**
 * 从标准输入流读取密钥.
 */
int abcdk_get_password(char *buf, int size, int enc_or_dec, void *opaque);

__END_DECLS

#endif //ABCDK_UTIL_GETPASS_H