/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_GETPASS_H
#define ABCDK_UTIL_GETPASS_H

#include "abcdk/util/general.h"
#include "abcdk/util/termios.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**获取密码。*/
abcdk_object_t *abcdk_getpass(FILE *istream,const char *prompt,...);

__END_DECLS

#endif //ABCDK_UTIL_GETPASS_H