/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_GETPASS_H
#define ABCDK_UTIL_GETPASS_H

#include "abcdk/util/general.h"
#include "abcdk/util/termios.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**获取密码。*/
abcdk_object_t *abcdk_getpass(const char *prompt,FILE *istream);

__END_DECLS

#endif //ABCDK_UTIL_GETPASS_H