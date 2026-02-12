/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SYSTEM_LOCALE_H
#define ABCDK_SYSTEM_LOCALE_H

#include "abcdk/util/general.h"
#include "abcdk/system/proc.h"

__BEGIN_DECLS

/**本地环境配置.*/
int abcdk_locale_setup(const char *lang_codeset, const char *domain_name, const char *domain_path);

__END_DECLS

#endif //ABCDK_SYSTEM_LOCALE_H
