/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_SYSTEM_MEDIUMX_H
#define ABCDK_SYSTEM_MEDIUMX_H

#include "abcdk/system/scsi.h"
#include "abcdk/util/mediumx.h"

__BEGIN_DECLS

/**
 * 格式化.
 * 
 * @param [in] fmt 格式{TEXT(1), XML(2), JSON(3)}.
*/
int abcdk_mediumx_element_status_format(abcdk_tree_t *list,int fmt, FILE* out);                        

__END_DECLS

#endif //ABCDK_SYSTEM_MEDIUMX_H