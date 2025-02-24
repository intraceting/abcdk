/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_STITCHER_H
#define ABCDK_OPENCV_STITCHER_H

#include "abcdk/util/object.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

/*简单的全景拼接。*/
typedef struct _abcdk_stitcher abcdk_stitcher_t;

/**销毁。 */
void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx);

/**创㶳。 */
abcdk_stitcher_t *abcdk_stitcher_create();

/**
 * 保存元数据。
 * 
 * @param [in] magic 魔法字符串。用于区别元数据的所有者。
 *
 * @return 0 成功，< 0 失败。
 */
abcdk_object_t *abcdk_stitcher_metadata_dump(abcdk_stitcher_t *ctx, const char *magic);

/**
 * 加载元数据。
 *
 * @return 0 成功，< 0 失败，-127 魔法字符串验证失败。
 */
int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data);


__END_DECLS


#endif //ABCDK_OPENCV_STITCHER_H
