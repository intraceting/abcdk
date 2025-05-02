/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_H
#define ABCDK_TORCH_DNN_H

#include "abcdk/util/option.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/imgutil.h"

__BEGIN_DECLS

/**DNN张量。*/
typedef struct _abcdk_torch_dnn_tensor
{
    /**索引。*/
    int index;

    /**
     * 名字。
     * 
     * @note 复制的指针，对象销毁前有效。
    */
    const char *name_p;

    /**
     * 模式。
     * 
     * 未知：0
     * 输入：1
     * 输出：2
    */
    int mode;

    /**维度。*/
    abcdk_torch_dims_t dims;

    /**
     * 数据。
     * 
     * @note 复制的指针，对象销毁前有效。
     * @note 仅输出有效，并且数据在主机内存。
    */
    const void *data_p;

} abcdk_torch_dnn_tensor;

/**DNN目标。*/
typedef struct _abcdk_torch_dnn_object
{
    /*标签。*/
    int label;

    /*评分。*/
    int score;

    /*坐标。*/
    int x1;
    int y1;
    int x2;
    int y2;

    /*旋转角度。-90 ~ 90 */
    int rotate;

    /*关键点。x,y,v*/
    int nkeypoint;
    int *kp;

    /*特征。*/
    int nfeature;
    float *ft;

    /*分割。*/
    int seg_step;
    uint8_t *seg;

} abcdk_torch_dnn_object_t;

__END_DECLS

#endif // ABCDK_TORCH_DNN_H
