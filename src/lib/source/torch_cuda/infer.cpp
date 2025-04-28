/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/infer.h"
#include "infer_forward.hxx"

__BEGIN_DECLS

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)


int abcdk_torch_infer_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk::torch_cuda::infer::forward forward;
    int dla_core;
    const char *data_type_p;
    const char *input_tensor_index_p;
    const char *input_tensor_dims_p;
    int input_tensor_new_dims[4];
    int input_tensor_old_dims[4];
    int input_tensor_idx_dims[4];
    int chk;

    assert(dst != NULL && src != NULL && opt != NULL);

    dla_core = abcdk_option_get_int(opt,"--dla-core",0,-1);
    data_type_p = abcdk_option_get(opt,"--data-type",0,"fp32");
    input_tensor_index_p = abcdk_option_get(opt,"--input-tensor-dims-index",0,"0,1,2,3");
    input_tensor_dims_p = abcdk_option_get(opt,"--input-tensor-dims-size",0,"-1,-1,-1,-1");

    sscanf(input_tensor_index_p, "%d,%d,%d,%d", &input_tensor_idx_dims[0], &input_tensor_idx_dims[1], &input_tensor_idx_dims[2], &input_tensor_idx_dims[3]);
    sscanf(input_tensor_dims_p, "%d,%d,%d,%d", &input_tensor_new_dims[0], &input_tensor_new_dims[1], &input_tensor_new_dims[2], &input_tensor_new_dims[3]);

    chk = forward.create((uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if(chk != 0)
        return -1;

    chk = forward.load_onnx(src);
    if(chk != 0)
        return -1;
        
    initLibNvInferPlugins((void*)forward.logger_ctx(),"");

    forward.get_input_dims(input_tensor_old_dims, input_tensor_idx_dims, 0);

    for (int i = 0; i < 4; i++)
    {
        if (input_tensor_new_dims[i] < 0)
            input_tensor_new_dims[i] = input_tensor_old_dims[i];
    }

    forward.set_input_dims(input_tensor_new_dims, input_tensor_idx_dims, 0);

    return 0;
}

#else // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

int abcdk_torch_infer_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return -1;
}

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

__END_DECLS