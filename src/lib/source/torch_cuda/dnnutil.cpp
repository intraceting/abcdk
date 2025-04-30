/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/dnnutil.h"
#include "dnn_forward.hxx"

__BEGIN_DECLS

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)


int abcdk_torch_dnn_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk::torch_cuda::dnn::forward forward;
    int dla_core;
    const char *data_type_p;
    const char *input_tensor_index_p;
    const char *input_tensor_dims_p;
    nvinfer1::Dims input_tensor_new_dims = {0};
    nvinfer1::Dims input_tensor_old_dims = {0};
    nvinfer1::Dims input_tensor_idx_dims = {0};
    int chk;

    assert(dst != NULL && src != NULL && opt != NULL);

    input_tensor_new_dims.nbDims = input_tensor_old_dims.nbDims = input_tensor_idx_dims.nbDims = nvinfer1::Dims::MAX_DIMS;

    dla_core = abcdk_option_get_int(opt,"--dla-core",0,-1);
    data_type_p = abcdk_option_get(opt,"--data-type",0,"fp32");
    input_tensor_index_p = abcdk_option_get(opt,"--input-tensor-dims-index",0,"0,1,2,3,4,5,6,7");
    input_tensor_dims_p = abcdk_option_get(opt,"--input-tensor-dims-size",0,"-1,-1,-1,-1,-1,-1,-1,-1");

    sscanf(input_tensor_index_p, "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld",
           &input_tensor_idx_dims.d[0], &input_tensor_idx_dims.d[1], &input_tensor_idx_dims.d[2], &input_tensor_idx_dims.d[3],
           &input_tensor_idx_dims.d[4], &input_tensor_idx_dims.d[5], &input_tensor_idx_dims.d[6], &input_tensor_idx_dims.d[7]);

    sscanf(input_tensor_dims_p, "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld",
           &input_tensor_new_dims.d[0], &input_tensor_new_dims.d[1], &input_tensor_new_dims.d[2], &input_tensor_new_dims.d[3],
           &input_tensor_new_dims.d[4], &input_tensor_new_dims.d[5], &input_tensor_new_dims.d[6], &input_tensor_new_dims.d[7]);

    chk = forward.create((uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if(chk != 0)
        return -1;

    chk = forward.load_onnx(src);
    if(chk != 0)
    {
        abcdk_trace_printf(LOG_INFO,TT("加载模型文件(%s)失败，无权限或不存在，或不支持的版本。"),src);
        return -1;
    }
        
    initLibNvInferPlugins((void*)forward.logger_ctx(),"");

    forward.get_input_dims(input_tensor_old_dims, input_tensor_idx_dims, 0);

    input_tensor_new_dims.nbDims = input_tensor_old_dims.nbDims;
    for (int i = 0; i < input_tensor_new_dims.nbDims; i++)
    {
        if (input_tensor_new_dims.d[i] < 0)
            input_tensor_new_dims.d[i] = input_tensor_old_dims.d[i];
    }

    forward.set_input_dims(input_tensor_new_dims, input_tensor_idx_dims, 0);

    forward.enable_dla(dla_core);

    if(abcdk_strcmp(data_type_p,"int8",0)==0)
        forward.set_flag((int)nvinfer1::BuilderFlag::kINT8);
    else if(abcdk_strcmp(data_type_p,"fp16",0)==0)
        forward.set_flag((int)nvinfer1::BuilderFlag::kFP16);
    else if(abcdk_strcmp(data_type_p,"tf32",0)==0)
        forward.set_flag((int)nvinfer1::BuilderFlag::kTF32);

    chk = forward.build();
    if(chk != 0)
    {
        abcdk_trace_printf(LOG_INFO,TT("构建加速模型失败，内存不足或其它。"));
        return -1;
    }

    chk = forward.save(dst);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_INFO, TT("保存加速模型文件(%s)失败，无权限或空间不足。"), dst);
        return -1;
    }

    return 0;
}

#else // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

int abcdk_torch_dnn_model_forward_cuda(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含CUDA或TensorRT工具。"));
    return -1;
}

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

__END_DECLS