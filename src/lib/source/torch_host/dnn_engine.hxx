/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_ENGINE_HXX
#define ABCDK_TORCH_HOST_DNN_ENGINE_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/opencv.h"
#include "../torch/memory.hxx"
#include "dnn_tensor.hxx"


#ifdef OPENCV_DNN_HPP

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class engine
            {
            private:
                std::vector<tensor> m_tensor_ctx;
                cv::dnn::Net m_engine_ctx;
            public:
                engine()
                {

                }

                virtual ~engine()
                {

                }
            public:
                int tensor_size() 
                {
                    return m_tensor_ctx.size();
                }

                tensor *tensor_ptr(int index)
                {
                    return &m_tensor_ctx[index];
                }

            public:
                void release()
                {

                }

                int load(const void *data, size_t size)
                {
                    assert(data != NULL && size > 0);

                    m_engine_ctx = cv::dnn::readNetFromONNX((char*)data,size);
                    if(m_engine_ctx.empty())
                        return -1;

                    m_engine_ctx.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    m_engine_ctx.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

                    return 0;
                }

                int load(const char *file)
                {
                    abcdk_object_t *model_data;
                    int chk;

                    assert(file != NULL);

                    model_data = abcdk_object_copyfrom_file(file);
                    if(!model_data)
                        return -1;

                    chk = load(model_data->pptrs[0],model_data->sizes[0]);
                    abcdk_object_unref(&model_data);
                    
                    return chk;
                }

                int prepare(abcdk_option_t *opt)
                {
                    int k = 0;
                    int chk;

                    //std::vector<std::string> vec_name = m_engine_ctx.getLayerNames();
                    std::vector<std::string> vec_name = m_engine_ctx.getUnconnectedOutLayersNames();

                    std::vector<std::string> vec_input_name;

                    for (int i = 0; i < vec_name.size(); i++)
                    {
                        //std::vector<cv::Ptr<cv::dnn::Layer>> layer2 = m_engine_ctx.getLayerInputs(m_engine_ctx.getLayerId(vec_name[i]));
                        cv::Ptr<cv::dnn::Layer> t = m_engine_ctx.getLayer(m_engine_ctx.getLayerId(vec_name[i]));

                      

                            abcdk_trace_printf(LOG_DEBUG, "name:%s", t->name.c_str());
                            abcdk_trace_printf(LOG_DEBUG, "type:%s", t->type.c_str());

                            auto a = m_engine_ctx.getParam(m_engine_ctx.getLayerId(vec_name[i]), 0);

                            std::cerr << a.size << std::endl;
                        
                    }

                    // std::vector<std::string> input_vec_name = m_engine_ctx.getInputsNames();
                    // std::vector<std::string> output_vec_name = m_engine_ctx.getUnconnectedOutLayersNames();

                    // m_tensor_ctx.resize(input_vec_name.size() + output_vec_name.size());

                    // for (int i = 0; i < input_vec_name.size(); i++)
                    // {
                    //     std::string name = input_vec_name[i];
                    //     cv::dnn::MatShape dims = m_engine_ctx.getInputShape(name);
                    //     cv::Mat blob = m_engine_ctx.getParam(m_engine_ctx.getLayerId(name), 0);

                    //     chk = m_tensor_ctx[k].prepare(i,name.c_str(),tensor::TensorIOMode::kINPUT,(tensor::DataType)blob.type(),dims,opt);
                    //     if(chk != 0)
                    //         return -1;

                    //     k += 1;
                    // }

                    // for (int i = 0; i < output_vec_name.size(); i++)
                    // {
                    //     std::string name = output_vec_name[i];
                    //     cv::dnn::MatShape dims = m_engine_ctx.getUnconnectedOutLayers(name);
                    //     cv::Mat blob = m_engine_ctx.getParam(m_engine_ctx.getLayerId(name), 0);

                    //     chk = m_tensor_ctx[k].prepare(i,name.c_str(),tensor::TensorIOMode::kOUTPUT,(tensor::DataType)blob.type(),dims,opt);
                    //     if(chk != 0)
                    //         return -1;

                    //     k += 1;
                    // }

                    return 0;
                }

                int execute(const std::vector<abcdk_torch_image_t *> &img)
                {
                    return 0;
                }

            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // OPENCV_DNN_HPP

#endif // ABCDK_TORCH_HOST_DNN_ENGINE_HXX