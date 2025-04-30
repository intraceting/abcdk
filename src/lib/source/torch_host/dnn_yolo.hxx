/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_YOLO_HXX
#define ABCDK_TORCH_HOST_DNN_YOLO_HXX

#include "dnn_model.hxx"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class yolo : public model
            {
            public:
                yolo()
                {
                }
                virtual ~yolo()
                {
                }

            public:
                virtual void collect_object(std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold);
                {
                }
            };

            class yolo_11 : public yolo
            {
            public:
                yolo_11()
                {
                }
                virtual ~yolo_11()
                {
                }

            public:
                virtual void collect_object(std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold)
                {
                }
            };

        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNN_YOLO_HXX