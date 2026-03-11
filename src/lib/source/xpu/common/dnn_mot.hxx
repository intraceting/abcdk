/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_DNN_MOT_HXX
#define ABCDK_XPU_COMMON_DNN_MOT_HXX

#include "abcdk/xpu/dnn_track.h"
#include "../base.in.h"

namespace abcdk_xpu
{
    namespace common
    {
        namespace dnn
        {
            class mot
            {
            private:
                std::string m_name;

            public:
                mot(const char *name = "")
                {
                    m_name = (name ? name : "");
                }

                virtual ~mot()
                {
                }

            public:
                const char *name()
                {
                    return m_name.c_str();
                }

            public:

                virtual void prepare(abcdk_option_t *opt)
                {

                }

                virtual void update(std::vector<abcdk_xpu_dnn_object_t> &dst)
                {
                    update(dst.size(),dst.data());
                }

                virtual void update(int count, abcdk_xpu_dnn_object_t object[])
                {
                    for (int i = 0; i < count; i++)
                        object[i].track_id = -1;
                }
            };

        } // namespace dnn
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_DNN_MOT_HXX