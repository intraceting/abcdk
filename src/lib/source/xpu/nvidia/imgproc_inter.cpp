/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace imgproc
        {
            NppiInterpolationMode inter_local_to_nppi(abcdk_xpu_inter_t mode)
            {
                if (mode == ABCDK_XPU_INTER_NEAREST)
                    return NPPI_INTER_NN;
                else if (mode == ABCDK_XPU_INTER_LINEAR)
                    return NPPI_INTER_LINEAR;
                else if (mode == ABCDK_XPU_INTER_CUBIC)
                    return NPPI_INTER_CUBIC;
                else
                    return NPPI_INTER_UNDEFINED;
            }
        } // namespace imgproc
    } // namespace nvidia
} // namespace abcdk_xpu

