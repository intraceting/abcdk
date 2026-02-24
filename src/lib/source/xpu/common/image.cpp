/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "image.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace image
        {
            int copy(const cv::Mat &src, AVFrame *dst)
            {
                AVFrame *ff_src;
                int chk;

                ff_src = av_frame_alloc();
                if(!ff_src)
                    return -ENOMEM;

                ff_src->width = src.cols;
                ff_src->height = src.rows;
                ff_src->format = dst->format;

                abcdk_ffmpeg_image_fill_stride(ff_src->linesize,ff_src->width,(AVPixelFormat)ff_src->format,1);//only step(1).
                abcdk_ffmpeg_image_fill_pointer(ff_src->data,ff_src->linesize,ff_src->height,(AVPixelFormat)ff_src->format,src.data);
                
                chk = copy(ff_src,dst);
                av_frame_free(&ff_src);
                                
                return chk;
            }
        } // namespace image
    } // namespace common
} // namespace abcdk_xpu
