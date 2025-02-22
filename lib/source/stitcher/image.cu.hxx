/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_IMAGE_HXX
#define ABCDK_CUDA_IMAGE_HXX

#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

namespace abcdk
{
    namespace cuda
    {
        class image
        {
        private:
            AVFrame *m_ctx;

        public:
            image()
            {
                m_ctx = NULL;
            }

            image(int width, int height, enum AVPixelFormat pixfmt, int align = 1)
                : image()
            {
                assert(create(width, height, pixfmt, align) == 0);
            }

            image(const image &src)
                : image()
            {
                *this = src;
            }

            virtual ~image()
            {
                destory();
            }

        public:
            operator AVFrame *() const
            {
                return m_ctx;
            }

            uint8_t *data(int planar = 0)
            {
                if (!m_ctx)
                    return NULL;

                return m_ctx->data[planar];
            }

            int step(int planar = 0)
            {
                if (!m_ctx)
                    return -1;

                return m_ctx->linesize[planar];
            }

            int width()
            {
                if (!m_ctx)
                    return -1;

                return m_ctx->width;
            }

            int height()
            {
                if (!m_ctx)
                    return -1;

                return m_ctx->height;
            }

            enum AVPixelFormat pixfmt()
            {
                if (!m_ctx)
                    return AV_PIX_FMT_NONE;

                return (enum AVPixelFormat)m_ctx->format;
            }

            image &operator=(const image &src)
            {
                assert(src.m_ctx != NULL);

                create(src.width(), src.height(), src.pixfmt());
                abcdk_cuda_avframe_copy(m_ctx, src);

                return *this;
            }

            void destory()
            {
                av_frame_free(&m_ctx);
            }

            int create(int width, int height, enum AVPixelFormat pixfmt, int align = 1)
            {
                int buf_size;

                assert(width > 0 && height > 0 && pixfmt > AV_PIX_FMT_NONE);

                if (m_ctx == NULL || m_ctx->width != width || m_ctx->height != height || m_ctx->format != (int)pixfmt || align > 1)
                {
                    destory();

                    m_ctx = abcdk_cuda_avframe_alloc(width, height, pixfmt, align);
                    if (!m_ctx)
                        return -1;
                }
                else
                {
                    buf_size = abcdk_avimage_size(m_ctx->linesize, m_ctx->height, (enum AVPixelFormat)m_ctx->format);
                    abcdk_cuda_memset(m_ctx->data[0], buf_size);
                }

                return 0;
            }
#ifdef OPENCV_CORE_HPP
            int copyto(cv::Mat &dst) const
            {
                AVFrame tmp_dst = {0};

                if (pixfmt() == AV_PIX_FMT_GRAY8)
                    dst.create(height(), width(), CV_8UC1);
                else if (pixfmt() == AV_PIX_FMT_RGB24)
                    dst.create(height(), width(), CV_8UC3);
                else if (pixfmt() == AV_PIX_FMT_RGB32)
                    dst.create(height(), width(), CV_8UC4);
                else if (pixfmt() == AV_PIX_FMT_GRAYF32)
                    dst.create(height(), width(), CV_32FC1);
                else
                    dst.release();

                if (dst.empty())
                    return -1;

                tmp_dst.data[0] = dst.data;
                tmp_dst.linesize[0] = dst.step;

                chk = abcdk_cuda_avimage_copy(tmp_dst.data, tmp_dst.linesize, 1, m_ctx->data, m_ctx->linesize, 0, width(), height(), pixfmt());
                if (chk != 0)
                    return -1;

                return 0;
            }

            int copyform(const cv::Mat &src)
            {
                AVFrame tmp_src = {0};
                int chk;

                if (src.type() == CV_8UC1)
                    chk = create(src.cols, src.rows, AV_PIX_FMT_GRAY8);
                else if (src.type() == CV_8UC3)
                    chk = create(src.cols, src.rows, AV_PIX_FMT_RGB24);
                else if (src.type() == CV_8UC4)
                    chk = create(src.cols, src.rows, AV_PIX_FMT_RGB32);
                else if (src.type() == CV_32FC1)
                    chk = create(src.cols, src.rows, AV_PIX_FMT_GRAYF32);

                if (chk != 0)
                    return -1;

                tmp_src.data[0] = src.data;
                tmp_src.linesize[0] = src.step;

                chk = abcdk_cuda_avimage_copy(m_ctx->data, m_ctx->linesize, 0, tmp_src.data, tmp_src.linesize, 1, width(), height(), pixfmt());
                if (chk != 0)
                    return -1;

                return 0;
            }
#endif // OPENCV_CORE_HPP
        };
    } //    namespace cuda
} // namespace abcdk

#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_IMAGE_HXX