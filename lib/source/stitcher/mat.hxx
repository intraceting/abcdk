/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_STITCHER_MAT_HXX
#define ABCDK_STITCHER_MAT_HXX

#include "abcdk/cuda/ndarray.h"
#include "../generic/imageproc.hxx"
#include "../generic/string.hxx"

#ifdef HAVE_OPENCV
#include "opencv2/opencv.hpp"
#ifdef OPENCV_ENABLE_NONFREE
#include "opencv2/xfeatures2d.hpp"
#endif // OPENCV_ENABLE_NONFREE
#endif // HAVE_OPENCV

#ifdef OPENCV_CORE_HPP

namespace abcdk
{
    namespace stitcher
    {
        class mat
        {
            enum memory_kind
            {
                HOST = 0,
                CUDA = 1
            };

        public:
            cv::Mat m_host_ctx;
#ifdef __cuda_cuda_h__
            abcdk_ndarray_t *m_cuda_ctx;
#endif //__cuda_cuda_h__

        public:
            mat()
            {
#ifdef __cuda_cuda_h__
                m_cuda_ctx = NULL;
#endif //__cuda_cuda_h__
            }

            virtual ~mat()
            {
                release();
            }

        public:
            int cols(memory_kind dst_kind = memory_kind::HOST)
            {
                if (dst_kind == memory_kind::HOST)
                    return m_host_ctx.cols;
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA)
                    return (m_cuda_ctx ? m_cuda_ctx->width : -1);
#endif //__cuda_cuda_h__
                else
                    return -1;

                return 0;
            }

            int rows(memory_kind dst_kind = memory_kind::HOST)
            {
                if (dst_kind == memory_kind::HOST)
                    return m_host_ctx.rows;
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA)
                    return (m_cuda_ctx ? m_cuda_ctx->height : -1);
#endif //__cuda_cuda_h__
                else
                    return -1;

                return 0;
            }

            int type(memory_kind dst_kind = memory_kind::HOST)
            {
                if (dst_kind == memory_kind::HOST)
                    return m_host_ctx.type();
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA)
                    return (m_cuda_ctx ? CV_8UC(m_cuda_ctx->depth) : -1);
#endif //__cuda_cuda_h__
                else
                    return -1;

                return -2;
            }

            bool empty(memory_kind dst_kind = memory_kind::HOST)
            {
                if (dst_kind == memory_kind::HOST)
                    return m_host_ctx.empty();
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA)
                    return (m_cuda_ctx ? false : true);
#endif //__cuda_cuda_h__
                else
                    return true;

                return true;
            }

        public:
            virtual void release()
            {
                m_host_ctx.release();
#ifdef __cuda_cuda_h__
                abcdk_ndarray_free(&m_cuda_ctx);
#endif //__cuda_cuda_h__
            }

            mat &operator=(const mat &src)
            {
                release();

                m_host_ctx = src.m_host_ctx;

#ifdef __cuda_cuda_h__
                m_cuda_ctx = abcdk_cuda_ndarray_clone(0, src.m_cuda_ctx);
#endif //__cuda_cuda_h__

                return *this;
            }

            int create(int h, int w, int type, memory_kind dst_kind = memory_kind::HOST)
            {
                assert(h > 0 && w > 0);
                assert(type == CV_8UC1 || type == CV_8UC3 || type == CV_8UC4);

                if (dst_kind == memory_kind::HOST)
                {
                    m_host_ctx.create(h, w, type);
                    if (m_host_ctx.empty())
                        return -1;
                }
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA)
                {
                    if (type == CV_8UC1)
                        m_cuda_ctx = abcdk_cuda_ndarray_alloc(ABCDK_NDARRAY_NCHW, 1, w, h, 1, sizeof(uint8_t), 1);
                    else if (type == CV_8UC3)
                        m_cuda_ctx = abcdk_cuda_ndarray_alloc(ABCDK_NDARRAY_NCHW, 1, w, h, 3, sizeof(uint8_t), 1);
                    else if (type == CV_8UC4)
                        m_cuda_ctx = abcdk_cuda_ndarray_alloc(ABCDK_NDARRAY_NCHW, 1, w, h, 4, sizeof(uint8_t), 1);

                    if (!m_cuda_ctx)
                        return -1;
                }
#endif //__cuda_cuda_h__
                else
                {
                    return -1;
                }

                return 0;
            }

            int clone(memory_kind dst_kind, cv::Mat &src, memory_kind src_kind = memory_kind::HOST)
            {
                assert(!src.empty());
                assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);

                release();

                if (dst_kind == memory_kind::HOST && src_kind == memory_kind::HOST)
                {
                    m_host_ctx = src.clone();
                }
#ifdef __cuda_cuda_h__
                else if (dst_kind == memory_kind::CUDA && src_kind == memory_kind::HOST)
                {
                    if (src.type() == CV_8UC1)
                        m_cuda_ctx = abcdk_cuda_ndarray_clone2(0, src.data, src.step, 1, ABCDK_NDARRAY_NCHW, 1, src.cols, src.rows, 1, sizeof(uint8_t));
                    else if (src.type() == CV_8UC3)
                        m_cuda_ctx = abcdk_cuda_ndarray_clone2(0, src.data, src.step, 1, ABCDK_NDARRAY_NCHW, 1, src.cols, src.rows, 3, sizeof(uint8_t));
                    else if (src.type() == CV_8UC4)
                        m_cuda_ctx = abcdk_cuda_ndarray_clone2(0, src.data, src.step, 1, ABCDK_NDARRAY_NCHW, 1, src.cols, src.rows, 4, sizeof(uint8_t));

                    if (!m_cuda_ctx)
                        return -1;
                }
#endif //__cuda_cuda_h__
                else
                {
                    return -1;
                }

                return 0;
            }
        };
    } // namespace stitcher
} // namespace abcdk

#endif // OPENCV_CORE_HPP

#endif // ABCDK_STITCHER_MAT_HXX