/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NDARRAY_HXX
#define ABCDK_TORCH_NDARRAY_HXX

#include "util.hxx"

namespace abcdk
{
    namespace torch
    {
        /**多维数组。*/
        class ndarray
        {
        private:
            void *m_data;
            bool m_packed;
            size_t m_block;
            size_t m_depth;
            size_t m_height;
            size_t m_width;
            size_t m_stride;

        public:
            ndarray(void *data, bool packed, size_t b, size_t c, size_t h, size_t w, size_t ws)
            {
                m_data = data;
                m_packed = packed;
                m_block = b;
                m_depth = c;
                m_height = h;
                m_width = w;
                m_stride = ws;
            }

            virtual ~ndarray()
            {
            }

        public:
            template <typename T>
            T *ptr(size_t n, size_t x, size_t y, size_t z)
            {
                return util::ptr<T>(m_data, util::off<T>(m_packed, m_width, m_stride, m_height, m_depth, n, x, y, z));
            }

            template <typename T>
            T obj(size_t n, size_t x, size_t y, size_t z)
            {
                return util::obj<T>(m_data, util::off<T>(m_packed, m_width, m_stride, m_height, m_depth, n, x, y, z));
            }
        };

    } // namespace torch

} // namespace abcdk

#endif // ABCDK_TORCH_NDARRAY_HXX