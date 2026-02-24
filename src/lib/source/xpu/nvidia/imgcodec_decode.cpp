/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgcodec.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgcodec
        {
            static image::metadata_t *_decode_cpu(const void *src, size_t size)
            {
                int chk;

                cv::Mat tmp_dst;
                chk = common::imgcodec::decode(src, size, tmp_dst);
                if (chk != 0)
                    return NULL;

                return image::clone(ABCDK_XPU_PIXFMT_BGR24, tmp_dst, 1, 16, 0);
            }

            static pthread_once_t _current_key_status = PTHREAD_ONCE_INIT;
            static pthread_key_t _current_key = 0xFFFFFFFF;

            static void _current_key_destroy_cb(void *opaque)
            {
                jenc::metadata_t *ctx = (jenc::metadata_t *)opaque;

                jenc::free(&ctx);
            }

            static void _current_key_create_cb()
            {
                pthread_key_create(&_current_key, _current_key_destroy_cb);
            }

            static void _current_init()
            {
                int chk;

                /*only once.*/
                chk = pthread_once(&_current_key_status, _current_key_create_cb);
                assert(chk == 0);
            }

            image::metadata_t *decode(const void *src, size_t size)
            {
                image::metadata_t *dst;
                jdec::metadata_t *ctx;
                int chk;

                _current_init();

                ctx = (jdec::metadata_t *)pthread_getspecific(_current_key);
                if (!ctx)
                {
                    ctx = jdec::alloc();
                    pthread_setspecific(_current_key, ctx);
                }

                if (ABCDK_PTR2U8(src, 0) == 0xFF && ABCDK_PTR2U8(src, 1) == 0xD8)
                {
                    return jdec::decode(ctx,src,size);
                }
                else
                {
                    return _decode_cpu(src, size);
                }
            }
        } // namespace imgcodec
    } // namespace nvidia

} // namespace abcdk_xpu

