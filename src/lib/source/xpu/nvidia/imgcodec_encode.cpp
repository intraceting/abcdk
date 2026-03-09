/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "imgcodec.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgcodec
        {
            static inline abcdk_object_t *_encode_cpu(const image::metadata_t *src, const char *ext)
            {
                image::metadata_t *tmp_src;
                std::vector<uint8_t> tmp_dst;
                abcdk_object_t *dst;
                int chk;

                if (src->format != AV_PIX_FMT_BGR24)
                {
                    tmp_src = image::create(src->width, src->height, ABCDK_XPU_PIXFMT_BGR24, 16, 0);
                    if (!tmp_src)
                        return NULL;

                    chk = imgproc::convert(src, tmp_src);
                    if (chk != 0)
                    {
                        image::free(&tmp_src);
                        return NULL;
                    }

                    dst = _encode_cpu(tmp_src, ext);
                    image::free(&tmp_src);

                    return dst;
                }

                tmp_src = image::clone(src, 0, 1, 1);
                if (!tmp_src)
                    return NULL;

                chk = common::imgcodec::encode(tmp_src, tmp_dst, ext);
                if (chk != 0)
                {
                    image::free(&tmp_src);
                    return NULL;
                }

                dst = abcdk_object_copyfrom(tmp_dst.data(), tmp_dst.size());
                image::free(&tmp_src);

                return dst;
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

            abcdk_object_t *encode(const image::metadata_t *src, const char *ext)
            {
                abcdk_object_t *dst;
                image::metadata_t *src_tmp;
                jenc::metadata_t *ctx;
                int chk;

                _current_init();

                ctx = (jenc::metadata_t *)pthread_getspecific(_current_key);
                if (!ctx)
                {
                    ctx = jenc::alloc();
                    pthread_setspecific(_current_key, ctx);
                }

                if (!ext || abcdk_strcmp(".jpg", ext, 0) == 0 || abcdk_strcmp(".jpeg", ext, 0) == 0)
                {
                    return jenc::encode(ctx, src);
                }
                else
                {
                    return _encode_cpu(src, ext);
                }
            }

        } // namespace imgcodec
    } // namespace nvidia

} // namespace abcdk_xpu
