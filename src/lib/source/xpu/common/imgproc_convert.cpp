/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            static pthread_once_t _current_key_status = PTHREAD_ONCE_INIT;
            static pthread_key_t _current_key = 0xFFFFFFFF;

            static void _current_key_destroy_cb(void *opaque)
            {
                abcdk_ffmpeg_sws_t *ctx = (abcdk_ffmpeg_sws_t *)opaque;

                abcdk_ffmpeg_sws_free(&ctx);
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

            int convert(const AVFrame *src, AVFrame *dst)
            {
                abcdk_ffmpeg_sws_t *ctx = NULL;
                int chk;

                _current_init();

                ctx = (abcdk_ffmpeg_sws_t *)pthread_getspecific(_current_key);
                if (!ctx)
                {
                    ctx = abcdk_ffmpeg_sws_alloc();
                    pthread_setspecific(_current_key, ctx);
                }

                if (!ctx)
                    return -ENOMEM;

                chk = abcdk_ffmpeg_sws_scale(ctx, src, dst);
                if (chk < 0)
                    return -1;

                return 0;
            }
        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
