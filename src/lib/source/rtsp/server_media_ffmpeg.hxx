/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_MEDIA_FFMPEG_HXX
#define ABCDK_RTSP_SERVER_MEDIA_FFMPEG_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/ffmpeg/ffeditor.h"
#include "server_media.hxx"


#ifdef _SERVER_MEDIA_SESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class media_ffmpeg : public media
        {
        private:
            abcdk_ffeditor_t *m_ff_ctx;

        public:
            static media_ffmpeg *createNew(UsageEnvironment &env, char const *filename)
            {
                return new media_ffmpeg(env, name, info, desc);
            }

            static void deleteOld(media_ffmpeg **ctx)
            {
                media_ffmpeg *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                ctx_p->deleteAllSubsessions();
                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }
        
        protected:
        media_ffmpeg(UsageEnvironment &env, char const *filename)
                : media(env, filename, info, desc)
            {
            }

            virtual ~media_ffmpeg()
            {
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_SERVER_MEDIA_SESSION_HH

#endif // ABCDK_RTSP_SERVER_MEDIA_HXX