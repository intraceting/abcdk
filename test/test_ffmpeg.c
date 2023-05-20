/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_ffmpeg(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG
    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");
    int max_frames = abcdk_option_get_int(args,"--max-frames",0,1000);

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open_capture(NULL,src,NULL,0);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open_writer(NULL,dst,NULL,NULL);

    int ov = -1;

    AVFrame *inframe = av_frame_alloc();
    for(int i = 0;i<max_frames;i++)
    {
        int chk = abcdk_ffmpeg_read2(r,inframe,-1);
        if(chk <0)
            break;

        printf("fmt=%d\n",inframe->format);
        if(inframe->format==8)
            continue;

        if(ov<0)
        {
            ov = abcdk_ffmpeg_add_stream(w,25,inframe->width,inframe->height,AV_CODEC_ID_H264,NULL,0,0);
            abcdk_ffmpeg_write_header(w,0);
        }


         abcdk_ffmpeg_write3(w,inframe,ov);
        
    }

    av_frame_free(&inframe);


    abcdk_ffmpeg_write_trailer(w);
    abcdk_ffmpeg_destroy(&w);
    abcdk_ffmpeg_destroy(&r);
#endif //HAVE_FFMPEG
}