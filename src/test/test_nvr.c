/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#ifdef HAVE_FFMPEG

int abcdk_test_nvr(abcdk_option_t *args)
{
    abcdk_ffmpeg_nvr_config_t src_cfg = {0};
    abcdk_ffmpeg_nvr_config_t record_cfg = {0};
    abcdk_ffmpeg_nvr_config_t push_cfg = {0};

    src_cfg.u.src.url = abcdk_option_get(args, "--src", 0, "");
    src_cfg.u.src.speed = 1.0;
    src_cfg.u.src.delay_max = 3.0;
    src_cfg.u.src.timeout = 5.0;

    abcdk_ffmpeg_nvr_t *nvr_ctx = abcdk_ffmpeg_nvr_create(&src_cfg);

    record_cfg.flag = 1;
    record_cfg.u.record.prefix = "/tmp/cccc/cccc_";
    record_cfg.u.record.count = 10;
    record_cfg.u.record.duration = 60;

    push_cfg.flag = 2;
    //  push_cfg.u.push.url = "rtmp://192.168.100.96/live/cccc";
    //  push_cfg.u.push.fmt = "rtmp";
    push_cfg.u.push.url = "rtsp://192.168.100.96/live/cccc";
    push_cfg.u.push.fmt = "rtsp";

    uint64_t record_id = abcdk_ffmpeg_nvr_task_add(nvr_ctx, &record_cfg);
    assert(record_id != 0);
    uint64_t push_id = abcdk_ffmpeg_nvr_task_add(nvr_ctx, &push_cfg);
    assert(push_id != 0);

    /*等待终止信号。*/
    //abcdk_proc_wait_exit_signal(-1);
    sleep(60);

    abcdk_ffmpeg_nvr_task_del(nvr_ctx,record_id);
    abcdk_ffmpeg_nvr_task_del(nvr_ctx,push_id);

    abcdk_ffmpeg_nvr_destroy(&nvr_ctx);
}

#else

int abcdk_test_nvr(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_FFMPEG
