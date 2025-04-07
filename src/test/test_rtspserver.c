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


int abcdk_test_rtspserver(abcdk_option_t *args)
{

    abcdk_rtsp_server_t *ctx = abcdk_rtsp_server_create(12345,NULL);

    int chk = abcdk_rtsp_server_create_media(ctx,"aaa.mp4","haha","haha test");
    assert(chk == 0);

    abcdk_rtsp_runloop(ctx);

    abcdk_rtsp_server_destroy(&ctx);

    return 0;
}