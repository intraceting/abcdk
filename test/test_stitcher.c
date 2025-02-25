/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_stitcher(abcdk_option_t *args)
{

  abcdk_stitcher_t *ctx = abcdk_stitcher_create();

  abcdk_object_t *metadata = abcdk_object_copyfrom_file(abcdk_option_get(args, "--metadata-load", 0, ""));

  if (metadata)
  {
    abcdk_stitcher_metadata_load(ctx, "7b1a5e0796419a8278e1f6df640f0bfb", metadata->pstrs[0]);
    abcdk_object_unref(&metadata);
  }

  metadata = abcdk_stitcher_metadata_dump(ctx, "abcdk");
  if (metadata)
  {
    fprintf(stderr, "%s\n", metadata->pstrs[0]);
    abcdk_object_unref(&metadata);
  }

  abcdk_media_frame_t *img[4] = {0}, *mask[4] = {0};

  img[0] = abcdk_opencv_image_load("/tmp/ccc/you1.jpg",0);
  img[3] = abcdk_opencv_image_load("/tmp/ccc/you2.jpg",0);
  img[1] = abcdk_opencv_image_load("/tmp/ccc/you3.jpg",0);
  img[2] = abcdk_opencv_image_load("/tmp/ccc/you4.jpg",0);

  int chk = abcdk_stitcher_estimate_transform(ctx, 4, img, mask, 0.8);

  abcdk_stitcher_build_panorama_param(ctx);

  abcdk_media_frame_t *out = NULL;
  chk = abcdk_stitcher_compose_panorama(ctx, &out, 4, img);

  //abcdk_media_frame_save("/tmp/ccc/pano.bmp",out);
  abcdk_media_frame_free(&out);

  for (int i = 0; i < 4; i++)
    abcdk_media_frame_free(&img[i]);

  abcdk_stitcher_destroy(&ctx);

  return 0;
}
