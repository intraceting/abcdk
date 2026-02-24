/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgproc
        {
            static int _remap(const image::metadata_t *src, image::metadata_t *dst,
                              const image::metadata_t *xmap, const image::metadata_t *ymap,
                              abcdk_xpu_inter_t inter_mode)
            {

                NppiSize tmp_dst_size = {0}, tmp_src_size = {0};
                NppiRect tmp_src_roi = {0};
                NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

                tmp_dst_size.width = dst->width;
                tmp_dst_size.height = dst->height;

                tmp_src_size.width = src->width;
                tmp_src_size.height = src->height;

                tmp_src_roi.x = 0;
                tmp_src_roi.y = 0;
                tmp_src_roi.width = src->width;
                tmp_src_roi.height = src->height;

                if (src->format == AV_PIX_FMT_GRAY8)
                {
                    npp_chk = nppiRemap_8u_C1R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                               (Npp32f*)xmap->data[0], xmap->linesize[0], (Npp32f*)ymap->data[0], ymap->linesize[0],
                                               dst->data[0], dst->linesize[0], tmp_dst_size,
                                               inter_local_to_nppi(inter_mode));
                }
                else if (src->format == AV_PIX_FMT_RGB24 || src->format == AV_PIX_FMT_BGR24)
                {
                    npp_chk = nppiRemap_8u_C3R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                               (Npp32f*)xmap->data[0], xmap->linesize[0], (Npp32f*)ymap->data[0], ymap->linesize[0],
                                               dst->data[0], dst->linesize[0], tmp_dst_size,
                                               inter_local_to_nppi(inter_mode));
                }
                else if (src->format == AV_PIX_FMT_RGB32 || src->format == AV_PIX_FMT_BGR32)
                {
                    npp_chk = nppiRemap_8u_C4R(src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                               (Npp32f*)xmap->data[0], xmap->linesize[0], (Npp32f*)ymap->data[0], ymap->linesize[0],
                                               dst->data[0], dst->linesize[0], tmp_dst_size,
                                               inter_local_to_nppi(inter_mode));
                }
                else if (src->format == AV_PIX_FMT_GRAYF32)
                {
                    npp_chk = nppiRemap_32f_C1R((Npp32f *)src->data[0], tmp_src_size, src->linesize[0], tmp_src_roi,
                                                (Npp32f *)xmap->data[0], xmap->linesize[0], (Npp32f *)ymap->data[0], ymap->linesize[0],
                                                (Npp32f *)dst->data[0], dst->linesize[0], tmp_dst_size,
                                                inter_local_to_nppi(inter_mode));
                }

                if (npp_chk != NPP_SUCCESS)
                    return -1;

                return 0;
            }

            int remap(const image::metadata_t *src, image::metadata_t *dst,
                      const image::metadata_t *xmap, const image::metadata_t *ymap,
                      abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _remap(src, dst, xmap, ymap, inter_mode);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
