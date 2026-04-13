/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"
#include "util.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgproc
        {
            static int _resize(const image::metadata_t *src, const abcdk_xpu_rect_t *src_roi, image::metadata_t *dst, abcdk_xpu_inter_t inter_mode)
            {
                NppStreamContext npp_ctx = {0};
                NppiSize tmp_dst_size = {0}, tmp_src_size = {0};
                NppiRect tmp_dst_roi = {0}, tmp_src_roi = {0};
                NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

                util::npp_get_context(&npp_ctx, 0);

                tmp_dst_size.width = dst->width;
                tmp_dst_size.height = dst->height;

                tmp_dst_roi.x = 0;
                tmp_dst_roi.y = 0;
                tmp_dst_roi.width = dst->width;
                tmp_dst_roi.height = dst->height;

                tmp_src_size.width = src->width;
                tmp_src_size.height = src->height;

                tmp_src_roi.x = (src_roi ? src_roi->x : 0);
                tmp_src_roi.y = (src_roi ? src_roi->y : 0);
                tmp_src_roi.width = (src_roi ? src_roi->width : src->width);
                tmp_src_roi.height = (src_roi ? src_roi->height : src->height);

                if (src->format == AV_PIX_FMT_GRAY8)
                {
                    npp_chk = nppiResize_8u_C1R_Ctx(src->data[0], src->linesize[0], tmp_src_size, tmp_src_roi,
                                                dst->data[0], dst->linesize[0], tmp_dst_size, tmp_dst_roi,
                                                util::inter_local_to_nppi(inter_mode),npp_ctx);
                }
                else if (src->format == AV_PIX_FMT_RGB24 || src->format == AV_PIX_FMT_BGR24)
                {
                    npp_chk = nppiResize_8u_C3R_Ctx(src->data[0], src->linesize[0], tmp_src_size, tmp_src_roi,
                                                dst->data[0], dst->linesize[0], tmp_dst_size, tmp_dst_roi,
                                                util::inter_local_to_nppi(inter_mode),npp_ctx);
                }
                else if (src->format == AV_PIX_FMT_RGB32 || src->format == AV_PIX_FMT_BGR32)
                {
                    npp_chk = nppiResize_8u_C4R_Ctx(src->data[0], src->linesize[0], tmp_src_size, tmp_src_roi,
                                                dst->data[0], dst->linesize[0], tmp_dst_size, tmp_dst_roi,
                                                util::inter_local_to_nppi(inter_mode),npp_ctx);
                }
                else if (src->format == AV_PIX_FMT_GRAYF32)
                {
                    npp_chk = nppiResize_32f_C1R_Ctx((Npp32f *)src->data[0], src->linesize[0], tmp_src_size, tmp_src_roi,
                                                 (Npp32f *)dst->data[0], dst->linesize[0], tmp_dst_size, tmp_dst_roi,
                                                 util::inter_local_to_nppi(inter_mode),npp_ctx);
                }

                if (npp_chk != NPP_SUCCESS)
                    return -1;

                return 0;
            }

            int resize(const image::metadata_t *src, const abcdk_xpu_rect_t *src_roi, image::metadata_t *dst, abcdk_xpu_inter_t inter_mode)
            {
                assert((src->format == AV_PIX_FMT_GRAY8 && dst->format == AV_PIX_FMT_GRAY8) ||
                       (src->format == AV_PIX_FMT_RGB24 && dst->format == AV_PIX_FMT_RGB24) ||
                       (src->format == AV_PIX_FMT_BGR24 && dst->format == AV_PIX_FMT_BGR24) ||
                       (src->format == AV_PIX_FMT_RGB32 && dst->format == AV_PIX_FMT_RGB32) ||
                       (src->format == AV_PIX_FMT_BGR32 && dst->format == AV_PIX_FMT_BGR32) ||
                       (src->format == AV_PIX_FMT_GRAYF32 && dst->format == AV_PIX_FMT_GRAYF32));

                return _resize(src, src_roi, dst, inter_mode);
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
