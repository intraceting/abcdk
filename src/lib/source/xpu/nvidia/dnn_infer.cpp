/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "dnn_infer.hxx"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace dnn
        {
            namespace infer
            {
                typedef struct _metadata
                {
                    engine *engine_ctx;
                } metadata_t;

                void free(metadata_t **ctx)
                {
                    metadata_t *ctx_p;

                    if (!ctx || !*ctx)
                        return;

                    ctx_p = *ctx;
                    *ctx = NULL;

                    common::util::delete_object(&ctx_p->engine_ctx);

                    delete ctx_p;
                }

                metadata_t *alloc()
                {
                    metadata_t *ctx;

                    ctx = new metadata_t;
                    if (!ctx)
                        return NULL;

                    ctx->engine_ctx = new engine;

                    return ctx;
                }

                int load_model(metadata_t *ctx, const char *file, abcdk_option_t *opt)
                {
                    int chk;

                    chk = ctx->engine_ctx->load(file);
                    if (chk != 0)
                        return chk;

                    chk = ctx->engine_ctx->prepare(opt);
                    if (chk != 0)
                        return chk;

                    return 0;
                }

                int fetch_tensor(metadata_t *ctx, int count, abcdk_xpu_dnn_tensor_t tensor[])
                {
                    int chk_count = 0;

                    assert(ctx != NULL && count > 0 && tensor != NULL);

                    for (int i = 0; i < count; i++)
                    {
                        dnn::tensor *src_p = ctx->engine_ctx->tensor_ptr(i);
                        if (!src_p)
                            break;

                        abcdk_xpu_dnn_tensor_t *dst_p = &tensor[i];

                        dst_p->index = src_p->index();
                        dst_p->name_p = src_p->name();
                        dst_p->mode = (src_p->input() ? 1 : 2);

                        dst_p->dims.nb = src_p->dims().nbDims;

                        for (int i = 0; i < dst_p->dims.nb; i++)
                            dst_p->dims.d[i] = (int)src_p->dims().d[i];

                        dst_p->data_p = (src_p->input() ? NULL : src_p->data(1));

                        chk_count += 1;
                    }

                    return chk_count;
                }

                int forward(metadata_t *ctx, int count, image::metadata_t *img[])
                {
                    std::vector<image::metadata_t *> tmp_img;
                    int chk;

                    assert(ctx != NULL && count > 0 && img != NULL);

                    tmp_img.resize(count);
                    for (int i = 0; i < count; i++)
                    {
                        auto &img_p = img[i];
                        if (!img_p)
                            continue;

                        assert(img_p->format == AV_PIX_FMT_RGB24 || img_p->format == AV_PIX_FMT_BGR24);
                        tmp_img[i] = img_p;
                    }

                    chk = ctx->engine_ctx->execute(tmp_img);
                    if (chk != 0)
                        return chk;

                    return 0;
                }

            } // namespace infer
        } // namespace dnn
    } // namespace nvidia
} // namespace abcdk_xpu
