/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_MOT_BYTETRACK_HXX
#define ABCDK_TORCH_HOST_DNN_MOT_BYTETRACK_HXX

#include "abcdk/torch/dnn.h"
#include "../torch/memory.hxx"
#include "dnn_mot.hxx"
#include "bytetrack/BYTETracker.h"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
#ifdef __BYTETRACK__
                class mot_bytetrack : public mot
                {
                private:
                    bytetrack::BYTETracker *m_ctx;
                    int m_fps;

                public:
                    mot_bytetrack(const char *name = "")
                        : mot(name)
                    {
                        m_ctx = NULL;
                    }

                    virtual ~mot_bytetrack()
                    {
                        abcdk::torch::memory::delete_object((bytetrack::BYTETracker **)&m_ctx);
                    }

                public:
                    virtual void prepare(abcdk_option_t *opt)
                    {
                        m_fps = abcdk_option_get_int(opt, "--frame-rate", 0, 30);

                        m_ctx = new bytetrack::BYTETracker(m_fps,m_fps);
                        assert(m_ctx != NULL);
                    }

                protected:
                    virtual void update(int count, abcdk_torch_dnn_object_t object[])
                    {
                        static uint64_t magic = 1;

                        std::vector<bytetrack::object> tmp_obj;
                        std::vector<bytetrack::STrack> ret_obj;

                        /*临时数组。*/
                        tmp_obj.resize(count);

                        /*复制目标的部分信息到临时数组，用于更新追踪器。*/
                        for (int i = 0; i < count; i++)
                        {
                            tmp_obj[i].x = object[i].rect.pt[0].x;
                            tmp_obj[i].y = object[i].rect.pt[0].y;
                            tmp_obj[i].w = object[i].rect.pt[1].x;
                            tmp_obj[i].h = object[i].rect.pt[1].y;
                            tmp_obj[i].label = object[i].label;
                            tmp_obj[i].score = (float)object[i].score / 100.;
                            tmp_obj[i].magic = abcdk_atomic_fetch_and_add(&magic, 1);
                        }

                        /*更新追踪器并返回追踪结果。*/
                        ret_obj = m_ctx->update(tmp_obj);

                        for (int i = 0; i < count; i++)
                        {
                            /*根据魔法数查找追踪ID。*/
                            auto fetch = [](bytetrack::object &t, std::vector<bytetrack::STrack> &rets)
                            {
                                for (auto &tmp : rets)
                                {
                                    if (tmp.m_magic == t.magic)
                                        return tmp.track_id;
                                }

                                return -1;
                            };

                            /*获取追踪ID。*/
                            object[i].track_id = fetch(tmp_obj[i], ret_obj);
                        }
                    }
                };
#endif // __BYTETRACK__
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNN_MOT_BYTETRACK_HXX