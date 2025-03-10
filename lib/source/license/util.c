/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/license/util.h"

void abcdk_license_dump(const abcdk_license_info_t *info)
{
    char msg[1024] = {0};
    struct tm begin_tm = {0}, end_tm = {0};

    assert(info != NULL);

    /*转换起止时间。*/
    abcdk_time_sec2tm(&begin_tm, info->begin, 0);
    abcdk_time_sec2tm(&end_tm, info->begin + info->duration * 24 * 3600ULL, 0);

    sprintf(msg + strlen(msg), TT("产品类别(%hhu)；"), info->category);
    sprintf(msg + strlen(msg), TT("产品型号(%hhu)；"), info->product);
    sprintf(msg + strlen(msg), TT("节点数量(%hu)；"), info->node);
    sprintf(msg + strlen(msg), TT("生效日期(%04d年%02d月%02d日)；"), begin_tm.tm_year + 1900, begin_tm.tm_mon + 1, begin_tm.tm_mday);
    sprintf(msg + strlen(msg), TT("终止日期(%04d年%02d月%02d日)；"), end_tm.tm_year + 1900, end_tm.tm_mon + 1, end_tm.tm_mday);

    abcdk_trace_printf(LOG_INFO, TT("授权摘要：%s\n"), msg);

    return;
}

int64_t abcdk_license_status(const abcdk_license_info_t *info, uint64_t realtime, int dump_if_expire)
{
    int64_t remain_sec = 0;

    assert(info != NULL);

    remain_sec = abcdk_clock_remainder(info->begin, info->duration * 24 * 3600ULL, realtime);

    if (remain_sec <= 0 && dump_if_expire)
        abcdk_license_dump(info);

    return remain_sec;
}

