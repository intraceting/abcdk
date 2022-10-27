/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_RTPS_H
#define ABCDK_UTIL_RTPS_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/** SDP分析。*/
abcdk_tree_t *abcdk_rtsp_sdp_parse(const char *data, size_t size);

/** SDP打印。*/
void abcdk_rtsp_sdp_dump(FILE *fp, abcdk_tree_t *sdp);

/** 
 * 在SDP中查找媒体信息或属性。
 * 
 * @param [in] fmt 媒体格式(载荷)。
 * 
 * @return !NULL(0) 成功(节点指针)，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_rtsp_sdp_find_media(abcdk_tree_t *sdp, uint8_t fmt);

__END_DECLS

#endif //ABCDK_UTIL_RTPS_H