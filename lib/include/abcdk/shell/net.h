/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SHELL_NET_H
#define ABCDK_SHELL_NET_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"
#include "abcdk/util/string.h"
#include "abcdk/util/socket.h"
#include "abcdk/shell/proc.h"

__BEGIN_DECLS

/**
 * 获取网卡线路连接状态。
 * 
 * @return 1 已连接，0 未加接，-1 未知的。
*/
int abcdk_net_get_link_state(const char *ifname);

/**
 * 获取网卡操作状态。
 * 
 * @return 1 上线，0 下线，< 0 未知的。
*/
int abcdk_net_get_oper_state(const char *ifname);

/**
 * 停用网卡。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_down(const char *ifname);

/**
 * 启用网卡。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_up(const char *ifname);

/**
 * 清理网卡地址配置。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_address_flush(const char *ifname);

/**
 * 清理网卡路由配置。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_route_flush(const char *ifname);

/**
 * 添加网卡路由配置。
 * 
 * @param [in] ver IPV4或IPV6。
 * @param [in] host 主机地址。
 * @param [in] prefix 子网掩码长度。
 * @param [in] gw 网关地址。
 * @param [in] metric 路由跃点。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_route_add(int ver, const char *host, int prefix, const char *gw, int metric, const char *ifname);

/**
 * 添加网卡地址配置。
 * 
 * @param [in] gw 网关地址。NULL(0) 忽略。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_address_add(int ver, const char *host, int prefix, const char *gw, int metric, const char *ifname);

/**
 * 设置网卡最大传输单元。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_set_mtu(uint16_t mtu,const char *ifname);

/**
 * 设置网卡队列长度。
 * 
 * @return 0 成功，-1 系统错误，-2 权限不足。
*/
int abcdk_net_set_txqueuelen(uint16_t len,const char *ifname);

__END_DECLS

#endif //ABCDK_SHELL_NET_H