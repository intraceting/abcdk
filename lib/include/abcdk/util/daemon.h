/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_DAEMON_H
#define ABCDK_UTIL_DAEMON_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 子进程入口函数。
 * 
 * @return 出错码。0~126之间有效。
 */
typedef int (*abcdk_daemon_entry_cb)(void *opaque);

/**
 * 子进程监视函数。
 *
 * @param [in] wstatus 子进程状态码。
 *
 * @return >= 0 重启间隔(秒)，< 0 结束。
 */
typedef int (*abcdk_daemon_monitor_cb)(int wstatus, void *opaque);

/**
 * 启动守护进程。
 * 
 * @param [in] background 运行模式。0 前台，!0 后台。
 * @param [in] entry_cb 子进程入口函数。
 * @param [in] monitor_cb 子进程监视函数。
 * @param [in] opaque 环境指针。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_daemon(int background,abcdk_daemon_entry_cb entry_cb,abcdk_daemon_monitor_cb monitor_cb,void *opaque);

__END_DECLS

#endif //ABCDK_UTIL_DAEMON_H