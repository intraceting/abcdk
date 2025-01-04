/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_TEST_ENTRY_H
#define ABCDK_TEST_ENTRY_H

#include "abcdk.h"

__BEGIN_DECLS

int abcdk_test_http(abcdk_option_t *args);
int abcdk_test_http2(abcdk_option_t *args);
int abcdk_test_uri(abcdk_option_t *args);
int abcdk_test_log(abcdk_option_t *args);
int abcdk_test_any(abcdk_option_t *args);
int abcdk_test_exec(abcdk_option_t *args);
int abcdk_test_com(abcdk_option_t *args);
int abcdk_test_path(abcdk_option_t *args);
int abcdk_test_ffmpeg(abcdk_option_t *args);
int abcdk_test_drm(abcdk_option_t *args);
int abcdk_test_ping(abcdk_option_t *args);
int abcdk_test_onvif(abcdk_option_t *args);
int abcdk_test_dhcp(abcdk_option_t *args);
int abcdk_test_tipc(abcdk_option_t *args);
int abcdk_test_timer(abcdk_option_t *args);
int abcdk_test_tun(abcdk_option_t *args);
int abcdk_test_srpc(abcdk_option_t *args);
int abcdk_test_fmp4(abcdk_option_t *args);
int abcdk_test_worker(abcdk_option_t *args);
int abcdk_test_sudp(abcdk_option_t *args);
int abcdk_test_pem(abcdk_option_t *args);
int abcdk_test_usb(abcdk_option_t *args);
int abcdk_test_rand(abcdk_option_t *args);
int abcdk_test_runonce(abcdk_option_t *args);
int abcdk_test_ncurses(abcdk_option_t *args);
int abcdk_test_fltk(abcdk_option_t *args);

__END_DECLS

#endif //ABCDK_TEST_ENTRY_H
