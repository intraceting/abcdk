/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_H
#define ABCDK_H

#include "abcdk/util/scsi.h"
#include "abcdk/util/mediumx.h"
#include "abcdk/util/ndarray.h"
#include "abcdk/util/epoll.h"
#include "abcdk/util/tape.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/crc.h"
#include "abcdk/util/crc32.h"
#include "abcdk/util/cap.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/base64.h"
#include "abcdk/util/uri.h"
#include "abcdk/util/url.h"
#include "abcdk/util/signal.h"
#include "abcdk/util/string.h"
#include "abcdk/util/charset.h"
#include "abcdk/util/reader.h"
#include "abcdk/util/pool.h"
#include "abcdk/util/math.h"
#include "abcdk/util/exec.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/defs.h"
#include "abcdk/util/general.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/getargs.h"
#include "abcdk/util/termios.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/object.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/option.h"
#include "abcdk/util/io.h"
#include "abcdk/util/map.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/iconv.h"
#include "abcdk/util/thread.h"
#include "abcdk/util/geometry.h"
#include "abcdk/util/clock.h"
#include "abcdk/util/lz4.h"
#include "abcdk/util/time.h"
#include "abcdk/util/popen.h"
#include "abcdk/util/shm.h"
#include "abcdk/util/path.h"
#include "abcdk/util/hash.h"
#include "abcdk/util/mmap.h"
#include "abcdk/util/hexdump.h"
#include "abcdk/util/fnmatch.h"
#include "abcdk/util/register.h"
#include "abcdk/util/queue.h"
#include "abcdk/util/waiter.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/md5.h"
#include "abcdk/util/mutex.h"
#include "abcdk/util/receiver.h"
#include "abcdk/util/random.h"
#include "abcdk/util/tar.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/stream.h"
#include "abcdk/util/package.h"
#include "abcdk/util/timer.h"
#include "abcdk/util/cpu.h"
#include "abcdk/util/enigma.h"
#include "abcdk/util/sha256.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/util/rwlock.h"
#include "abcdk/util/worker.h"
#include "abcdk/util/context.h"
#include "abcdk/util/asio.h"
#include "abcdk/util/asioex.h"
#include "abcdk/util/wred.h"
#include "abcdk/util/getpass.h"
#include "abcdk/util/ipool.h"
#include "abcdk/util/registry.h"
#include "abcdk/util/nonce.h"

#include "abcdk/log/logger.h"

#include "abcdk/shell/scsi.h"
#include "abcdk/shell/mtab.h"
#include "abcdk/shell/mmc.h"
#include "abcdk/shell/user.h"
#include "abcdk/shell/file.h"
#include "abcdk/shell/proc.h"
#include "abcdk/shell/block.h"
#include "abcdk/shell/dmi.h"
#include "abcdk/shell/net.h"


#include "abcdk/mp4/demuxer.h"
#include "abcdk/mp4/muxer.h"
#include "abcdk/mp4/file.h"
#include "abcdk/mp4/atom.h"

#include "abcdk/rtp/rtp.h"

#include "abcdk/sdp/sdp.h"

#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/avutil.h"
#include "abcdk/ffmpeg/swscale.h"
#include "abcdk/ffmpeg/avcodec.h"
#include "abcdk/ffmpeg/avformat.h"
#include "abcdk/ffmpeg/ffserver.h"

#include "abcdk/database/redis.h"
#include "abcdk/database/sqlite.h"
/*
 *1: 与freeimage存在宏定义冲突，因此不能同时包含。
 *2: 调用者需要在合适位置引用下面的文件。
#include "abcdk/database/odbc.h"
#include "abcdk/database/odbcpool.h"
*/

#include "abcdk/image/bmp.h"
/*
 *1: 与odbc存在宏定义冲突，因此不能同时包含。
 *2: 调用者需要在合适位置引用下面的文件。
#include "abcdk/image/freeimage.h"
*/

#include "abcdk/audio/aac.h"

#include "abcdk/video/h2645.h"
#include "abcdk/video/h264.h"
#include "abcdk/video/hevc.h"

#include "abcdk/json/json.h"

#include "abcdk/http/util.h"

#include "abcdk/openssl/openssl.h"
#include "abcdk/openssl/cipher.h"
#include "abcdk/openssl/darknet.h"
#include "abcdk/openssl/bio.h"
#include "abcdk/openssl/cipherex.h"

#include "abcdk/curl/curl.h"

#include "abcdk/net/stcp.h"
#include "abcdk/net/sudp.h"
#include "abcdk/net/tipc.h"
#include "abcdk/net/https.h"
#include "abcdk/net/srpc.h"

#include "abcdk/license/license.h"

#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/cuda/imageproc.h"

#endif //ABCDK_H
