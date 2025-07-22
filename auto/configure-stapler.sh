#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname "$0"`; pwd)

# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}

#加载配置环境。
source ${1}

#   -d PACKAGE_SUFFIX="-stapler-gcc_${STAPLER_TARGET_COMPILER_VERSION}-glibc_${STAPLER_TARGET_GLIBC_MAX_VERSION}" \

#
${SHELLDIR}/../configure.sh \
    -d INSTALL_PREFIX="${STAPLER_RELEASE_PATH}" \
    -d PACKAGE_SUFFIX="-stapler-glibc_${STAPLER_TARGET_GLIBC_MAX_VERSION}" \
    -d THIRDPARTY_PACKAGES="openssl,ffmpeg,opencv,live555,qrencode,curl,sqlite,nghttp2,eigen" \
    -d COMPILER_LD_FLAGS="-Wl,-rpath=/usr/local/stapler/lib/ -Wl,-rpath=/usr/local/stapler/lib/compat/ -Wl,-rpath=/usr/local/stapler/lib64/  -Wl,-rpath=/usr/local/stapler/lib64/compat/" \
    -d COMPILER_PREFIX=${STAPLER_TARGET_COMPILER_PREFIX}
checkReturnCode
