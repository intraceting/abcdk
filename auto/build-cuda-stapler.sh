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

#
COMPILER_PREFIX=${1}

#
GCC_VERSION=$(${SHELLDIR}/../tools/get-compiler-version.sh ${COMPILER_PREFIX}gcc)
checkReturnCode

#
GLIBC_VERSION=$(${SHELLDIR}/../tools/get-compiler-glibc-version.sh ${COMPILER_PREFIX}gcc)
checkReturnCode

#
${SHELLDIR}/../configure.sh \
    -d INSTALL_PREFIX="/usr/local/stapler/" \
    -d PACKAGE_SUFFIX="-cuda-stapler-gcc_${GCC_VERSION}-glibc_${GLIBC_VERSION}" \
    -d THIRDPARTY_PACKAGES="openssl,ffmpeg,opencv,live555,qrencode,curl,sqlite,cuda" \
    -d THIRDPARTY_FIND_ROOT="/usr/local/stapler/" \
    -d CUDA_FIND_ROOT="/usr/local/cuda/" \
    -d COMPILER_PREFIX=${COMPILER_PREFIX}
checkReturnCode

#
make -s -C ${SHELLDIR}/../ clean

#
make -s -j4 -C ${SHELLDIR}/../
checkReturnCode

#
#make -s -C ${SHELLDIR}/../ install 
#checkReturnCode

#
make -s -C ${SHELLDIR}/../ pack
checkReturnCode

#
make -s -C ${SHELLDIR}/../ clean