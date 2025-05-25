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
${SHELLDIR}/../configure.sh \
    -d INSTALL_PREFIX="/usr/local/stapler/" \
    -d PACKAGE_SUFFIX="-stapler" \
    -d THIRDPARTY_PACKAGES="openssl,ffmpeg,opencv,live555,qrencode,onnxruntime,curl,sqlite" \
    -d THIRDPARTY_FIND_ROOT="/usr/local/stapler/" \
    -d COMPILER_PREFIX=${1}
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