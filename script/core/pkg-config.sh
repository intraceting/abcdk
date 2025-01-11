#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)


#
if [ "${ABCDK_THIRDPARTY_PREFIX}" == "" ];then
{
    pkg-config $@
}
else 
{
    export PKG_CONFIG_LIBDIR=${ABCDK_THIRDPARTY_PREFIX}/lib/pkgconfig:${ABCDK_THIRDPARTY_PREFIX}/share/pkgconfig
    pkg-config --define-variable=prefix=${ABCDK_THIRDPARTY_PREFIX} $@
}
fi
