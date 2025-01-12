#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ "${_THIRDPARTY_PKG_CONFIG_LIBDIR}" != "" ];then
    export PKG_CONFIG_LIBDIR=${_THIRDPARTY_PKG_CONFIG_LIBDIR}
fi

#
if [ "${_THIRDPARTY_PKG_CONFIG_PATH}" != "" ];then
    export PKG_CONFIG_PATH=${_THIRDPARTY_PKG_CONFIG_PATH}
fi

#
if [ "${_THIRDPARTY_PKG_CONFIG_PREFIX}" == "" ];then
    pkg-config $@
else 
    pkg-config --define-variable=prefix=${_THIRDPARTY_PKG_CONFIG_PREFIX} $@
fi
