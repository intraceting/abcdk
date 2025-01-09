#!/bin/bash
#
# This file is part of SHORTCUT.
#  
# Copyright (c) 2025 The SHORTCUT project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)


#
if [ "${SHORTCUT_PKG_PREFIX}" == "" ];then
{
    pkg-config $*
}
else 
{
    export PKG_CONFIG_LIBDIR=${SHORTCUT_PKG_PREFIX}/lib/pkgconfig:${SHORTCUT_PKG_PREFIX}/share/pkgconfig
    pkg-config --define-variable=prefix=${SHORTCUT_PKG_PREFIX} $*
}
fi
