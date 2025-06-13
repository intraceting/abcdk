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
make -s -C ${SHELLDIR}/../ uninstall
checkReturnCode

#
make -s -C ${SHELLDIR}/../ clean

#
make -s -j4 -C ${SHELLDIR}/../
checkReturnCode

#
make -s -C ${SHELLDIR}/../ install 
checkReturnCode

#
make -s -C ${SHELLDIR}/../ pack INSTALL_PREFIX=/usr/local/stapler/ 
checkReturnCode
