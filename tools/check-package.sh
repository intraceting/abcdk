#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
{
    echo "22"
    exit 22
}
fi

#0 已安装，!0 未安装。
STATUS="1"
#
KIT_NAME=$(${SHELLDIR}/get-kit-name.sh)

#
if [ "deb" == "${KIT_NAME}" ];then 
    STATUS=$(dpkg -V ${1} >> /dev/null 2>&1 ; echo $?)
elif [ "rpm" == "${KIT_NAME}" ];then
	STATUS=$(rpm -q ${1} >> /dev/null 2>&1 ; echo $?)
fi

#
exit ${STATUS}
