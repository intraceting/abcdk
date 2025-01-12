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
if [ -f /etc/os-release ];then 
    grep '^VERSION_ID=' /etc/os-release |cut -d = -f 2 |sed 's/\"//g'
elif [ -f /usr/lib/os-release ];then 
    grep '^VERSION_ID=' /usr/lib/os-release |cut -d = -f 2 |sed 's/\"//g'
elif [ -f /etc/redhat-release ];then
    head -n 1 /etc/redhat-release |cut -d ' ' -f 3
elif [ -f /etc/debian_version ];then
    head -n 1 cat /etc/issue |cut -d ' ' -f 2
else 
    echo "0"
fi 