#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 2 ];then
    exit 22
fi

#
PREFIX_PATH="$1"
HDNAME="$2"

#
if [ "${PREFIX_PATH}" == "" ];then
PREFIX_PATH="/usr/"
fi

#
if [ -f ${PREFIX_PATH}/include/${HDNAME} ];then
    echo "${PREFIX_PATH}/include/"
elif [ -f ${PREFIX_PATH}/${HDNAME} ];then
    echo "${PREFIX_PATH}/"
else 
    exit 1
fi

#
exit $?