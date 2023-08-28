#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 4 ];then
    exit 22
fi

#
PREFIX_PATH="$1"
TARGET_PLATFORM="$2"
TARGET_BITWIDE="$3"
SONAME="$4"

#
if [ "${PREFIX_PATH}" == "" ];then
PREFIX_PATH="/usr/"
fi

#
if [ "${TARGET_PLATFORM}" == "" ];then
TARGET_PLATFORM="$(uname -m)-linux-gnu"
fi 

#
if [ "${TARGET_BITWIDE}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        TARGET_BITWIDE="64"
    else 
        TARGET_BITWIDE="32"
    fi 
}
fi

#
if [ -f ${PREFIX_PATH}/lib${TARGET_BITWIDE}/${TARGET_PLATFORM}/${SONAME} ];then
    echo "${PREFIX_PATH}/lib${TARGET_BITWIDE}/${TARGET_PLATFORM}/"
elif [ -f ${PREFIX_PATH}/lib${TARGET_BITWIDE}/${SONAME} ];then
    echo "${PREFIX_PATH}/lib${TARGET_BITWIDE}/"
elif [ -f ${PREFIX_PATH}/lib/${TARGET_PLATFORM}/${SONAME} ];then
    echo "${PREFIX_PATH}/lib/${TARGET_PLATFORM}/"
elif [ -f ${PREFIX_PATH}/lib/${SONAME} ];then
    echo "${PREFIX_PATH}/lib/"
else 
    exit 1
fi

#
exit $?