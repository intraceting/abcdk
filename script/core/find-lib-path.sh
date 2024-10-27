#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
    exit 22
fi

SONAME="$1"

#
if [ "${FIND_KIT_TARGET_PREFIX}" == "" ];then
FIND_KIT_TARGET_PREFIX="/usr/"
fi

#
if [ "${FIND_KIT_TARGET_PLATFORM}" == "" ];then
FIND_KIT_TARGET_PLATFORM="$(uname -m)-linux-gnu"
fi 

#
if [ "${FIND_KIT_TARGET_BITWIDE}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        FIND_KIT_TARGET_BITWIDE="64"
    else 
        FIND_KIT_TARGET_BITWIDE="32"
    fi 
}
fi

#
if [ -f ${FIND_KIT_TARGET_PREFIX}/lib${FIND_KIT_TARGET_BITWIDE}/${FIND_KIT_TARGET_PLATFORM}/${SONAME} ];then
    echo "${FIND_KIT_TARGET_PREFIX}/lib${FIND_KIT_TARGET_BITWIDE}/${FIND_KIT_TARGET_PLATFORM}/"
elif [ -f ${FIND_KIT_TARGET_PREFIX}/lib${FIND_KIT_TARGET_BITWIDE}/${SONAME} ];then
    echo "${FIND_KIT_TARGET_PREFIX}/lib${FIND_KIT_TARGET_BITWIDE}/"
elif [ -f ${FIND_KIT_TARGET_PREFIX}/lib/${FIND_KIT_TARGET_PLATFORM}/${SONAME} ];then
    echo "${FIND_KIT_TARGET_PREFIX}/lib/${FIND_KIT_TARGET_PLATFORM}/"
elif [ -f ${FIND_KIT_TARGET_PREFIX}/lib/${SONAME} ];then
    echo "${FIND_KIT_TARGET_PREFIX}/lib/"
else 
    exit 1
fi

#
exit $?