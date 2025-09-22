#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname "$0"`; pwd)

#
SHELLKITS_HOME_CHECK_LISTS[0]="${SHELLKITS_HOME}"
SHELLKITS_HOME_CHECK_LISTS[1]="${SHELLDIR}/../SHellKits"
SHELLKITS_HOME_CHECK_LISTS[2]="${SHELLDIR}/../../SHellKits"
SHELLKITS_HOME_CHECK_LISTS[3]="${SHELLDIR}/../../../SHellKits"
SHELLKITS_HOME_CHECK_LISTS[4]="${SHELLDIR}/../../../../SHellKits"
SHELLKITS_HOME_CHECK_LISTS[5]="${SHELLDIR}/../../../../../SHellKits"

#clear.
SHELLKITS_HOME=""

#
for CHECK_ONE in "${SHELLKITS_HOME_CHECK_LISTS[@]}"; do
{
    if [ "${CHECK_ONE}" != "" ];then
        CHECK_ONE=$(realpath -s "${CHECK_ONE}")
    fi

    if [ -d "${CHECK_ONE}" ];then
    {
        SHELLKITS_HOME="${CHECK_ONE}"
        break
    }
    fi
}
done

#
if [ "${SHELLKITS_HOME}" == "" ] || [ ! -d "${SHELLKITS_HOME}" ];then
{
    echo "The environment variable SHELLKITS_HOME points to an invalid or non-existent path."
    echo "The required toolset can be downloaded from 'https://github.com/intraceting/SHellKits.git'."
    exit 1
}
fi 

#导出SHELLKITS_HOME变量给其它子工具集使用。
export SHELLKITS_HOME


# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}

#
BUILD_PATH=${SHELLDIR}/build/
INSTALL_PREFIX=/usr/local/

#创建不存在的路径。
mkdir -p ${BUILD_PATH}

#
${SHELLKITS_HOME}/solution/aaaaa-configure.sh $@ \
    -d BUILD_PATH=${BUILD_PATH} \
    -d INSTALL_PREFIX=${INSTALL_PREFIX}
exit $?