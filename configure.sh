#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname "$0"`; pwd)

#
SHELLKITS_HOME_CHECK_LIST+=("${SHELLKITS_HOME}")
SHELLKITS_HOME_CHECK_LIST+=("${SHELLDIR}/../SHellKits")
SHELLKITS_HOME_CHECK_LIST+=("${SHELLDIR}/../../SHellKits")
SHELLKITS_HOME_CHECK_LIST+=("${SHELLDIR}/../../../SHellKits")
SHELLKITS_HOME_CHECK_LIST+=("${SHELLDIR}/../../../../SHellKits")
SHELLKITS_HOME_CHECK_LIST+=("${SHELLDIR}/../../../../../SHellKits")

#clear.
SHELLKITS_HOME=""

#
for CHECK_ONE in "${SHELLKITS_HOME_CHECK_LIST[@]}"; do
{
    if [ "${CHECK_ONE}" != "" ];then
        CHECK_ONE=$(realpath -m "${CHECK_ONE}")
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


#
exit_if_error()
#errno
#errstr
#exitcode
{
    if [ $# -ne 3 ];then
    {
        echo "Requires three parameters: errno, errstr, exitcode."
        exit 1
    }
    fi 
    
    if [ $1 -ne 0 ];then
    {
        echo $2
        exit $3
    }
    fi
}

#
${SHELLKITS_HOME}/fast-c-cxx/configure.sh "$@" -d SOURCE_PATH=${SHELLDIR} -d PRIVATE_CONF_PATH=${SHELLDIR}/configure.d
exit_if_error $? "Configuration failed." $?


