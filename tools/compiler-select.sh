#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}


#
native_compiler_agent()
{
    if [ -f "${NATIVE_COMPILER_BIN}" ];then
        ${NATIVE_COMPILER_BIN} "$@" 2>>/dev/null
    else 
        return 127
    fi
}

#
target_compiler_agent()
{
    if [ -f "${TARGET_COMPILER_BIN}" ];then
        ${TARGET_COMPILER_BIN} "$@" 2>>/dev/null
    else 
        return 127
    fi
}

#
NATIVE_COMPILER_PREFIX=/usr/bin/
NATIVE_COMPILER_NAME=gcc
NATIVE_COMPILER_BIN=
NATIVE_COMPILER_FORTRAN=
NATIVE_COMPILER_SYSROOT=
NATIVE_COMPILER_AR=
NATIVE_COMPILER_LD=
NATIVE_COMPILER_RANLIB=
NATIVE_COMPILER_READELF=

#
TARGET_COMPILER_PREFIX=/usr/bin/
TARGET_COMPILER_NAME=gcc
TARGET_COMPILER_BIN=
TARGET_COMPILER_SYSROOT=
TARGET_COMPILER_AR=
TARGET_COMPILER_LD=
TARGET_COMPILER_RANLIB=
TARGET_COMPILER_READELF=

#
VAR_PREFIX=GNU

#
OUTPUT_FILE=


#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]
    -h
    打印此文档。

    -d < name=value >
     自定义环境变量。
     
     NATIVE_COMPILER_PREFIX=${NATIVE_COMPILER_PREFIX}
     NATIVE_COMPILER_NAME=${NATIVE_COMPILER_NAME}
     NATIVE_COMPILER_BIN=\${NATIVE_COMPILER_PREFIX}\${NATIVE_COMPILER_NAME}
     NATIVE_COMPILER_SYSROOT=\$(\${NATIVE_COMPILER_BIN} "--print-sysroot")
     NATIVE_COMPILER_AR=\$(\${NATIVE_COMPILER_BIN} "-print-prog-name=ar")
     NATIVE_COMPILER_LD=\$(\${NATIVE_COMPILER_BIN} "-print-prog-name=ld")
     NATIVE_COMPILER_RANLIB=\$(\${NATIVE_COMPILER_BIN} "-print-prog-name=ranlib")
     NATIVE_COMPILER_READELF=\$(\${NATIVE_COMPILER_BIN} "-print-prog-name=readelf")
     
     TARGET_COMPILER_PREFIX=${TARGET_COMPILER_PREFIX}
     TARGET_COMPILER_NAME=${TARGET_COMPILER_NAME}
     TARGET_COMPILER_BIN=\${TARGET_COMPILER_PREFIX}\${TARGET_COMPILER_NAME}
     TARGET_COMPILER_SYSROOT=\$(\${TARGET_COMPILER_BIN} "--print-sysroot")
     TARGET_COMPILER_AR=\$(\${TARGET_COMPILER_BIN} "-print-prog-name=ar")
     TARGET_COMPILER_LD=\$(\${TARGET_COMPILER_BIN} "-print-prog-name=ld")
     TARGET_COMPILER_RANLIB=\$(\${TARGET_COMPILER_BIN} "-print-prog-name=ranlib")
     TARGET_COMPILER_READELF=\$(\${TARGET_COMPILER_BIN} "-print-prog-name=readelf")

    -p < PREFIX >
     变量前缀。默认：${VAR_PREFIX}

    -o < FILE >
     输出文件。
EOF
}

#
while getopts "hd:p:o:" ARGKEY 
do
    case $ARGKEY in
    \?)
        PrintUsage
        exit 22
    ;;
    h)
        PrintUsage
        exit 0
    ;;
    d)
        # 转换字符串为变量名和变量值。
        # KEY="${STR%%=*}"
        # VAL="${STR#*=}"
        # declare "${KEY}"="${VAL}"

        # 使用正则表达式检查参数是否为 "key=value" 或 "key=" 的格式.
        if [[ ${OPTARG} =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
            declare "${OPTARG%%=*}"="${OPTARG#*=}"
        else 
            echo "'-d ${OPTARG}' will be ignored, the parameter of '- e' only supports the format of 'key=value' or 'key=' ."
        fi 
    ;;
    p)
        VAR_PREFIX=${OPTARG}
    ;;
    o)
        OUTPUT_FILE=${OPTARG}
    ;;
    esac
done

#################################################################################

#修复默认值。
if [ "${NATIVE_COMPILER_BIN}" == "" ];then
NATIVE_COMPILER_BIN=${NATIVE_COMPILER_PREFIX}${NATIVE_COMPILER_NAME}
fi

#检查参数。
if [ ! -f "${NATIVE_COMPILER_BIN}" ];then
echo "'NATIVE_COMPILER_BIN=${NATIVE_COMPILER_BIN}' invalid or non-existent."
exit 22
fi

#修复默认值。
if [ "${NATIVE_COMPILER_SYSROOT}" == "" ];then
NATIVE_COMPILER_SYSROOT=$(native_compiler_agent "--print-sysroot")
fi
#修复默认值。
if [ "${NATIVE_COMPILER_AR}" == "" ];then
NATIVE_COMPILER_AR=$(native_compiler_agent "-print-prog-name=ar")
NATIVE_COMPILER_AR=$(which "${NATIVE_COMPILER_AR}")
fi
#修复默认值。
if [ "${NATIVE_COMPILER_LD}" == "" ];then
NATIVE_COMPILER_LD=$(native_compiler_agent "-print-prog-name=ld")
NATIVE_COMPILER_LD=$(which "${NATIVE_COMPILER_LD}")
fi
#修复默认值。
if [ "${NATIVE_COMPILER_RANLIB}" == "" ];then
NATIVE_COMPILER_RANLIB=$(native_compiler_agent "-print-prog-name=ranlib")
NATIVE_COMPILER_RANLIB=$(which "${NATIVE_COMPILER_RANLIB}")
fi
#修复默认值。
if [ "${NATIVE_COMPILER_READELF}" == "" ];then
NATIVE_COMPILER_READELF=$(native_compiler_agent "-print-prog-name=readelf")
NATIVE_COMPILER_READELF=$(which "${NATIVE_COMPILER_READELF}")
fi


#################################################################################

#修复默认值。
if [ "${TARGET_COMPILER_BIN}" == "" ];then
TARGET_COMPILER_BIN=${TARGET_COMPILER_PREFIX}${TARGET_COMPILER_NAME}
fi

#检查参数。
if [ ! -f "${TARGET_COMPILER_BIN}" ];then
echo "'TARGET_COMPILER_BIN=${TARGET_COMPILER_BIN}' invalid or non-existent."
exit 22
fi


#修复默认值。
if [ "${TARGET_COMPILER_SYSROOT}" == "" ];then
TARGET_COMPILER_SYSROOT=$(target_compiler_agent "--print-sysroot")
fi
#修复默认值。
if [ "${TARGET_COMPILER_AR}" == "" ];then
TARGET_COMPILER_AR=$(target_compiler_agent "-print-prog-name=ar")
TARGET_COMPILER_AR=$(which "${TARGET_COMPILER_AR}")
fi
#修复默认值。
if [ "${TARGET_COMPILER_LD}" == "" ];then
TARGET_COMPILER_LD=$(target_compiler_agent "-print-prog-name=ld")
TARGET_COMPILER_LD=$(which "${TARGET_COMPILER_LD}")
fi
#修复默认值。
if [ "${TARGET_COMPILER_RANLIB}" == "" ];then
TARGET_COMPILER_RANLIB=$(target_compiler_agent "-print-prog-name=ranlib")
TARGET_COMPILER_RANLIB=$(which "${TARGET_COMPILER_RANLIB}")
fi
#修复默认值。
if [ "${TARGET_COMPILER_READELF}" == "" ];then
TARGET_COMPILER_READELF=$(target_compiler_agent "-print-prog-name=readelf")
TARGET_COMPILER_READELF=$(which "${TARGET_COMPILER_READELF}")
fi


#################################################################################

#
NATIVE_MACHINE=$(native_compiler_agent "-dumpmachine")
TARGET_MACHINE=$(target_compiler_agent "-dumpmachine")

#
NATIVE_PLATFORM=$(echo ${NATIVE_MACHINE} | cut -d - -f 1)
TARGET_PLATFORM=$(echo ${TARGET_MACHINE} | cut -d - -f 1)

#转换构建平台架构关键字。
if [ "${NATIVE_PLATFORM}" == "x86_64" ];then
{
    NATIVE_ARCH="amd64"
    NATIVE_BITWIDE="64"
}
elif [ "${NATIVE_PLATFORM}" == "aarch64" ] || [ "${NATIVE_PLATFORM}" == "armv8l" ];then
{
    NATIVE_ARCH="arm64"
    NATIVE_BITWIDE="64"
}
elif [ "${NATIVE_PLATFORM}" == "arm" ] || [ "${NATIVE_PLATFORM}" == "armv7l" ] || [ "${NATIVE_PLATFORM}" == "armv7a" ];then
{
    NATIVE_ARCH="arm"
    NATIVE_BITWIDE="32"
}
fi

#转换构建平台架构关键字。
if [ "${TARGET_PLATFORM}" == "x86_64" ];then
{
    TARGET_ARCH="amd64"
    TARGET_BITWIDE="64"
}
elif [ "${TARGET_PLATFORM}" == "aarch64" ] || [ "${TARGET_PLATFORM}" == "armv8l" ];then
{
    TARGET_ARCH="arm64"
    TARGET_BITWIDE="64"
}
elif [ "${TARGET_PLATFORM}" == "arm" ] || [ "${TARGET_PLATFORM}" == "armv7l" ] || [ "${TARGET_PLATFORM}" == "armv7a" ];then
{
    TARGET_ARCH="arm"
    TARGET_BITWIDE="32"
}
fi

#################################################################################

#
NATIVE_COMPILER_VERSION=$(native_compiler_agent "-dumpversion")
TARGET_COMPILER_VERSION=$(target_compiler_agent "-dumpversion")


#提取本机平台的glibc最大版本。
NATIVE_GLIBC_MAX_VER=$(ldd --version |head -n 1 |rev |cut -d ' ' -f 1 |rev)

#
if [ "${NATIVE_PLATFORM}" == "${TARGET_PLATFORM}" ];then
{
    #提取目标平台的glibc最大版本。
    TARGET_GLIBC_MAX_VER=$(ldd --version |head -n 1 |rev |cut -d ' ' -f 1 |rev)
}
else
{
    #提取目标平台的glibc最大版本。
    if [ -f ${TARGET_COMPILER_SYSROOT}/lib64/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_COMPILER_READELF} -V ${TARGET_COMPILER_SYSROOT}/lib64/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    elif [ -f ${TARGET_COMPILER_SYSROOT}/lib/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_COMPILER_READELF} -V ${TARGET_COMPILER_SYSROOT}/lib/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    elif [ -f ${TARGET_COMPILER_SYSROOT}/${TARGET_PLATFORM}-linux-gun/lib/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_COMPILER_READELF} -V ${TARGET_COMPILER_SYSROOT}/lib/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    else
        TARGET_GLIBC_MAX_VER="0.0"
    fi
}
fi

#
if [ "${NATIVE_GLIBC_MAX_VER}" == "" ];then
echo "Unable to obtain the glibc version of the native platform."
exit 1
fi

#
if [ "${TARGET_GLIBC_MAX_VER}" == "" ];then
echo "Unable to obtain the glibc version for the target platform."
exit 1
fi


#################################################################################

#检查参数。
if [ "${OUTPUT_FILE}" == "" ];then
echo "'OUTPUT_FILE=${OUTPUT_FILE}' invalid or non-existent."
exit 22
fi

#提取路径。
OUTPUT_PATH="$(dirname "${OUTPUT_FILE}")"

#创建不存在的路径。
mkdir -p "${OUTPUT_PATH}"


#################################################################################

cat >${OUTPUT_FILE} <<EOF
#
${VAR_PREFIX}_NATIVE_COMPILER_PREFIX=${NATIVE_COMPILER_PREFIX}
${VAR_PREFIX}_NATIVE_COMPILER_NAME=${NATIVE_COMPILER_NAME}
${VAR_PREFIX}_NATIVE_COMPILER_BIN=${NATIVE_COMPILER_BIN}
${VAR_PREFIX}_NATIVE_COMPILER_SYSROOT=${NATIVE_COMPILER_SYSROOT}
${VAR_PREFIX}_NATIVE_COMPILER_AR=${NATIVE_COMPILER_AR}
${VAR_PREFIX}_NATIVE_COMPILER_LD=${NATIVE_COMPILER_LD}
${VAR_PREFIX}_NATIVE_COMPILER_RANLIB=${NATIVE_COMPILER_RANLIB}
${VAR_PREFIX}_NATIVE_COMPILER_READELF=${NATIVE_COMPILER_READELF}
#
${VAR_PREFIX}_TARGET_COMPILER_PREFIX=${TARGET_COMPILER_PREFIX}
${VAR_PREFIX}_TARGET_COMPILER_NAME=${TARGET_COMPILER_NAME}
${VAR_PREFIX}_TARGET_COMPILER_BIN=${TARGET_COMPILER_BIN}
${VAR_PREFIX}_TARGET_COMPILER_SYSROOT=${TARGET_COMPILER_SYSROOT}
${VAR_PREFIX}_TARGET_COMPILER_AR=${TARGET_COMPILER_AR}
${VAR_PREFIX}_TARGET_COMPILER_LD=${TARGET_COMPILER_LD}
${VAR_PREFIX}_TARGET_COMPILER_RANLIB=${TARGET_COMPILER_RANLIB}
${VAR_PREFIX}_TARGET_COMPILER_READELF=${TARGET_COMPILER_READELF}
#
${VAR_PREFIX}_NATIVE_MACHINE=${NATIVE_MACHINE}
${VAR_PREFIX}_TARGET_MACHINE=${TARGET_MACHINE}
#
${VAR_PREFIX}_NATIVE_PLATFORM=${NATIVE_PLATFORM}
${VAR_PREFIX}_TARGET_PLATFORM=${TARGET_PLATFORM}
#
${VAR_PREFIX}_NATIVE_ARCH=${NATIVE_ARCH}
${VAR_PREFIX}_TARGET_ARCH=${TARGET_ARCH}
#
${VAR_PREFIX}_NATIVE_BITWIDE=${NATIVE_BITWIDE}
${VAR_PREFIX}_TARGET_BITWIDE=${TARGET_BITWIDE}
#
${VAR_PREFIX}_NATIVE_COMPILER_VERSION=${NATIVE_COMPILER_VERSION}
${VAR_PREFIX}_TARGET_COMPILER_VERSION=${TARGET_COMPILER_VERSION}
#
${VAR_PREFIX}_NATIVE_GLIBC_MAX_VER=${NATIVE_GLIBC_MAX_VER}
${VAR_PREFIX}_TARGET_GLIBC_MAX_VER=${TARGET_GLIBC_MAX_VER}
EOF
checkReturnCode