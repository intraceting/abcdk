#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
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
PrintUsage()
{
cat << EOF
usage: [ OPTIONS [ VARIABLE ... ] ]

OPTIONS:

    -h 
     打印帮助信息。

    -d < KEY=VALUE | KEY= >
     定义变量及变量赋值。

VARIABLE:

    VENDOR_NAME(供应商的名称).
        
    PACK_NAME(组件包名称).
    
    VERSION_MAJOR(主版本).

    VERSION_MINOR(副版本).

    VERSION_RELEASE(发行版本).

    TARGET_PLATFORM(目标平台).

    FILES_NAME(文件列表的文件名).

    POST_NAME(安装后运行的脚本文件名).

    POSTUN_NAME(卸载后运行的脚本文件名).

    OUTPUT(SPEC文件名)用于存放SPEC文件.

    REQUIRE_LIST(依赖列表).
EOF
}

#
while getopts "hd:" ARGKEY 
do
    case $ARGKEY in
    h)
        PrintUsage
        exit 0
    ;;
    d)
        # 使用正则表达式检查参数是否为 "key=value" 或 "key=" 的格式.
        if [[ ${OPTARG} =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
            declare "${OPTARG%%=*}"="${OPTARG#*=}"
        else 
            echo "'-d ${OPTARG}' will be ignored, the parameter of '- d' only supports the format of 'key=value' or 'key=' ."
        fi 
    ;;
    esac
done


#
if [ "${VENDOR_NAME}" == "" ];then
    echo "VENDOR_NAME=${VENDOR_NAME}, invalid."
    exit 22
fi

#
if [ "${PACK_NAME}" == "" ];then
    echo "PACK_NAME=${PACK_NAME}, invalid."
    exit 22
fi

#
if [ "${VERSION_MAJOR}" == "" ];then
    echo "VERSION_MAJOR=${VERSION_MAJOR}, invalid."
    exit 22
fi

#
if [ "${VERSION_MINOR}" == "" ];then
    echo "VERSION_MINOR=${VERSION_MINOR}, invalid."
    exit 22
fi

#
if [ "${VERSION_RELEASE}" == "" ];then
    echo "VERSION_RELEASE=${VERSION_RELEASE}, invalid."
    exit 22
fi

#
if [ "${TARGET_PLATFORM}" == "" ];then
    echo "TARGET_PLATFORM=${TARGET_PLATFORM}, invalid."
    exit 22
fi

#
if [ "${FILES_NAME}" == "" ];then
    echo "FILES_NAME=${FILES_NAME}, invalid."
    exit 22
fi

#
if [ "${POST_NAME}" == "" ];then
    echo "POST_NAME=${POST_NAME}, invalid."
    exit 22
fi

#
if [ "${POSTUN_NAME}" == "" ];then
    echo "POSTUN_NAME=${POSTUN_NAME}, invalid."
    exit 22
fi

#
if [ "${OUTPUT}" == "" ];then
    echo "OUTPUT=${OUTPUT}, invalid."
    exit 22
fi

#提取父级路径。
FATHER_PATH=$(dirname ${OUTPUT})
#创建不存在的父级路径。
mkdir -p "${FATHER_PATH}"

#
cat >${OUTPUT} <<EOF
Name: ${PACK_NAME}
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Summary: ${PACK_NAME}.
Vendor: ${VENDOR_NAME}
Group: Applications/System
Exclusivearch : ${TARGET_PLATFORM}
License: none
Requires: ${REQUIRE_LIST} 
AutoReqProv: no

# disable '.build-id soft-link'.
%global debug_package %{nil}
%define _build_id_links none

# disable 'debug-info'.
%define _enable_debug_package 0

# use gzip for old RPM.
%define _binary_payload w9.gzdio

%description
This is the ${PACK_NAME} component package.
.
This package contains the runtime files(headers, static libraries).

EOF
checkReturnCode


#
echo "%files" >> ${OUTPUT}
#
cat ${FILES_NAME} >> ${OUTPUT}
#空行
echo "" >> ${OUTPUT}


echo "%post" >> ${OUTPUT}
echo "#!/bin/sh" >> ${OUTPUT}
#
cat ${POST_NAME} >> ${OUTPUT}
#
echo "exit 0" >> ${OUTPUT}
#空行
echo "" >> ${OUTPUT}

echo "%postun" >> ${OUTPUT}
echo "#!/bin/sh" >> ${OUTPUT}
#
cat ${POSTUN_NAME} >> ${OUTPUT}
#
echo "exit 0" >> ${OUTPUT}
#空行
echo "" >> ${OUTPUT}