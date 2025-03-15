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
    
    INSTALL_PREFIX(安装路经的前缀).
    
    VERSION_MAJOR(主版本).

    VERSION_MINOR(副版本).

    VERSION_RELEASE(发行版本).

    TARGET_ARCH(目标平台).

    OUTPUT(SPEC文件名)用于存放SPEC文件.

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
if [ "${INSTALL_PREFIX}" == "" ];then
    echo "INSTALL_PREFIX=${INSTALL_PREFIX}, invalid."
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
if [ "${TARGET_ARCH}" == "" ];then
    echo "TARGET_ARCH=${TARGET_ARCH}, invalid."
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
Name: abcdk
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Summary: A Better C language Development Kit.
URL: https://github.com/intraceting/abcdk
Group: Applications/System
License: MIT
AutoReqProv: yes

%description
This is the ABCDK component package.
.
This package contains the runtime files(bin,doc).


%files
${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR}
${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
${INSTALL_PREFIX}/bin/abcdk-tool
${INSTALL_PREFIX}/bin/abcdk-script
${INSTALL_PREFIX}/share/abcdk

%post
#!/bin/sh
#
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/abcdk.sh
chmod 0755 /etc/profile.d/abcdk.sh
#
echo "${INSTALL_PREFIX}/lib" > /etc/ld.so.conf.d/abcdk.conf
ldconfig
#
exit 0

%postun
#!/bin/sh
#
rm -f /etc/profile.d/abcdk.sh
#
rm -f /etc/ld.so.conf.d/abcdk.conf
ldconfig
#
exit 0
EOF
checkReturnCode