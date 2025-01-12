#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
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
CheckSystemName()
# $1 System Name
{
    ${SHELLDIR}/script/devel/check-os-id.sh "$1"
}

#
GetSystemVersion()
{
    ${SHELLDIR}/script/devel/get-os-ver.sh
}

#
CheckPackageKitName()
{
	${SHELLDIR}/script/devel/get-kit-name.sh
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    ${SHELLDIR}/script/devel/check-package.sh "$1"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	${SHELLDIR}/script/devel/check-which.sh "$1"
}

#
CheckHavePackage()
# $1 PKG_NAME
# $2 FLAG
{
    ${SHELLDIR}/3rdparty/$1.sh "$2"
}

#
CheckKeyword()
# $1 keywords
# $2 word
{
    ${SHELLDIR}/script/devel/check-keyword.sh "$1" "$2"
}

#
CheckSTD()
# $1 COMPILER
# $2 STD
{
    ${SHELLDIR}/script/devel/check-c-std.sh "$1" "$2"
}

#
CheckCompiler()
# $1 CC
# $2 AR
# $3 OUTPUT 
{
    ${SHELLDIR}/script/devel/compiler-select.sh "-e" "TARGET_COMPILER_PREFIX=$1" "-e" "TARGET_COMPILER_NAME=$2" "-o" "$3"
}

#
DependPackageCheck()
# 1 key
# 2 def
{
    PACKAGE_KEY=$1
    PACKAGE_DEF=$2
    #
    if [ $(CheckKeyword ${THIRDPARTY_PACKAGES} ${PACKAGE_KEY}) -eq 1 ];then
    {
        CheckHavePackage ${PACKAGE_KEY} 3
        CHK=$?

        if [ ${CHK} -eq 0 ];then
        {
            DEPEND_FLAGS="-D${PACKAGE_DEF} $(CheckHavePackage ${PACKAGE_KEY} 2) ${DEPEND_FLAGS}"
            DEPEND_LINKS="$(CheckHavePackage ${PACKAGE_KEY} 3) ${DEPEND_LINKS}"
        }
        else
        {
            THIRDPARTY_NOFOUND="$(CheckHavePackage ${PACKAGE_KEY} 4) ${THIRDPARTY_NOFOUND}"
        }
        fi

        echo -n "Check ${PACKAGE_KEY}"
        if [ ${CHK} -eq 0 ];then
            echo -e "\x1b[32m Ok \x1b[0m"
        else 
            echo -e "\x1b[31m Failed \x1b[0m"
        fi
    }
    fi

#    echo ${DEPEND_FLAGS} 
#    echo ${DEPEND_LINKS}
}

#LSB发行版。
LSB_RELEASE="linux-gnu"

#
COMPILER_PREFIX=/usr/bin/
COMPILER_NAME=gcc
COMPILER_STD=c99

#
BUILD_TYPE="release"
#
OPTIMIZE_LEVEL=""

#
BUILD_PATH="${SHELLDIR}/build/"
BUILD_PACKAGE_PATH="${SHELLDIR}/package/"
#
INSTALL_PREFIX="/usr/local/"

#
KIT_NAME=""

#
THIRDPARTY_PREFIX=""
THIRDPARTY_PACKAGES="openmp,openssl,archive,libmagic,nghttp2,lz4,ffmpeg"
THIRDPARTY_NOFOUND=""
#
DEPEND_FLAGS=""
DEPEND_LINKS=""


#主版本
VERSION_MAJOR="2"
#副版本
VERSION_MINOR="0"
#发行版本
VERSION_RELEASE="1"

#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]

    -h 
     打印帮助信息。


    -S < release >
     LSB发行版。默认：${LSB_RELEASE}

     支持以下关键字：
     linux-gnu,android

    -o
     优化级别，默认：${OPTIMIZE_LEVEL}。

    -g  
     生成调试符号。

    -f 
     附加的编译参数。

    -l 
     附加的链接参数。

    -d < key,key,... > 
     依赖项目，以英文“,”为分割符。默认：${THIRDPARTY_PACKAGES}
     
     支持以下关键字：
     openmp,unixodbc,sqlite,openssl,ffmpeg,
     freeimage,fuse,libnm,lz4,zlib,
     archive,modbus,libusb,mqtt,redis,json-c,
     bluez,blkid,libcap,fastcgi,systemd,
     libudev,dmtx,qrencode,zbar,magickwand,
     kafka,uuid,libmagic,nghttp2,libdrm,
     pam,curl,ncurses,fltk

    -e < name=value >
     自定义环境变量。
     
     COMPILER_PREFIX=${COMPILER_PREFIX}
     COMPILER_NAME=${COMPILER_NAME}
     THIRDPARTY_PREFIX=${THIRDPARTY_PREFIX}

    -i < path > 
     安装路径。默认：${INSTALL_PREFIX}

    -b < path >
     构建目录。默认：${BUILD_PATH}

    -B < path >
     发行目录。默认：${BUILD_PACKAGE_PATH}

EOF
}

#
while getopts "hS:o:gf:l:d:e:i:b:B:" ARGKEY 
do
    case $ARGKEY in
    h)
        PrintUsage
        exit 0
    ;;
    S)
        LSB_RELEASE="${OPTARG}"
    ;;
    O)
        BUILD_OPTIMIZE="yes"
    ;;
    o)
        OPTIMIZE_LEVEL="$OPTARG"
    ;;
    g)
        BUILD_TYPE="debug"
    ;;
    f)
        DEPEND_FLAGS="$OPTARG"
    ;;
    l)
        DEPEND_LINKS="$OPTARG"

    ;;
    d)
        THIRDPARTY_PACKAGES="${OPTARG}"
    ;;
    e)
        # 使用正则表达式检查参数是否为 "key=value" 或 "key=" 的格式.
        if [[ "$OPTARG" =~ ^[a-zA-Z_][a-zA-Z0-9_]*=.*$ ]]; then
            eval ${OPTARG}
        else 
            echo "'-e ${OPTARG}' will be ignored, the parameter of '- e' only supports the format of 'key=value' or 'key=' ."
        fi 
    ;;
    i)
        INSTALL_PREFIX="${OPTARG}"
    ;;
    b)
        BUILD_PATH="${OPTARG}"
    ;;
    B)
        BUILD_PACKAGE_PATH="${OPTARG}"
    ;;
    esac
done


#
mkdir -p ${BUILD_PATH}
if [ ! -d ${BUILD_PATH} ];then
{
    echo "'${BUILD_PATH}' not found."
    exit 22
}
fi

#
mkdir -p ${BUILD_PACKAGE_PATH}
if [ ! -d ${BUILD_PACKAGE_PATH} ];then
{
    echo "'${BUILD_PACKAGE_PATH}' not found."
    exit 22
}
fi

#去掉末尾的‘/’。
INSTALL_PREFIX_TMP="${INSTALL_PREFIX%/}"
#删除‘/’前面的所有字符，包括‘/’自身。
LAST_NAME="${INSTALL_PREFIX_TMP##*/}"

#如果路径最深层的目录名称不是项目名称则拼接项目名称。
if [ ! "${LAST_NAME}" == "abcdk" ];then
INSTALL_PREFIX="${INSTALL_PREFIX}/abcdk"
fi

#获组件包名称。
KIT_NAME=$(CheckPackageKitName)

#
if [ "${KIT_NAME}" == "rpm" ];then
{
    #
    CheckHavePackage rpmbuild 1
    if [ $? -ne 0 ];then
        echo "'$(CheckHavePackage rpmbuild 4)' not found."
        exit 22
    fi
}
elif [ "${KIT_NAME}" == "deb" ];then
{
    #
    CheckHavePackage dpkg 1
    if [ $? -ne 0 ];then
        echo "'$(CheckHavePackage dpkg 4)' not found."
        exit 22
    fi
}
fi

#
CheckHavePackage pkgconfig 1
if [ $? -ne 0 ];then
    echo "'$(CheckHavePackage pkgconfig 4)' not found."
    exit 22
fi

#检查编译器。
CheckCompiler "${COMPILER_PREFIX}" "${COMPILER_NAME}" "${BUILD_PATH}/compiler.conf"
if [ $? -ne 0 ];then
{
    echo "'${COMPILER_PREFIX}${COMPILER_NAME}' not found."
    exit 22
}
fi

#加载编译器环境。
source ${BUILD_PATH}/compiler.conf

#
CheckSTD "${_TARGET_COMPILER_BIN}" "${COMPILER_STD}"
if [ $? -ne 0 ];then
{
    echo "The '${COMPILER_STD}' standard is not supported."
    exit 22
}
fi


#设置环境变量，用于搜索依赖包。
export _THIRDPARTY_PREFIX=${THIRDPARTY_PREFIX}
export _THIRDPARTY_MACHINE=${_TARGET_MACHINE}
export _THIRDPARTY_BITWIDE=${_TARGET_BITWIDE}

if [ "${_NATIVE_MACHINE}" != "${_TARGET_MACHINE}" ];then
{
    export _THIRDPARTY_PKG_CONFIG_PREFIX=${THIRDPARTY_PREFIX}
    export _THIRDPARTY_PKG_CONFIG_LIBDIR=${THIRDPARTY_PREFIX}/lib64/pkgconfig:${THIRDPARTY_PREFIX}/lib/pkgconfig:${THIRDPARTY_PREFIX}/share/pkgconfig
}
else 
{
    export _THIRDPARTY_PKG_CONFIG_PATH=${THIRDPARTY_PREFIX}/lib64/pkgconfig:${THIRDPARTY_PREFIX}/lib/pkgconfig:${THIRDPARTY_PREFIX}/share/pkgconfig
}
fi

#
DependPackageCheck openmp HAVE_OPENMP
DependPackageCheck unixodbc HAVE_UNIXODBC
DependPackageCheck sqlite HAVE_SQLITE
DependPackageCheck openssl HAVE_OPENSSL
DependPackageCheck ffmpeg HAVE_FFMPEG
DependPackageCheck freeimage HAVE_FREEIMAGE
DependPackageCheck fuse HAVE_FUSE
DependPackageCheck libnm HAVE_LIBNM
DependPackageCheck lz4 HAVE_LZ4
DependPackageCheck zlib HAVE_ZLIB
DependPackageCheck archive HAVE_ARCHIVE
DependPackageCheck modbus HAVE_MODBUS
DependPackageCheck libusb HAVE_LIBUSB
DependPackageCheck mqtt HAVE_MQTT
DependPackageCheck redis HAVE_REDIS
DependPackageCheck json-c HAVE_JSON_C
DependPackageCheck bluez HAVE_BLUEZ
DependPackageCheck blkid HAVE_BLKID
DependPackageCheck libcap HAVE_LIBCAP
DependPackageCheck fastcgi HAVE_FASTCGI
DependPackageCheck samba HAVE_SAMBA
DependPackageCheck systemd HAVE_SYSTEMD
DependPackageCheck libudev HAVE_LIBUDEV
DependPackageCheck dmtx HAVE_LIBDMTX
DependPackageCheck qrencode HAVE_QRENCODE
DependPackageCheck zbar HAVE_ZBAR
DependPackageCheck magickwand HAVE_MAGICKWAND
DependPackageCheck kafka HAVE_KAFKA
DependPackageCheck uuid HAVE_UUID
DependPackageCheck libmagic HAVE_LIBMAGIC
DependPackageCheck nghttp2 HAVE_NGHTTP2
DependPackageCheck libdrm HAVE_LIBDRM
DependPackageCheck pam HAVE_PAM
DependPackageCheck curl HAVE_CURL
DependPackageCheck ncurses HAVE_NCURSES
DependPackageCheck fltk HAVE_FLTK
DependPackageCheck gtk HAVE_GTK
DependPackageCheck x264 HAVE_H264
DependPackageCheck x265 HAVE_H265

#恢复默认。
export _THIRDPARTY_PREFIX=""
export _THIRDPARTY_MACHINE=""
export _THIRDPARTY_BITWIDE=""
export _THIRDPARTY_PKG_CONFIG_PREFIX=""
export _THIRDPARTY_PKG_CONFIG_LIBDIR=""
export _THIRDPARTY_PKG_CONFIG_PATH=""

#
if [ "${THIRDPARTY_NOFOUND}" != "" ];then
{
    echo -e "\x1b[33m${THIRDPARTY_NOFOUND}\x1b[31m not found. \x1b[0m"
    exit 22
}
fi


#
MAKE_CONF=${BUILD_PATH}/makefile.conf

#
PKG_PC=${BUILD_PATH}/pkg_conf.pc

#
RPM_RT_SPEC=${BUILD_PATH}/rpm_rt.spec
RPM_DEV_SPEC=${BUILD_PATH}/rpm_devel.spec

#
DEB_RT_CTL=${BUILD_PATH}/deb_rt.ctl
DEB_DEV_CTL=${BUILD_PATH}/deb_devel.ctl

#
cat >${MAKE_CONF} <<EOF
#
KIT_NAME = ${KIT_NAME}
#
BUILD_PATH = ${BUILD_PATH}
BUILD_PACKAGE_PATH = ${BUILD_PACKAGE_PATH}
#
LSB_RELEASE = ${LSB_RELEASE}
#
STD = ${COMPILER_STD}
#
CC = ${_TARGET_COMPILER_BIN}
AR = ${_TARGET_COMPILER_AR}
#
NATIVE_PLATFORM = ${_NATIVE_PLATFORM}
NATIVE_ARCH = ${_NATIVE_ARCH}
TARGET_PLATFORM = ${_TARGET_PLATFORM}
TARGET_ARCH = ${_TARGET_ARCH}
#
VERSION_MAJOR = ${VERSION_MAJOR}
VERSION_MINOR = ${VERSION_MINOR}
VERSION_RELEASE = ${VERSION_RELEASE}
VERSION_STR = ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
#
DEPEND_FLAGS = ${DEPEND_FLAGS}
DEPEND_LINKS = ${DEPEND_LINKS}
#
BUILD_TYPE = ${BUILD_TYPE}
#
OPTIMIZE_LEVEL = ${OPTIMIZE_LEVEL}
#
INSTALL_PREFIX = ${INSTALL_PREFIX}
#
ROOT_PATH ?= /

EOF
checkReturnCode

#
cat >${PKG_PC} <<EOF
prefix=${INSTALL_PREFIX}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: ABCDK
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Description: ABCDK library
Requires:
Libs: -L\${libdir} -labcdk
Cflags: -I\${includedir}
EOF
checkReturnCode


#
if [ "${KIT_NAME}" == "rpm" ];then
{

#
cat >>${MAKE_CONF} <<EOF
#
PKG_PC = ${PKG_PC}
#
RPM_RT_SPEC = ${RPM_RT_SPEC}
RPM_DEV_SPEC = ${RPM_DEV_SPEC}
EOF
checkReturnCode

#
cat >${RPM_RT_SPEC} <<EOF
Name: abcdk
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Summary: A Better C language Development Kit.
URL: https://github.com/intraceting/abcdk
Group: Applications/System
License: MIT
AutoReqProv: yes

%description
ABCDK is a solution created to support the rapid development of software projects in Linux/Unix systems, 
providing development interfaces for networks, databases, linked lists, multi-trees, hard disks, tapes, 
files, directories, multimedia, etc.
.
This package contains the development files(documents,scripts,libraries).


%files
${INSTALL_PREFIX}

%post
#!/bin/sh
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/abcdk.sh
echo "export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PREFIX}/lib" >> /etc/profile.d/abcdk.sh
exit 0

%postun
#!/bin/sh
rm -f /etc/profile.d/abcdk.sh
exit 0
EOF
checkReturnCode

#
cat >${RPM_DEV_SPEC} <<EOF
Name: abcdk-devel
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Summary: A Better C language Development Kit.
URL: https://github.com/intraceting/abcdk
Group: Applications/System
License: MIT
Requires: abcdk = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_RELEASE}
AutoReqProv: yes

%description
ABCDK is a solution created to support the rapid development of software projects in Linux/Unix systems, 
providing development interfaces for networks, databases, linked lists, multi-trees, hard disks, tapes, 
files, directories, multimedia, etc.
.
This package contains the development files(headers, static libraries).

%files
${INSTALL_PREFIX}

%post
#!/bin/sh
echo "export PKG_CONFIG_PATH=\\\$PKG_CONFIG_PATH:${INSTALL_PREFIX}/pkgconfig" >/etc/profile.d/abcdk-devel.sh
exit 0

%postun
#!/bin/sh
rm -f /etc/profile.d/abcdk-devel.sh
exit 0
EOF
checkReturnCode

}
elif [ "${KIT_NAME}" == "deb" ];then
{

#
mkdir -p ${DEB_RT_CTL}
mkdir -p ${DEB_DEV_CTL}

#
rm -rf ${DEB_RT_CTL}/*
rm -rf ${DEB_DEV_CTL}/*

#
cat >>${MAKE_CONF} <<EOF
#
PKG_PC = ${PKG_PC}
#
DEB_RT_CTL = ${DEB_RT_CTL}
DEB_DEV_CTL = ${DEB_DEV_CTL}
#
DEB_TOOL_ROOT = ${SHELLDIR}/script/devel/
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/control <<EOF
Source: abcdk
Package: abcdk
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Section: Applications/System
Priority: optional
Architecture: ${_TARGET_ARCH}
Maintainer: https://github.com/intraceting/abcdk
Pre-Depends: \${shlibs:Depends}
Description: ABCDK is a solution created to support the rapid development of software projects in Linux/Unix systems, 
 providing development interfaces for networks, databases, linked lists, multi-trees, hard disks, tapes, 
 files, directories, multimedia, etc.
 .
 This package contains the runtime files(documents,scripts,libraries).
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/postinst <<EOF
#!/bin/sh
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/abcdk.sh
echo "export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PREFIX}/lib" >> /etc/profile.d/abcdk.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/postrm <<EOF
#!/bin/sh
rm -f /etc/profile.d/abcdk.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/control <<EOF
Source: abcdk
Package: abcdk-devel
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Section: Applications/System
Priority: optional
Architecture: ${_TARGET_ARCH}
Maintainer: https://github.com/intraceting/abcdk
Pre-Depends: abcdk (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE})
Description: ABCDK is a solution created to support the rapid development of software projects in Linux/Unix systems, 
 providing development interfaces for networks, databases, linked lists, multi-trees, hard disks, tapes, 
 files, directories, multimedia, etc.
 .
 This package contains the development files(headers, static libraries).
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/postinst <<EOF
#!/bin/sh
echo "export PKG_CONFIG_PATH=\\\$PKG_CONFIG_PATH:${INSTALL_PREFIX}/pkgconfig" >/etc/profile.d/abcdk-devel.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/postrm <<EOF
#!/bin/sh
rm -f /etc/profile.d/abcdk-devel.sh
exit 0
EOF
checkReturnCode

#
chmod 755 ${DEB_RT_CTL}/postinst
chmod 755 ${DEB_RT_CTL}/postrm
chmod 755 ${DEB_DEV_CTL}/postinst
chmod 755 ${DEB_DEV_CTL}/postrm

}
fi
