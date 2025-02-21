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
# $1 LANG
# $2 COMPILER
# $3 STD
{
    ${SHELLDIR}/script/devel/check-$1-std.sh "$2" "$3"
}

#
CheckCompiler()
# $1 CC
# $2 AR
# $3 OUTPUT 
{
    ${SHELLDIR}/script/devel/compiler-select.sh "-d" "TARGET_COMPILER_PREFIX=$1" "-d" "TARGET_COMPILER_NAME=$2" "-o" "$3" "-p" ""
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


#主版本
VERSION_MAJOR="3"
#副版本
VERSION_MINOR="0"
#发行版本
VERSION_RELEASE="1"
#
VERSION_STR_MAIN="${VERSION_MAJOR}.${VERSION_MINOR}"
VERSION_STR_FULL="${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}"

#
LSB_RELEASE="linux-gnu"

#
COMPILER_PREFIX=/usr/bin/
COMPILER_NAME=gcc

#
COMPILER_C_FLAGS=""
COMPILER_CXX_FLAGS=""
COMPILER_LD_FLAGS=""

#
BUILD_TYPE="release"
#
OPTIMIZE_LEVEL=""

#
BUILD_PATH="${SHELLDIR}/build/"
BUILD_PACKAGE_PATH="${SHELLDIR}/release/"
#
INSTALL_PREFIX="/usr/local/"

#
KIT_NAME=""

#
THIRDPARTY_PACKAGES="openssl,ffmpeg"
THIRDPARTY_FIND_ROOT="/usr/local/"
THIRDPARTY_FIND_MODE="both"
THIRDPARTY_NOFOUND=""

#
CUDA_FIND_ROOT=
CUDA_COMPILER_BIN=

#
CUDNN_FIND_ROOT=

#
TRNSORRT_FIND_ROOT=

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

     LSB_RELEASE=${LSB_RELEASE}

     LSB_RELEASE(发行版名称)支持以下关键字:
     linux-gnu,android
     
     OPTIMIZE_LEVEL=${OPTIMIZE_LEVEL}

     OPTIMIZE_LEVEL(优化级别)支持以下关键字：
     1,2,3,s,fast

     BUILD_TYPE=${BUILD_TYPE}
    
     BUILD_TYPE(构建类型)支持以下关键字：
     debug,release
     
     COMPILER_PREFIX=${COMPILER_PREFIX}

     COMPILER_PREFIX(C/C++编译器路径的前缀)与编译器名字组成完整路径.

     COMPILER_NAME=${COMPILER_NAME}

     COMPILER_NAME(C/C++编译器的名字)与编译器前缀组成完整路径.

     COMPILER_C_FLAGS=${COMPILER_C_FLAGS}

     COMPILER_C_FLAGS(C编译器的编译参数)用于编译器的源码编译. 

     COMPILER_CXX_FLAGS=${COMPILER_CXX_FLAGS}

     COMPILER_CXX_FLAGS(C++编译器的编译参数)用于编译器的源码编译. 

     COMPILER_LD_FLAGS=${COMPILER_LD_FLAGS}

     COMPILER_LD_FLAGS(编译器的链接参数)用于编译器的目标链接. 

     THIRDPARTY_PACKAGES=${THIRDPARTY_PACKAGES}

     THIRDPARTY_PACKAGES(依赖组件列表)支持以下关键字:
     openmp,unixodbc,sqlite,openssl,ffmpeg,
     freeimage,fuse,libnm,lz4,zlib,
     archive,modbus,libusb,mqtt,redis,json-c,
     bluez,blkid,libcap,fastcgi,systemd,
     libudev,dmtx,qrencode,zbar,magickwand,
     kafka,uuid,libmagic,nghttp2,libdrm,
     pam,curl,ncurses,fltk,x264,x265,ffnvcodec,
     cuda,cudnn,tensorrt,opencv

     THIRDPARTY_FIND_ROOT=${THIRDPARTY_FIND_ROOT}

     THIRDPARTY_FIND_ROOT(依赖组件搜索根路径)用于查找依赖组件完整路径.

     THIRDPARTY_FIND_MODE=${THIRDPARTY_FIND_MODE}

     THIRDPARTY_FIND_MODE(依赖组件搜索模式)支持以下关键字:
     only,both,(default)

     CUDA_FIND_ROOT=\${THIRDPARTY_FIND_ROOT}/cuda/

     CUDA_FIND_ROOT(CUDA组件搜索根路径)用于查找依赖组件完整路径.

     CUDA_COMPILER_BIN=\${THIRDPARTY_FIND_ROOT}/cuda/bin/nvcc

     CUDA_COMPILER_BIN(CUDA编译器的完整路径).

     CUDNN_FIND_ROOT=\${THIRDPARTY_FIND_ROOT}/cudnn/

     CUDNN_FIND_ROOT(CUDNN组件搜索根路径)用于查找依赖组件完整路径.

     TRNSORRT_FIND_ROOT=\${THIRDPARTY_FIND_ROOT}/TensorRT/

     TRNSORRT_FIND_ROOT(TensorRT组件搜索根路径)用于查找依赖组件完整路径.

     INSTALL_PREFIX=${INSTALL_PREFIX}

     INSTALL_PREFIX(安装路经的前缀).

     BUILD_PATH=${BUILD_PATH}

     BUILD_PATH(过程文件存放的路径)用于存放构建过程文件.

     BUILD_PACKAGE_PATH=${BUILD_PACKAGE_PATH}

     BUILD_PACKAGE_PATH(发行包存放的路径)用于存放发行包.
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

#转换为绝对路径。
INSTALL_PREFIX=$(realpath ${INSTALL_PREFIX})
BUILD_PATH=$(realpath ${BUILD_PATH})
BUILD_PACKAGE_PATH=$(realpath ${BUILD_PACKAGE_PATH})

#
mkdir -p ${BUILD_PATH}
if [ ! -d ${BUILD_PATH} ];then
{
    echo "'BUILD_PATH=${BUILD_PATH}' invalid or unsupported."
    exit 22
}
fi

#
mkdir -p ${BUILD_PACKAGE_PATH}
if [ ! -d ${BUILD_PACKAGE_PATH} ];then
{
    echo "'BUILD_PACKAGE_PATH=${BUILD_PACKAGE_PATH}' invalid or unsupported."
    exit 22
}
fi

#安装路径必须有效，并且不支持安装到根(/)路径。
if [ "${INSTALL_PREFIX}" == "" ] || [ "${INSTALL_PREFIX}" == "/" ];then
{
    echo "'INSTALL_PREFIX=${INSTALL_PREFIX}' invalid or unsupported."
    exit 22
}
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
CheckSTD c "${_TARGET_COMPILER_BIN}" "c99"
if [ $? -ne 0 ];then
{
    echo "The compiler supports at least the c99 standard."
    exit 22
}
fi

#
CheckSTD cxx "${_TARGET_COMPILER_BIN}" "c++11"
if [ $? -ne 0 ];then
{
    echo "The compiler supports at least the c++11 standard."
    exit 22
}
fi


#
if [ "${_NATIVE_PLATFORM}" == "${_TARGET_PLATFORM}" ];then
{
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
}
fi

#
CheckHavePackage pkgconfig 1
if [ $? -ne 0 ];then
    echo "'$(CheckHavePackage pkgconfig 4)' not found."
    exit 22
fi

#
if [ "${CUDA_FIND_ROOT}" == "" ];then
CUDA_FIND_ROOT="${THIRDPARTY_FIND_ROOT}/cuda/"
fi
if [ "${CUDA_COMPILER_BIN}" == "" ];then
CUDA_COMPILER_BIN="${THIRDPARTY_FIND_ROOT}/cuda/bin/nvcc"
fi

#
if [ "${CUDNN_FIND_ROOT}" == "" ];then
CUDNN_FIND_ROOT="${THIRDPARTY_FIND_ROOT}/cudnn/"
fi

#
if [ " ${TRNSORRT_FIND_ROOT}" == "" ];then
TRNSORRT_FIND_ROOT="${THIRDPARTY_FIND_ROOT}/TensorRT/"
fi


#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${_TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${_TARGET_BITWIDE}
export _3RDPARTY_PKG_FIND_ROOT=${THIRDPARTY_FIND_ROOT}
export _3RDPARTY_PKG_FIND_MODE=${THIRDPARTY_FIND_MODE}

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
DependPackageCheck ffnvcodec HAVE_FFNVCODEC
DependPackageCheck opencv HAVE_OPENCV

#恢复默认。
export _3RDPARTY_PKG_MACHINE=
export _3RDPARTY_PKG_WORDBIT=
export _3RDPARTY_PKG_FIND_ROOT=
export _3RDPARTY_PKG_FIND_MODE=


#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${_TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${_TARGET_BITWIDE}
export _3RDPARTY_PKG_FIND_ROOT=${CUDA_FIND_ROOT}
export _3RDPARTY_PKG_FIND_MODE=${THIRDPARTY_FIND_MODE}

if [ $(CheckKeyword ${THIRDPARTY_PACKAGES} cuda) -eq 1 ];then
{
    #查找NVCC。
    if [ "${CUDA_COMPILER_BIN}" == "" ];then
        CUDA_COMPILER_BIN=$(CheckHavePackage cuda 5)
    fi

    #如果NVCC存在，再查找依赖组件。
    if [ -f ${CUDA_COMPILER_BIN} ];then
        DependPackageCheck cuda HAVE_CUDA
    fi
}
fi

#恢复默认。
export _3RDPARTY_PKG_MACHINE=
export _3RDPARTY_PKG_WORDBIT=
export _3RDPARTY_PKG_FIND_ROOT=
export _3RDPARTY_PKG_FIND_MODE=

#
if [ "${THIRDPARTY_NOFOUND}" != "" ];then
{
    echo -e "\x1b[33m${THIRDPARTY_NOFOUND}\x1b[31m not found. \x1b[0m"
    exit 22
}
fi

#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${_TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${_TARGET_BITWIDE}
export _3RDPARTY_PKG_FIND_ROOT=${CUDNN_FIND_ROOT}
export _3RDPARTY_PKG_FIND_MODE=${THIRDPARTY_FIND_MODE}

#如果NVCC存在，再查找依赖组件。
if [ -f ${CUDA_COMPILER_BIN} ];then
    DependPackageCheck cudnn HAVE_CUDNN
fi

#恢复默认。
export _3RDPARTY_PKG_MACHINE=
export _3RDPARTY_PKG_WORDBIT=
export _3RDPARTY_PKG_FIND_ROOT=
export _3RDPARTY_PKG_FIND_MODE=

#
if [ "${THIRDPARTY_NOFOUND}" != "" ];then
{
    echo -e "\x1b[33m${THIRDPARTY_NOFOUND}\x1b[31m not found. \x1b[0m"
    exit 22
}
fi


#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${_TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${_TARGET_BITWIDE}
export _3RDPARTY_PKG_FIND_ROOT=${TRNSORRT_FIND_ROOT}
export _3RDPARTY_PKG_FIND_MODE=${THIRDPARTY_FIND_MODE}

#如果NVCC存在，再查找依赖组件。
if [ -f ${CUDA_COMPILER_BIN} ];then
    DependPackageCheck tensorrt HAVE_TENSORRT
fi

#恢复默认。
export _3RDPARTY_PKG_MACHINE=
export _3RDPARTY_PKG_WORDBIT=
export _3RDPARTY_PKG_FIND_ROOT=
export _3RDPARTY_PKG_FIND_MODE=

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
CC = ${_TARGET_COMPILER_BIN}
AR = ${_TARGET_COMPILER_AR}
#
NVCC = ${CUDA_COMPILER_BIN}
#
C_FLAGS = ${COMPILER_C_FLAGS}
CXX_FLAGS = ${COMPILER_CXX_FLAGS}
LD_FLAGS = ${COMPILER_LD_FLAGS}
#
DEPEND_FLAGS = ${DEPEND_FLAGS}
DEPEND_LINKS = ${DEPEND_LINKS} 
#
BUILD_TYPE = ${BUILD_TYPE}
#
OPTIMIZE_LEVEL = ${OPTIMIZE_LEVEL}
#
NATIVE_PLATFORM = ${_NATIVE_PLATFORM}
NATIVE_ARCH = ${_NATIVE_ARCH}
NATIVE_BITWIDE = ${_NATIVE_BITWIDE}
TARGET_PLATFORM = ${_TARGET_PLATFORM}
TARGET_ARCH = ${_TARGET_ARCH}
TARGET_BITWIDE = ${_TARGET_BITWIDE}
#
VERSION_MAJOR = ${VERSION_MAJOR}
VERSION_MINOR = ${VERSION_MINOR}
VERSION_RELEASE = ${VERSION_RELEASE}
VERSION_STR_MAIN = ${VERSION_STR_MAIN}
VERSION_STR_FULL = ${VERSION_STR_FULL}
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
${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN}
${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}
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
${INSTALL_PREFIX}/lib/libabcdk.so
${INSTALL_PREFIX}/lib/libabcdk.a
${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc
${INSTALL_PREFIX}/include/abcdk
${INSTALL_PREFIX}/include/abcdk.h

%post
#!/bin/sh
#
#echo "nothing to do."
#
exit 0

%postun
#!/bin/sh
#
#echo "nothing to do."
#
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
#
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/abcdk.sh
chmod 0755 /etc/profile.d/abcdk.sh
#
echo "${INSTALL_PREFIX}/lib" > /etc/ld.so.conf.d/abcdk.conf
ldconfig
#
exit 0
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/postrm <<EOF
#!/bin/sh
#
rm -f /etc/profile.d/abcdk.sh
#
rm -f /etc/ld.so.conf.d/abcdk.conf
#
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
#
#echo "nothing to do."
#
exit 0
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/postrm <<EOF
#!/bin/sh
#
#echo "nothing to do."
#
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
