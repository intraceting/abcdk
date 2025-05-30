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
CheckHavePackage()
# $1 PKG_NAME
# $2 FLAG
{
    ${SHELLDIR}/3rdparty/$1.sh "$2"
}

#
CheckPackageKitName()
{
	${SHELLDIR}/tools/get-kit-name.sh
}

#
CheckKeyword()
# $1 keywords
# $2 word
{
    ${SHELLDIR}/tools/check-keyword.sh "$1" "$2"
}

#
CheckSTD()
# $1 LANG
# $2 COMPILER
# $3 STD
{
    ${SHELLDIR}/tools/check-$1-std.sh "$2" "$3"
}

#
GetCompilerArch()
#$1 BIN
{
    ${SHELLDIR}/tools/get-compiler-arch.sh "$1" 
}


#
GetCompilerBitWide()
#$1 BIN
{
    ${SHELLDIR}/tools/get-compiler-bitwide.sh "$1" 
}


#
GetCompilerMachine()
#$1 BIN
{
    ${SHELLDIR}/tools/get-compiler-machine.sh "$1" 
}

#
GetCompilerPlatform()
#$1 BIN
{
    ${SHELLDIR}/tools/get-compiler-platform.sh "$1" 
}

#
GetCompilerProgName()
#$1 BIN
#$2 NAME
{
    ${SHELLDIR}/tools/get-compiler-prog-name.sh "$1" "$2"
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
            THIRDPARTY_FLAGS="-D${PACKAGE_DEF} $(CheckHavePackage ${PACKAGE_KEY} 2) ${THIRDPARTY_FLAGS}"
            THIRDPARTY_LINKS="$(CheckHavePackage ${PACKAGE_KEY} 3) ${THIRDPARTY_LINKS}"
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

#    echo ${THIRDPARTY_FLAGS} 
#    echo ${THIRDPARTY_LINKS}
}

#主版本
VERSION_MAJOR="3"
#副版本
VERSION_MINOR="5"
#发行版本
VERSION_RELEASE="7"

#
LSB_RELEASE="linux-gnu"

#
SYSROOT_PREFIX="/"
INSTALL_PREFIX="/usr/local/"

#
BUILD_PATH="${SHELLDIR}/build/"

#组件包名称。
KIT_NAME=""

#安装包后缀。
PACKAGE_SUFFIX=""

#安装包存放路径。
PACKAGE_PATH="${SHELLDIR}/package/${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}"

#
XGETTEXT_BIN=$(which xgettext)
MSGFMT_BIN=$(which msgfmt)
MSGCAT_BIN=$(which msgcat)

#
COMPILER_PREFIX=/usr/bin/
COMPILER_C_NAME=gcc
COMPILER_CXX_NAME=g++

#
COMPILER_C_FLAGS=""
COMPILER_CXX_FLAGS=""
COMPILER_LD_FLAGS=""

#
BUILD_TYPE="release"
#
OPTIMIZE_LEVEL=""

#
THIRDPARTY_PACKAGES=""
THIRDPARTY_FIND_ROOT=""
THIRDPARTY_FIND_MODE="both"
THIRDPARTY_NOFOUND=""

#
CUDA_FIND_ROOT=""
CUDA_COMPILER_BIN=""

#
CUDNN_FIND_ROOT=""

#
TRNSORRT_FIND_ROOT=""

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
    
     VERSION_MAJOR=${VERSION_MAJOR}

     VERSION_MAJOR=(主版本)
     
     VERSION_MINOR=${VERSION_MINOR}

     VERSION_MINOR=(副版本)
     
     VERSION_RELEASE=${VERSION_RELEASE}

     VERSION_RELEASE=(发行版本)

     LSB_RELEASE=${LSB_RELEASE}

     LSB_RELEASE(发行版名称)支持以下关键字:
     linux-gnu,android
          
     SYSROOT_PREFIX=${SYSROOT_PREFIX}

     SYSROOT_PREFIX(系统路径的前缀).

     INSTALL_PREFIX=${INSTALL_PREFIX}

     INSTALL_PREFIX(安装路经的前缀).

     BUILD_PATH=${BUILD_PATH}

     BUILD_PATH(过程文件存放的路径)用于存放构建过程文件.

     KIT_NAME=${TARGET_KIT_NAME}

     KIT_NAME(组件包名字)支持以下关键字:
     deb,rpm,(local)

     PACKAGE_SUFFIX=${PACKAGE_SUFFIX}

     PACKAGE_SUFFIX(安装包名称后缀).

     PACKAGE_PATH=${PACKAGE_PATH}

     PACKAGE_PATH(安装包存放的路径)用于存放安装包文件.

     OPTIMIZE_LEVEL=${OPTIMIZE_LEVEL}

     OPTIMIZE_LEVEL(优化级别)支持以下关键字：
     1,2,3,s,fast

     BUILD_TYPE=${BUILD_TYPE}
    
     BUILD_TYPE(构建类型)支持以下关键字：
     debug,release
     
     COMPILER_PREFIX=${COMPILER_PREFIX}

     COMPILER_PREFIX(C/C++编译器路径的前缀)与编译器名字组成完整路径.

     COMPILER_C_NAME=${COMPILER_C_NAME}

     COMPILER_C_NAME(C编译器的名字)与编译器前缀组成完整路径.

     COMPILER_CXX_NAME=${COMPILER_CXX_NAME}

     COMPILER_CXX_NAME(C++编译器的名字)与编译器前缀组成完整路径.

     COMPILER_C_FLAGS=${COMPILER_C_FLAGS}

     COMPILER_C_FLAGS(C编译器的编译参数)用于编译器的源码编译. 

     COMPILER_CXX_FLAGS=${COMPILER_CXX_FLAGS}

     COMPILER_CXX_FLAGS(C++编译器的编译参数)用于编译器的源码编译. 

     COMPILER_LD_FLAGS=${COMPILER_LD_FLAGS}

     COMPILER_LD_FLAGS(编译器的链接参数)用于编译器的目标链接. 

     THIRDPARTY_PACKAGES=${THIRDPARTY_PACKAGES}

     THIRDPARTY_PACKAGES(依赖组件列表)支持以下关键字:
     ffmpeg,json-c,lz4,cuda,cudnn,ffnvcodec,
     unixodbc,opencv,openssl,redis,sqlite,curl,
     archive,nghttp2,libmagic,gtk,appindicator,
     tensorrt,live555,onnxruntime,x264,x265,
     qrencode,zlib,freeimage,fuse,libnm,
     openmp,modbus,libusb,mqtt,
     bluez,blkid,libcap,fastcgi,systemd,
     libudev,dmtx,,zbar,magickwand,
     kafka,uuid,libdrm,
     pam,ncurses,fltk,
     

     THIRDPARTY_FIND_ROOT=\${INSTALL_PREFIX}

     THIRDPARTY_FIND_ROOT(依赖组件搜索根路径)用于查找依赖组件完整路径.

     THIRDPARTY_FIND_MODE=${THIRDPARTY_FIND_MODE}

     THIRDPARTY_FIND_MODE(依赖组件搜索模式)支持以下关键字:
     only,both,(default)

     CUDA_FIND_ROOT=\${THIRDPARTY_FIND_ROOT}/cuda/

     CUDA_FIND_ROOT(CUDA组件搜索根路径)用于查找依赖组件完整路径.

     CUDA_COMPILER_BIN=\${CUDA_FIND_ROOT}/bin/nvcc

     CUDA_COMPILER_BIN(CUDA编译器的完整路径).

     CUDNN_FIND_ROOT=\${CUDA_FIND_ROOT}/

     CUDNN_FIND_ROOT(CUDNN组件搜索根路径)用于查找依赖组件完整路径.

     TRNSORRT_FIND_ROOT=\${THIRDPARTY_FIND_ROOT}/TensorRT/

     TRNSORRT_FIND_ROOT(TensorRT组件搜索根路径)用于查找依赖组件完整路径.

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
mkdir -p ${BUILD_PATH}
if [ ! -d ${BUILD_PATH} ];then
{
    echo "'BUILD_PATH=${BUILD_PATH}' invalid or unsupported."
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

#如果未指定组件包名称，则认为是与本地平台一致。
if [ "${KIT_NAME}" == "" ];then
KIT_NAME=$(CheckPackageKitName)
fi

#安装包存放路径必须有效。
if [ "${PACKAGE_PATH}" == "" ] ;then
{
    echo "'PACKAGE_PATH=${PACKAGE_PATH}' invalid or unsupported."
    exit 22
}
fi

#
TARGET_COMPILER_C=${COMPILER_PREFIX}${COMPILER_C_NAME}
TARGET_COMPILER_CXX=${COMPILER_PREFIX}${COMPILER_CXX_NAME}

#
CheckSTD c "${TARGET_COMPILER_C}" "c99"
if [ $? -ne 0 ];then
{
    echo "The compiler supports at least the c99 standard."
    exit 22
}
fi

#
CheckSTD cxx "${TARGET_COMPILER_CXX}" "c++11"
if [ $? -ne 0 ];then
{
    echo "The compiler supports at least the c++11 standard."
    exit 22
}
fi

#
TARGET_COMPILER_AR=$(GetCompilerProgName "${TARGET_COMPILER_C}" "ar")
TARGET_MACHINE=$(GetCompilerMachine "${TARGET_COMPILER_C}" )
TARGET_PLATFORM=$(GetCompilerPlatform "${TARGET_COMPILER_C}")
TARGET_ARCH=$(GetCompilerArch "${TARGET_COMPILER_C}")
TARGET_BITWIDE=$(GetCompilerBitWide "${TARGET_COMPILER_C}")

#
CheckHavePackage pkgconfig 1
if [ $? -ne 0 ];then
    echo "'$(CheckHavePackage pkgconfig 4)' not found."
    exit 22
fi


#如果未指定第三方根路径，则直接用安装路径。
if [ "${THIRDPARTY_FIND_ROOT}" == "" ];then
    THIRDPARTY_FIND_ROOT=${INSTALL_PREFIX}
fi

#
if [ "${CUDA_FIND_ROOT}" == "" ];then
CUDA_FIND_ROOT="${THIRDPARTY_FIND_ROOT}/cuda/"
fi

#
if [ "${CUDA_COMPILER_BIN}" == "" ];then
CUDA_COMPILER_BIN="${CUDA_FIND_ROOT}/bin/nvcc"
fi

#
if [ "${CUDNN_FIND_ROOT}" == "" ];then
CUDNN_FIND_ROOT="${CUDA_FIND_ROOT}/"
fi

#
if [ " ${TRNSORRT_FIND_ROOT}" == "" ];then
TRNSORRT_FIND_ROOT="${THIRDPARTY_FIND_ROOT}/TensorRT/"
fi


#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${TARGET_BITWIDE}
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
DependPackageCheck appindicator HAVE_APPINDICATOR
DependPackageCheck x264 HAVE_H264
DependPackageCheck x265 HAVE_H265
DependPackageCheck ffnvcodec HAVE_FFNVCODEC
DependPackageCheck opencv HAVE_OPENCV
DependPackageCheck live555 HAVE_LIVE555
DependPackageCheck onnxruntime HAVE_ONNXRUNTIME


#恢复默认。
export _3RDPARTY_PKG_MACHINE=
export _3RDPARTY_PKG_WORDBIT=
export _3RDPARTY_PKG_FIND_ROOT=
export _3RDPARTY_PKG_FIND_MODE=


#设置环境变量，用于搜索依赖包。
export _3RDPARTY_PKG_MACHINE=${TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${TARGET_BITWIDE}
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
    else 
        THIRDPARTY_NOFOUND="${CUDA_COMPILER_BIN} ${THIRDPARTY_NOFOUND}"
    fi
}
else
{
    CUDA_COMPILER_BIN=""
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
export _3RDPARTY_PKG_MACHINE=${TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${TARGET_BITWIDE}
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
export _3RDPARTY_PKG_MACHINE=${TARGET_MACHINE}
export _3RDPARTY_PKG_WORDBIT=${TARGET_BITWIDE}
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

#提取第三方依整包的所有路径。
THIRDPARTY_LIBS_PATH=$(echo "${THIRDPARTY_LINKS}" | tr ' ' '\n' | grep "^-L" | sed 's/^-L//' | sort | uniq | tr '\n' ':' | sed 's/:$//')

#
MAKE_CONF=${BUILD_PATH}/makefile.conf

#
cat >${MAKE_CONF} <<EOF
#
VERSION_MAJOR = ${VERSION_MAJOR}
VERSION_MINOR = ${VERSION_MINOR}
VERSION_RELEASE = ${VERSION_RELEASE}
#
LSB_RELEASE = ${LSB_RELEASE}
#
SYSROOT_PREFIX ?= ${SYSROOT_PREFIX}
#
INSTALL_PREFIX ?= ${INSTALL_PREFIX}
#
BUILD_PATH = ${BUILD_PATH}
#
KIT_NAME = ${KIT_NAME}
#
PACKAGE_SUFFIX = ${PACKAGE_SUFFIX}
#
PACKAGE_PATH = ${PACKAGE_PATH}
#
XGETTEXT = ${XGETTEXT_BIN}
MSGFMT = ${MSGFMT_BIN}
MSGCAT = ${MSGCAT_BIN}
#
CC = ${TARGET_COMPILER_C}
CXX = ${TARGET_COMPILER_CXX}
AR = ${TARGET_COMPILER_AR}
#
NVCC = ${CUDA_COMPILER_BIN}
#
EXTRA_C_FLAGS = ${COMPILER_C_FLAGS}
EXTRA_CXX_FLAGS = ${COMPILER_CXX_FLAGS}
EXTRA_LD_FLAGS = ${COMPILER_LD_FLAGS}
#
BUILD_TYPE = ${BUILD_TYPE}
#
OPTIMIZE_LEVEL = ${OPTIMIZE_LEVEL}
#
TARGET_PLATFORM = ${TARGET_PLATFORM}
TARGET_ARCH = ${TARGET_ARCH}
TARGET_BITWIDE = ${TARGET_BITWIDE}
#
DEPEND_FLAGS = ${THIRDPARTY_FLAGS}
DEPEND_LINKS = ${THIRDPARTY_LINKS}
#
DEPEND_LIB_PATH = ${THIRDPARTY_LIBS_PATH}
#
DEV_TOOL_HOME = ${SHELLDIR}/tools/

EOF
checkReturnCode

