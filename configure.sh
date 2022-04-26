#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##

#
CURDIR=$(cd `dirname $0`; pwd)

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
    echo "$(${CURDIR}/3party/myscript/linux/core/check_os_id.sh $1)"
}

#
GetSystemVersion()
{
    echo "$(${CURDIR}/3party/myscript/linux/core/get_os_ver.sh)"
}

#
CheckPackageKitName()
{
	echo "$(${CURDIR}/3party/myscript/linux/core/get_kit_name.sh)"
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    echo "$(${CURDIR}/3party/myscript/linux/core/check_package.sh $1)"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	echo "$(${CURDIR}/3party/myscript/linux/core/check_which.sh $1)"
}

#
CheckHavePackage()
# $1 PKG_NAME
# $2 FLAG
{
    echo "$(${CURDIR}/3party/myscript/linux/devel/check_config.sh $1 $2)"
}

#
CheckKeyword()
# $1 keywords
# $2 word
{
	NUM=$(echo "$1" |grep -wi "$2" | wc -l)
    echo ${NUM}
}

#拉取子项目
git submodule update --init --remote --merge --recursive
checkReturnCode

#
MAKE_CONF=${CURDIR}/build/makefile.conf

#
KIT_NAME=$(CheckPackageKitName)

#
SOLUTION_NAME=abcdk

#
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_PATH=$(realpath "${CURDIR}/build/")


#主版本
VERSION_MAJOR="1"
#副版本
VERSION_MINOR="3"
#发行版本
VERSION_RELEASE="7"

#
BUILD_TYPE="release"

#
INSTALL_PREFIX="/usr/local/"

#
DEPEND_FUNC="Nothing"
DEPEND_NOFOUND=""
DEPEND_REQUIRES=""

#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]
    -g  
     生成调试符号。默认：关闭

     自定义编译器，并且定义环境变量。如下：
     export CC=gcc
     export AR=ar

    -V < number > 
     主版本。默认：${VERSION_MAJOR}

    -v < number > 
     副版本。默认：${VERSION_MINOR}

    -r < number > 
     发行版本。默认：${VERSION_RELEASE}

    -i < path > 
     安装路径。默认：${INSTALL_PREFIX}

    -d < key,key,... > 
     依赖项目，以英文“,”为分割符。支持以下关键字：
     openmp,unixodbc,sqlite,openssl,ffmpeg,
     freeimage,fuse,libnm,lz4,zlib,
     archive,modbus,libusb,mqtt,redis,json-c,
     bluez,blkid,libcap,fastcgi,systemd,
     libudev,dmtx,qrencode,zbar,magickwand

     自定义依赖项，key前缀增加“with-”，并且定义环境变量。如下：
     export DEPEND_FLAGS="-I/tmp/3party/include/"
     export DEPEND_LIBS="-l:3party.so -l:3party.a -l3party -L/tmp/3party/lib/"
EOF
}

#
while getopts "hgV:v:r:i:d:" ARGKEY 
do
    case $ARGKEY in
    h)
        PrintUsage
        exit 22
    ;;
    g)
        BUILD_TYPE="debug"
    ;;
    V)
        VERSION_MAJOR="${OPTARG}"
    ;;
    v)
        VERSION_MINOR="${OPTARG}"
    ;;
    r)
        VERSION_RELEASE="${OPTARG}"
    ;;
    i)
        INSTALL_PREFIX=$(realpath "${OPTARG}")
    ;;
    d)
        DEPEND_FUNC="${OPTARG}"
    ;;
    esac
done

# 设置编译器。
if [ "${CC}" == "" ];then
    CC=gcc
fi
if [ "${AR}" == "" ];then
    AR=ar
fi

#
STATUS=$(CheckHavePackageFromWhich ${CC})
if [ ${STATUS} -ne 0 ];then
{
    echo "${CC} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackageFromWhich ${AR})
if [ ${STATUS} -ne 0 ];then
{
    echo "${AR} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackage pkgconfig 1)
if [ ${STATUS} -ne 0 ];then
{
    echo "$(CheckHavePackage pkgconfig 0) not found."
    exit 22
}
fi

#
DependPackageCheck()
# 1 key
# 2 def
{
    PACKAGE_KEY=$1
    PACKAGE_DEF=$2
    #
    if [ $(CheckKeyword ${DEPEND_FUNC} ${PACKAGE_KEY}) -eq 1 ];then
    {
        if [ $(CheckKeyword ${DEPEND_FUNC} with-${PACKAGE_KEY}) -eq 1 ];then
        {
            DEPEND_FLAGS=" -D${PACKAGE_DEF} ${DEPEND_FLAGS}"
        }
        else
        {
            CHK=$(CheckHavePackage ${PACKAGE_KEY} 1)
            if [ ${CHK} -eq 0 ];then
            {
                DEPEND_FLAGS=" -D${PACKAGE_DEF} $(CheckHavePackage ${PACKAGE_KEY} 2) ${DEPEND_FLAGS}"
                DEPEND_LIBS=" $(CheckHavePackage ${PACKAGE_KEY} 3) ${DEPEND_LIBS}"
            }
            else
            {
                DEPEND_NOFOUND="$(CheckHavePackage ${PACKAGE_KEY} 0) ${DEPEND_NOFOUND}"
            }
            fi
        }
        fi
    }
    fi

#    echo ${DEPEND_FLAGS} 
#    echo ${DEPEND_LIBS}
}

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

#
if [ "${DEPEND_NOFOUND}" != "" ];then
{
    echo "${DEPEND_NOFOUND} no found."
    exit 22
}
fi 

#
TARGET_PLATFORM=$(${CC} -dumpmachine)

#
VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}

#
mkdir -p ${BUILD_PATH}

#
if [ ! -d ${BUILD_PATH} ];then
{
    echo "'${BUILD_PATH}' must be an existing directory."
    exit 22
}
fi

#
if [ ! -d ${INSTALL_PREFIX} ];then
{
    echo "'${INSTALL_PREFIX}' must be an existing directory."
    exit 22
}
else
{
    INSTALL_PREFIX="${INSTALL_PREFIX}/${SOLUTION_NAME}-${VERSION_STR}/"
}
fi

#
DEPEND_FLAGS="${DEPEND_FLAGS} -D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"

#
DEPEND_LIBS="${DEPEND_LIBS} -ldl -pthread -lrt -lc -lm"

#
echo "MAKE_CONF=${MAKE_CONF}"

#
echo "SOLUTION_NAME=${SOLUTION_NAME}"

#
echo "BUILD_TIME=${BUILD_TIME}"
echo "BUILD_PATH=${BUILD_PATH}"

#
echo "TARGET_PLATFORM=${TARGET_PLATFORM}"

#
echo "CC=${CC}"
echo "AR=${AR}"

#
echo "VERSION_MAJOR=${VERSION_MAJOR}"
echo "VERSION_MINOR=${VERSION_MINOR}"
echo "VERSION_RELEASE=${VERSION_RELEASE}"
echo "VERSION_STR=${VERSION_STR}"

#
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "ROOT_PATH?=/"

#
echo "#" > ${MAKE_CONF}
checkReturnCode

#
echo "# A bad c development kit." >> ${MAKE_CONF}
echo "#" >> ${MAKE_CONF}
echo "" >> ${MAKE_CONF}

#
echo "SOLUTION_NAME = ${SOLUTION_NAME}" >> ${MAKE_CONF}

#
echo "BUILD_TIME = ${BUILD_TIME}" >> ${MAKE_CONF}
echo "BUILD_PATH = ${BUILD_PATH}" >> ${MAKE_CONF}

#
echo "TARGET_PLATFORM = ${TARGET_PLATFORM}" >> ${MAKE_CONF}


#
echo "CC = ${CC}" >> ${MAKE_CONF}
echo "AR = ${AR}" >> ${MAKE_CONF}

#
echo "VERSION_MAJOR = ${VERSION_MAJOR}" >> ${MAKE_CONF}
echo "VERSION_MINOR = ${VERSION_MINOR}" >> ${MAKE_CONF}
echo "VERSION_RELEASE = ${VERSION_RELEASE}" >> ${MAKE_CONF}
echo "VERSION_STR = ${VERSION_STR}" >> ${MAKE_CONF}

#
echo "DEPEND_FLAGS = ${DEPEND_FLAGS}" >> ${MAKE_CONF}
echo "DEPEND_LIBS = ${DEPEND_LIBS}" >> ${MAKE_CONF}

#
echo "BUILD_TYPE = ${BUILD_TYPE}" >> ${MAKE_CONF}

#
echo "INSTALL_PREFIX = ${INSTALL_PREFIX}" >> ${MAKE_CONF}

#
echo "ROOT_PATH ?= /" >> ${MAKE_CONF}


