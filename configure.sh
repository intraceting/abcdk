#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##


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
    if [ -f /etc/os-release ];then 
	    grep '^ID=' /etc/os-release |cut -d = -f 2 |sed 's/\"//g' | grep -iE "${1}" |wc -l
    elif [ -f /usr/lib/os-release ];then 
        grep '^ID=' /usr/lib/os-release |cut -d = -f 2 |sed 's/\"//g' | grep -iE "${1}" |wc -l
    else 
        echo "0"
    fi 
}

#
GetSystemVersion()
{
    if [ -f /etc/os-release ];then 
	    grep '^VERSION_ID=' /etc/os-release |cut -d = -f 2 |sed 's/\"//g'
    elif [ -f /usr/lib/os-release ];then 
        grep '^VERSION_ID=' /usr/lib/os-release |cut -d = -f 2 |sed 's/\"//g'
    else 
        echo "0"
    fi 
}

#
CheckPackageKitName()
{
	if [ $(CheckSystemName "Ubuntu|Debian") -ge 1 ];then
		echo "deb"
	elif [ $(CheckSystemName "CentOS|Red Hat|RedHat|RHEL|fedora|Amazon|amzn|Oracle") -ge 1 ];then
		echo "rpm"
	else
		echo ""
	fi
}

#
CheckHavePackageFromKit()
# $1 KIT_NAME
# $1 PACKAGE
{
    #
    STATUS="1"

    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
	if [ "deb" == "${KIT_NAME}" ];then 
        STATUS=$(dpkg -V ${PACKAGE} >> /dev/null 2>&1 ; echo $?)
	elif [ "rpm" == "${KIT_NAME}" ];then
		STATUS=$(rpm -q ${PACKAGE} >> /dev/null 2>&1 ; echo $?)
    fi

	#
	echo "${STATUS}"
}

#
CheckHavePackageFromWhich()
# $1 KIT_NAME
# $2 PACKAGE
{
    #
    STATUS="1"

    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
    STATUS=$(which ${PACKAGE} >> /dev/null 2>&1 ; echo $?)

	#
	echo "${STATUS}"
}

#
CheckHavePackage()
# $1 KIT_NAME
# $2 PKG_NAME
# $3 FLAG
{
    #
    KIT_NAME="$1"
    PKG_NAME="$2"
    FLAG="$3"

    #
    SYS_VERID=$(GetSystemVersion)

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{  
        if [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich ${KIT_NAME} pkg-config)"
            else 
                echo "pkg-config"
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libgomp1)"
            elif [ ${FLAG} -eq 2 ];then
                echo "-fopenmp"
            elif [ ${FLAG} -eq 3 ];then
                echo "-fopenmp"
            else
                echo "libgomp1"
            fi
        }
        elif [ "${PKG_NAME}" == "unixodbc" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} unixodbc-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lodbc"
            else
                echo "unixodbc-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "sqlite" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libsqlite3-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags sqlite3)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs sqlite3)"
            else
                echo "libsqlite3-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "openssl" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libssl-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags openssl)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs openssl)"
            else
                echo "libssl-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "ffmpeg" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev")"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            else
                echo "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "freeimage" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libfreeimage-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfreeimage"
            else
                echo "libfreeimage-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "fuse" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libfuse-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags fuse)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs fuse)"
            else
                echo "libfuse-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libnm" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libnm-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libnm)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libnm)"
            else
                echo "libnm-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "mpi" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmpich-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags mpi)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs mpi)"
            else
                echo "libmpich-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "lz4" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} liblz4-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags liblz4)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs liblz4)"
            else
                echo "liblz4-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "zlib" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} zlib1g-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zlib)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zlib)"
            else
                echo "zlib1g-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "archive" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libarchive-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libarchive)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libarchive)"
            else
                echo "libarchive-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "modbus" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmodbus-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmodbus)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmodbus)"
            else
                echo "libmodbus-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libusb" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libusb-1.0-0-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libusb-1.0)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libusb-1.0)"
            else
                echo "libusb-1.0-0-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "mqtt" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmosquitto-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lmosquitto"
            else
                echo "libmosquitto-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "redis" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libhiredis-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags hiredis)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs hiredis)"
            else
                echo "libhiredis-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "json-c" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libjson-c-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags json-c)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs json-c)"
            else
                echo "libjson-c-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "bluez" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libbluetooth-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags bluez)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs bluez)"
            else
                echo "libbluetooth-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "blkid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libblkid-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags blkid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs blkid)"
            else
                echo "libblkid-dev"
            fi
        }
        else
            echo "1"
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich ${KIT_NAME} pkg-config)"
            else
                echo "pkgconfig"
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libgomp)"
            elif [ ${FLAG} -eq 2 ];then
                echo "-fopenmp"
            elif [ ${FLAG} -eq 3 ];then
                echo "-fopenmp"
            else
                echo "libgomp"
            fi
        }
        elif [ "${PKG_NAME}" == "unixodbc" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} unixODBC-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lodbc"
            else
                echo "unixODBC-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "sqlite" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} sqlite-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags sqlite3)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs sqlite3)"
            else
                echo "sqlite-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "openssl" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} openssl-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags openssl)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs openssl)"
            else
                echo "openssl-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "ffmpeg" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} ffmpeg-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            else
                echo "ffmpeg-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "freeimage" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} freeimage-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfreeimage"
            else
                echo "freeimage-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "fuse" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} fuse-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags fuse)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs fuse)"
            else
                echo "fuse-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libnm" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} NetworkManager-libnm-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libnm)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libnm)"
            else
                echo "NetworkManager-libnm-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "mpi" ];then
        {
            if [ ${FLAG} -eq 1 ];then
            {
                if [ ${SYS_VERID} -lt 8 ];then
                    echo "$(CheckHavePackageFromKit ${KIT_NAME} mpich-3.2-devel)"
                else 
                    echo "$(CheckHavePackageFromKit ${KIT_NAME} mpich-devel hwloc-devel)"
                fi
            }
            else
            {
                if [ ${SYS_VERID} -lt 8 ];then
                    export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/usr/lib64/mpich-3.2/lib/pkgconfig
                else 
                    export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/usr/lib64/mpich/lib/pkgconfig
                fi

                if [ ${FLAG} -eq 2 ];then
                    echo "$(pkg-config --cflags mpich)"
                elif [ ${FLAG} -eq 3 ];then
                    echo "$(pkg-config --libs mpich)"
                else
                {
                    if [ ${SYS_VERID} -lt 8 ];then
                        echo "mpich-3.2-devel"
                    else
                        echo "mpich-devel hwloc-devel"
                    fi
                }
                fi
            }
            fi
        }
        elif [ "${PKG_NAME}" == "lz4" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} lz4-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags liblz4)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs liblz4)"
            else
                echo "lz4-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "zlib" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} zlib-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zlib)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zlib)"
            else
                echo "zlib-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "archive" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libarchive-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libarchive)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libarchive)"
            else
                echo "libarchive-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "modbus" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmodbus-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmodbus)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmodbus)"
            else
                echo "libmodbus-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libusb" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libusbx-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libusb-1.0)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libusb-1.0)"
            else
                echo "libusbx-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "mqtt" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} mosquitto-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmosquitto)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmosquitto)"
            else
                echo "mosquitto-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "redis" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} hiredis-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags hiredis)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs hiredis)"
            else
                echo "hiredis-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "json-c" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} json-c-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags json-c)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs json-c)"
            else
                echo "json-c-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "bluez" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} bluez-libs-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags bluez)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs bluez)"
            else
                echo "bluez-libs-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "blkid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libblkid-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags blkid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs blkid)"
            else
                echo "libblkid-devel"
            fi
        }
        else 
            echo "1"
        fi
    }
    else 
        echo "1"
    fi
}

#
CheckKeyword()
# $1 keywords
# $2 word
{
	NUM=$(echo "$1" |grep -wi "$2" | wc -l)
    echo ${NUM}
}

#
SHELL_PWD=$(cd `dirname $0`; pwd)

#
OS_ID=$(grep "^ID=" /etc/os-release |cut -d = -f 2 |sed 's/\"//g' |tr 'A-Z' 'a-z')
OS_VER=$(grep "^VERSION_ID=" /etc/os-release |cut -d = -f 2 |sed 's/\"//g' |tr 'A-Z' 'a-z')

#
MAKE_CONF=${SHELL_PWD}/build/makefile.conf

#
KIT_NAME=$(CheckPackageKitName)

#
SOLUTION_NAME=abcdk

#
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_PATH=$(realpath "${SHELL_PWD}/build/")

#
HOST_PLATFORM=$(uname -m)
TARGET_PLATFORM=${HOST_PLATFORM}

#主版本
VERSION_MAJOR="1"
#副版本
VERSION_MINOR="2"
#发行版本
VERSION_RELEASE="1"

#
BUILD_TYPE="release"

#
INSTALL_PREFIX="/usr/local/"

#
DEPEND_FUNC="Nothing"

#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]
    -p < STRING >  
     目标系统平台。支持x86_64，aarch64。
     默认：${TARGET_PLATFORM}

    -g  
     生成调试符号。默认：关闭

    -V < NUMBER > 
     主版本。默认：${VERSION_MAJOR}

    -v < NUMBER > 
     副版本。默认：${VERSION_MINOR}

    -r < NUMBER > 
     发行版本。默认：${VERSION_RELEASE}

    -i < PATH > 
     安装路径。默认：${INSTALL_PREFIX}

    -d < KEY,KEY,... > 
     依赖项目，以英文“,”为分割符。支持以下关键字：
     openmp,unixodbc,sqlite,openssl,ffmpeg,
     freeimage,fuse,libnm,mpi,lz4,zlib,
     archive,modbus,libusb,mqtt,redis,json-c,
     bluez,blkid
EOF
}

#
while getopts "?p:gV:v:r:i:d:" ARGKEY 
do
    case $ARGKEY in
    \?)
        PrintUsage
        exit 22
    ;;
    p)
        TARGET_PLATFORM="${OPTARG}"
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
        DEPEND_FUNC="$OPTARG"
    ;;
    esac
done

# Compiler
CC=gcc
AR=ar

# 可能在交叉编译环中。
if [ "${TARGET_PLATFORM}" != "${HOST_PLATFORM}" ];then
CC=${TARGET_PLATFORM}-linux-gnu-gcc
AR=${TARGET_PLATFORM}-linux-gnu-ar
fi

#
STATUS=$(CheckHavePackageFromWhich ${KIT_NAME} ${CC})
if [ ${STATUS} -ne 0 ];then
{
    echo "${CC} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackageFromWhich ${KIT_NAME} ${AR})
if [ ${STATUS} -ne 0 ];then
{
    echo "${AR} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackage ${KIT_NAME} pkgconfig 1)
if [ ${STATUS} -ne 0 ];then
{
    echo "$(CheckHavePackage ${KIT_NAME} pkgconfig 0) not found."
    exit 22
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "openmp") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} openmp 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_OPENMP="Yes"
        DEPEND_FLAGS=" -DHAVE_OPENMP $(CheckHavePackage ${KIT_NAME} openmp 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} openmp 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} openmp 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "unixodbc") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} unixodbc 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_UNIXODBC="Yes"
        DEPEND_FLAGS=" -DHAVE_UNIXODBC $(CheckHavePackage ${KIT_NAME} unixodbc 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage  ${KIT_NAME} unixodbc 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} unixodbc 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "sqlite") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} sqlite 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_SQLITE="Yes"
        DEPEND_FLAGS=" -DHAVE_SQLITE $(CheckHavePackage ${KIT_NAME} sqlite 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} sqlite 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} sqlite 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "openssl") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} openssl 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_OPENSSL="Yes"
        DEPEND_FLAGS=" -DHAVE_OPENSSL $(CheckHavePackage ${KIT_NAME} openssl 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} openssl 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} openssl 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "ffmpeg") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} ffmpeg 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FFMPEG="Yes"
        DEPEND_FLAGS=" -DHAVE_FFMPEG $(CheckHavePackage ${KIT_NAME} ffmpeg 2 ) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage  ${KIT_NAME} ffmpeg 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} ffmpeg 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "freeimage") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} freeimage 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FREEIMAGE="Yes"
        DEPEND_FLAGS=" -DHAVE_FREEIMAGE $(CheckHavePackage ${KIT_NAME} freeimage 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} freeimage 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} freeimage 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "fuse") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} fuse 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FUSE="Yes"
        DEPEND_FLAGS=" -DHAVE_FUSE $(CheckHavePackage ${KIT_NAME} fuse 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} fuse 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} fuse 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "libnm") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} libnm 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_LIBNM="Yes"
        DEPEND_FLAGS=" -DHAVE_LIBNM $(CheckHavePackage ${KIT_NAME} libnm 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} libnm 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} libnm 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "mpi") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} mpi 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_MPI="Yes"
        DEPEND_FLAGS=" -DHAVE_MPI $(CheckHavePackage ${KIT_NAME} mpi 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} mpi 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} mpi 0) not found."
        exit 22
    }
    fi
}
fi


#
if [ $(CheckKeyword ${DEPEND_FUNC} "lz4") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} lz4 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_LZ4="Yes"
        DEPEND_FLAGS=" -DHAVE_LZ4 $(CheckHavePackage ${KIT_NAME} lz4 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} lz4 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} lz4 0) not found."
        exit 22
    }
    fi
}
fi


#
if [ $(CheckKeyword ${DEPEND_FUNC} "zlib") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} zlib 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_ZLIB="Yes"
        DEPEND_FLAGS=" -DHAVE_ZLIB $(CheckHavePackage ${KIT_NAME} zlib 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} zlib 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} zlib 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "archive") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} archive 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_ARCHIVE="Yes"
        DEPEND_FLAGS=" -DHAVE_ARCHIVE $(CheckHavePackage ${KIT_NAME} archive 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} archive 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} archive 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "modbus") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} modbus 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_MODBUS="Yes"
        DEPEND_FLAGS=" -DHAVE_MODBUS $(CheckHavePackage ${KIT_NAME} modbus 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} modbus 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} modbus 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "libusb") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} libusb 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_LIBUSB="Yes"
        DEPEND_FLAGS=" -DHAVE_LIBUSB $(CheckHavePackage ${KIT_NAME} libusb 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} libusb 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} libusb 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "mqtt") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} mqtt 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_MQTT="Yes"
        DEPEND_FLAGS=" -DHAVE_MQTT $(CheckHavePackage ${KIT_NAME} mqtt 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} mqtt 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} mqtt 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "redis") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} redis 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_REDIS="Yes"
        DEPEND_FLAGS=" -DHAVE_REDIS $(CheckHavePackage ${KIT_NAME} redis 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} redis 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} redis 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "json-c") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} json-c 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_JSON_C="Yes"
        DEPEND_FLAGS=" -DHAVE_JSON_C $(CheckHavePackage ${KIT_NAME} json-c 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} json-c 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} json-c 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "bluez") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} bluez 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_BLUEZ="Yes"
        DEPEND_FLAGS=" -DHAVE_BLUEZ $(CheckHavePackage ${KIT_NAME} bluez 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} bluez 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} bluez 0) not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "blkid") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} blkid 1)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_BLKID="Yes"
        DEPEND_FLAGS=" -DHAVE_BLKID $(CheckHavePackage ${KIT_NAME} blkid 2) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} blkid 3) ${DEPEND_LIBS}"
    }
    else
    {
        echo "$(CheckHavePackage ${KIT_NAME} blkid 0) not found."
        exit 22
    }
    fi
}
fi
#
mkdir -p ${BUILD_PATH}

#
if [ ! -d ${BUILD_PATH} ];then
echo "'${BUILD_PATH}' must be an existing directory."
exit 22
fi

#
if [ ! -d ${INSTALL_PREFIX} ];then
echo "'${INSTALL_PREFIX}' must be an existing directory."
exit 22
else
INSTALL_PREFIX="${INSTALL_PREFIX}/${SOLUTION_NAME}/"
fi

#
DEPEND_FLAGS="${DEPEND_FLAGS} -D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"

#
DEPEND_LIBS="${DEPEND_LIBS} -ldl -pthread -lrt -lc -lm"

#
echo "MAKE_CONF=${MAKE_CONF}"

#
echo "OS_ID=${OS_ID}"
echo "OS_VER=${OS_VER}"

#
echo "KIT_NAME=${KIT_NAME}"

#
echo "SOLUTION_NAME=${SOLUTION_NAME}"

#
echo "BUILD_TIME=${BUILD_TIME}"
echo "BUILD_PATH=${BUILD_PATH}"

#
echo "HOST_PLATFORM=${HOST_PLATFORM}"
echo "TARGET_PLATFORM=${TARGET_PLATFORM}"

#
echo "CC=${CC}"
echo "AR=${AR}"

#
echo "VERSION_MAJOR=${VERSION_MAJOR}"
echo "VERSION_MINOR=${VERSION_MINOR}"
echo "VERSION_RELEASE=${VERSION_RELEASE}"

#
echo "HAVE_OPENMP=${HAVE_OPENMP}"
echo "HAVE_UNIXODBC=${HAVE_UNIXODBC}"
echo "HAVE_SQLITE=${HAVE_SQLITE}"
echo "HAVE_OPENSSL=${HAVE_OPENSSL}"
echo "HAVE_FFMPEG=${HAVE_FFMPEG}"
echo "HAVE_FREEIMAGE=${HAVE_FREEIMAGE}"
echo "HAVE_FUSE=${HAVE_FUSE}"
echo "HAVE_LIBNM=${HAVE_LIBNM}"
echo "HAVE_MPI=${HAVE_MPI}"
echo "HAVE_LZ4=${HAVE_LZ4}"
echo "HAVE_ZLIB=${HAVE_ZLIB}"
echo "HAVE_ARCHIVE=${HAVE_ARCHIVE}"
echo "HAVE_MODBUS=${HAVE_MODBUS}"
echo "HAVE_LIBUSB=${HAVE_LIBUSB}"
echo "HAVE_MQTT=${HAVE_MQTT}"
echo "HAVE_REDIS=${HAVE_REDIS}"
echo "HAVE_JSON_C=${HAVE_JSON_C}"
echo "HAVE_BLUEZ=${HAVE_BLUEZ}"
echo "HAVE_BLKID=${HAVE_BLKID}"

#
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "ROOT_PATH?=/"

#
echo "#" > ${MAKE_CONF}
checkReturnCode

echo "# A better c development kit." >> ${MAKE_CONF}
echo "#" >> ${MAKE_CONF}
echo "" >> ${MAKE_CONF}

#
echo "OS_ID=${OS_ID}" >> ${MAKE_CONF}
echo "OS_VER=${OS_VER}" >> ${MAKE_CONF}

#
echo "KIT_NAME = ${KIT_NAME}" >> ${MAKE_CONF}

#
echo "SOLUTION_NAME = ${SOLUTION_NAME}" >> ${MAKE_CONF}

#
echo "BUILD_TIME = ${BUILD_TIME}" >> ${MAKE_CONF}
echo "BUILD_PATH = ${BUILD_PATH}" >> ${MAKE_CONF}

#
echo "HOST_PLATFORM = ${HOST_PLATFORM}" >> ${MAKE_CONF}
echo "TARGET_PLATFORM = ${TARGET_PLATFORM}" >> ${MAKE_CONF}
#
echo "CC = ${CC}" >> ${MAKE_CONF}
echo "AR = ${AR}" >> ${MAKE_CONF}

#
echo "VERSION_MAJOR = ${VERSION_MAJOR}" >> ${MAKE_CONF}
echo "VERSION_MINOR = ${VERSION_MINOR}" >> ${MAKE_CONF}
echo "VERSION_RELEASE = ${VERSION_RELEASE}" >> ${MAKE_CONF}

#
echo "DEPEND_FLAGS = ${DEPEND_FLAGS}" >> ${MAKE_CONF}
echo "DEPEND_LIBS = ${DEPEND_LIBS}" >> ${MAKE_CONF}

#
echo "BUILD_TYPE = ${BUILD_TYPE}" >> ${MAKE_CONF}

#
echo "INSTALL_PREFIX = ${INSTALL_PREFIX}" >> ${MAKE_CONF}

#
echo "ROOT_PATH ?= /" >> ${MAKE_CONF}
