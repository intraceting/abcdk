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
CheckPackageKitName()
{
	if [ $(CheckSystemName "Ubuntu|Debian") -ge 1 ];then
		echo "deb"
	elif [ $(CheckSystemName "CentOS|Red Hat|RedHat|RHEL|fedora|Amazon|Oracle") -ge 1 ];then
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
# $2 PACKAGE
{
    # 0 is Ok, otherwise not Ok.
    STATUS="1" 
    NAMES=""

    # 1 is KIT,2 is WHICH.
    METHOD=1

    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{  
        if [ "${PACKAGE}" == "pkgconfig" ];then
        {
            METHOD=2
            NAMES="pkg-config"
        } 
        elif [ "${PACKAGE}" == "openmp" ];then
            NAMES="libomp-dev"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            NAMES="unixodbc-dev"
        elif [ "${PACKAGE}" == "sqlite" ];then
            NAMES="libsqlite3-dev"
        elif [ "${PACKAGE}" == "openssl" ];then
            NAMES="libssl-dev"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            NAMES="libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev"
        elif [ "${PACKAGE}" == "freeimage" ];then
            NAMES="libfreeimage-dev"
        elif [ "${PACKAGE}" == "fuse" ];then
            NAMES="libfuse-dev"
        elif [ "${PACKAGE}" == "libnm" ];then
            NAMES="libnm-dev"
        elif [ "${PACKAGE}" == "mpi" ];then
            NAMES="libmpich-dev"
        elif [ "${PACKAGE}" == "lz4" ];then
            NAMES="liblz4-dev"
        elif [ "${PACKAGE}" == "zlib" ];then
            NAMES="zlib1g-dev"
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PACKAGE}" == "pkgconfig" ];then
        {
            METHOD=2
            NAMES="pkg-config"
        }
        elif [ "${PACKAGE}" == "openmp" ];then
            NAMES="gcc"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            NAMES="unixODBC-devel"
        elif [ "${PACKAGE}" == "sqlite" ];then
            NAMES="sqlite-devel"
        elif [ "${PACKAGE}" == "openssl" ];then
            NAMES="openssl-devel"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            NAMES="ffmpeg-devel"
        elif [ "${PACKAGE}" == "freeimage" ];then
            NAMES="freeimage-devel"
        elif [ "${PACKAGE}" == "fuse" ];then
            NAMES="fuse-devel"
        elif [ "${PACKAGE}" == "libnm" ];then
            NAMES="NetworkManager-libnm-devel"
        elif [ "${PACKAGE}" == "mpi" ];then
            NAMES="mpich-3.2-devel"
        elif [ "${PACKAGE}" == "lz4" ];then
            NAMES="lz4-devel"
        elif [ "${PACKAGE}" == "zlib" ];then
            NAMES="zlib-devel"
        fi
    }
    fi 

    #
    if [ ${METHOD} -eq 1 ];then
        STATUS=$(CheckHavePackageFromKit ${KIT_NAME} "${NAMES}")
    elif [ ${METHOD} -eq 2 ];then
        STATUS=$(CheckHavePackageFromWhich ${KIT_NAME} "${NAMES}")
    fi

	#
	echo "${STATUS}"
}

#
GetDependFlags()
# $1 KIT_NAME
# $2 PACKAGE
{
    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{   
        if [ "${PACKAGE}" == "openmp" ];then
            echo "-fopenmp"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            echo ""
        elif [ "${PACKAGE}" == "sqlite" ];then
            echo "$(pkg-config --cflags sqlite3)"
        elif [ "${PACKAGE}" == "openssl" ];then
            echo "$(pkg-config --cflags openssl)"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
        elif [ "${PACKAGE}" == "freeimage" ];then
            echo ""
        elif [ "${PACKAGE}" == "fuse" ];then
            echo "$(pkg-config --cflags fuse)"
        elif [ "${PACKAGE}" == "libnm" ];then
            echo "$(pkg-config --cflags libnm)"
        elif [ "${PACKAGE}" == "mpi" ];then
            echo "$(pkg-config --cflags mpi)"
        elif [ "${PACKAGE}" == "lz4" ];then
            echo "$(pkg-config --cflags liblz4)"
        elif [ "${PACKAGE}" == "zlib" ];then
            echo "$(pkg-config --cflags zlib)"
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PACKAGE}" == "openmp" ];then
            echo "-fopenmp"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            echo ""
        elif [ "${PACKAGE}" == "sqlite" ];then
            echo "$(pkg-config --cflags sqlite3)"
        elif [ "${PACKAGE}" == "openssl" ];then
            echo "$(pkg-config --cflags openssl)"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
        elif [ "${PACKAGE}" == "freeimage" ];then
            echo ""
        elif [ "${PACKAGE}" == "fuse" ];then
            echo "$(pkg-config --cflags fuse)"
        elif [ "${PACKAGE}" == "libnm" ];then
            echo "$(pkg-config --cflags libnm)"
        elif [ "${PACKAGE}" == "mpi" ];then
        {
            export PKG_CONFIG_PATH=/usr/lib64/mpich-3.2/lib/pkgconfig
            echo "$(pkg-config --cflags mpich)"
        }
        elif [ "${PACKAGE}" == "lz4" ];then
            echo "$(pkg-config --cflags liblz4)"
        elif [ "${PACKAGE}" == "zlib" ];then
            echo "$(pkg-config --cflags zlib)"
        fi
    }
    fi 
}


#
GetDependLibs()
# $1 KIT_NAME
# $2 PACKAGE
{
    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{   
        if [ "${PACKAGE}" == "openmp" ];then
            echo "-fopenmp"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            echo "-lodbc"
        elif [ "${PACKAGE}" == "sqlite" ];then
            echo "$(pkg-config --libs sqlite3)"
        elif [ "${PACKAGE}" == "openssl" ];then
            echo "$(pkg-config --libs openssl)"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
        elif [ "${PACKAGE}" == "freeimage" ];then
            echo "-lfreeimage"
        elif [ "${PACKAGE}" == "fuse" ];then
            echo "$(pkg-config --libs fuse)"
        elif [ "${PACKAGE}" == "libnm" ];then
            echo "$(pkg-config --libs libnm)"
        elif [ "${PACKAGE}" == "mpi" ];then
            echo "$(pkg-config --libs mpi)"
        elif [ "${PACKAGE}" == "lz4" ];then
            echo "$(pkg-config --libs liblz4)"
        elif [ "${PACKAGE}" == "zlib" ];then
            echo "$(pkg-config --libs zlib)"
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PACKAGE}" == "openmp" ];then
            echo "-fopenmp"
        elif [ "${PACKAGE}" == "unixodbc" ];then
            echo "-lodbc"
        elif [ "${PACKAGE}" == "sqlite" ];then
            echo "$(pkg-config --libs sqlite3)"
        elif [ "${PACKAGE}" == "openssl" ];then
            echo "$(pkg-config --libs openssl)"
        elif [ "${PACKAGE}" == "ffmpeg" ];then
            echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
        elif [ "${PACKAGE}" == "freeimage" ];then
            echo "-lfreeimage"
        elif [ "${PACKAGE}" == "fuse" ];then
            echo "$(pkg-config --libs fuse)"
        elif [ "${PACKAGE}" == "libnm" ];then
            echo "$(pkg-config --libs libnm)"
        elif [ "${PACKAGE}" == "mpi" ];then
        {
            export PKG_CONFIG_PATH=/usr/lib64/mpich-3.2/lib/pkgconfig
            echo "$(pkg-config --libs mpich)"
        }
        elif [ "${PACKAGE}" == "lz4" ];then
            echo "$(pkg-config --libs liblz4)"
        elif [ "${PACKAGE}" == "zlib" ];then
            echo "$(pkg-config --libs zlib)"
        fi
    }
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
VERSION_MINOR="1"
#发行版本
VERSION_RELEASE="20"

#
BUILD_TYPE="release"

#
INSTALL_PREFIX="/usr/local/"

#
DEPEND_FUNC="Nothing"

#
PrintUsage()
{
    echo "usage: [ OPTIONS ]"
    echo -e "\n\t-g"
    echo -e "\t\t生成调试符号。默认：关闭"
    echo -e "\n\t-V"
    echo -e "\t\t主版本。默认：${VERSION_MAJOR}"
    echo -e "\n\t-v"
    echo -e "\t\t副版本。默认：${VERSION_MINOR}"
    echo -e "\n\t-r"
    echo -e "\t\t发行版本。默认：${VERSION_RELEASE}"
    echo -e "\n\t-i < PATH >"
    echo -e "\t\t安装路径。默认：${INSTALL_PREFIX}"
    echo -e "\n\t-d < KEY,KEY,... >"
    echo -e "\t\t依赖项目。关键字：have-openmp,have-unixodbc,have-sqlite,have-openssl,have-ffmpeg,have-freeimage,have-fuse,have-libnm,have-mpi,have-lz4,have-zlib"
}

#
while getopts "?gV:v:r:i:d:" ARGKEY 
do
    case $ARGKEY in
    \?)
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
        DEPEND_FUNC="$OPTARG"
    ;;
    esac
done


#
STATUS=$(CheckHavePackage ${KIT_NAME} pkgconfig)
if [ ${STATUS} -ne 0 ];then
{
    echo "pkgconfig kit not found."
    exit 22
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-openmp") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} openmp)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_OPENMP="Yes"
        DEPEND_FLAGS=" -DHAVE_OPENMP $(GetDependFlags ${KIT_NAME} openmp) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} openmp) ${DEPEND_LIBS}"
    }
    else
    {
        echo "openmp kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-unixodbc") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} unixodbc)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_UNIXODBC="Yes"
        DEPEND_FLAGS=" -DHAVE_UNIXODBC $(GetDependFlags ${KIT_NAME} unixodbc) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} unixodbc) ${DEPEND_LIBS}"
    }
    else
    {
        echo "unixodbc kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-sqlite") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} sqlite)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_SQLITE="Yes"
        DEPEND_FLAGS=" -DHAVE_SQLITE $(GetDependFlags ${KIT_NAME} sqlite) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} sqlite) ${DEPEND_LIBS}"
    }
    else
    {
        echo "sqlite kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-openssl") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} openssl)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_OPENSSL="Yes"
        DEPEND_FLAGS=" -DHAVE_OPENSSL $(GetDependFlags ${KIT_NAME} openssl) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} openssl) ${DEPEND_LIBS}"
    }
    else
    {
        echo "openssl kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-ffmpeg") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} ffmpeg)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FFMPEG="Yes"
        DEPEND_FLAGS=" -DHAVE_FFMPEG $(GetDependFlags ${KIT_NAME} ffmpeg) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} ffmpeg) ${DEPEND_LIBS}"
    }
    else
    {
        echo "ffmpeg kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-freeimage") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} freeimage)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FREEIMAGE="Yes"
        DEPEND_FLAGS=" -DHAVE_FREEIMAGE $(GetDependFlags ${KIT_NAME} freeimage) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} freeimage) ${DEPEND_LIBS}"
    }
    else
    {
        echo "freeimage kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-fuse") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} fuse)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_FUSE="Yes"
        DEPEND_FLAGS=" -DHAVE_FUSE $(GetDependFlags ${KIT_NAME} fuse) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} fuse) ${DEPEND_LIBS}"
    }
    else
    {
        echo "fuse kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-libnm") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} libnm)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_LIBNM="Yes"
        DEPEND_FLAGS=" -DHAVE_LIBNM $(GetDependFlags ${KIT_NAME} libnm) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} libnm) ${DEPEND_LIBS}"
    }
    else
    {
        echo "libnm kit not found."
        exit 22
    }
    fi
}
fi

#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-mpi") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} mpi)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_MPI="Yes"
        DEPEND_FLAGS=" -DHAVE_MPI $(GetDependFlags ${KIT_NAME} mpi) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} mpi) ${DEPEND_LIBS}"
    }
    else
    {
        echo "mpich kit not found."
        exit 22
    }
    fi
}
fi


#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-lz4") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} lz4)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_LZ4="Yes"
        DEPEND_FLAGS=" -DHAVE_LZ4 $(GetDependFlags ${KIT_NAME} lz4) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} lz4) ${DEPEND_LIBS}"
    }
    else
    {
        echo "lz4 kit not found."
        exit 22
    }
    fi
}
fi


#
if [ $(CheckKeyword ${DEPEND_FUNC} "have-zlib") -eq 1 ];then
{
    STATUS=$(CheckHavePackage ${KIT_NAME} zlib)
    if [ ${STATUS} -eq 0 ];then
    {
        HAVE_ZLIB="Yes"
        DEPEND_FLAGS=" -DHAVE_ZLIB $(GetDependFlags ${KIT_NAME} zlib) ${DEPEND_FLAGS}"
        DEPEND_LIBS=" $(GetDependLibs ${KIT_NAME} zlib) ${DEPEND_LIBS}"
    }
    else
    {
        echo "zlib kit not found."
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