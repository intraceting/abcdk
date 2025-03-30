#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
STATUS=$(${SHELLDIR}/check-package.sh dpkg-dev; echo $?)
if [ ${STATUS} -ne 0 ];then
    exit 1
fi

#
ROOT_PATH=$(realpath $1)
EXE_FILES="";

#遍历项目的目录，找so和exe文件。
for FILE in `find ${ROOT_PATH}/ -type f`
do
	CHK=$(ldd "${FILE}" >>/dev/null 2>&1 ; echo $?)
	if [ ${CHK} -eq 0 ];then
		EXE_FILES="${FILE} ${EXE_FILES}"
	fi
done

#
#DEPENDS=$(cd ${ROOT_PATH};dpkg-shlibdeps -e ${EXE_FILES} -O 2>/dev/null)
DEPENDS=$(cd ${ROOT_PATH};dpkg-shlibdeps --ignore-missing-info -e ${EXE_FILES} -O 2>/dev/null)

#替换shlibs:Depends=为空。
#例：${字符串变量/待查找的字符串/替换字符串(允许无)}
DEPENDS=$(echo ${DEPENDS/shlibs:Depends=/})

#
TMPFILE=$(mktemp ${ROOT_PATH}/debian/control.XXXXXX)

#按行读取，替换变量${shlibs:Depends}，同时保留格式。
IFS_OLD=$IFS
IFS=''
while read LINE
do
echo ${LINE//\$\{shlibs\:Depends\}/${DEPENDS}} >> ${TMPFILE}
done < ${ROOT_PATH}/debian/control
IFS=$IFS

#
mv -f ${TMPFILE} ${ROOT_PATH}/debian/control

exit 0