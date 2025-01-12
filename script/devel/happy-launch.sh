#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLNAME=$(basename $0)
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
{
    echo "usage: ${SHELLNAME} < project-path > "
    exit 22
}
fi

#
PJT_PATH=$(realpath $1)

#
PJT_BIN_PATH=""
PJT_LIB_PATH=""

#遍历项目的目录，查找so库所在目录。
for DIR in `find ${PJT_PATH}/ -type d`
do
	CHK=$(find ${DIR} -maxdepth 1 -type f -name "lib*.so" -o -name "lib*.so.*" |wc -l)
	if [ ${CHK} -ge 1 ];then
		PJT_LIB_PATH=${DIR}:${PJT_LIB_PATH}
	fi
done

#遍历项目的目录，查找具有执行权限的文件所在目录。
for DIR in `find ${PJT_PATH}/ -type d`
do
	CHK=$(find ${DIR} -maxdepth 1 -type f -executable |wc -l)
	if [ ${CHK} -ge 1 ];then
		PJT_BIN_PATH=${DIR}:${PJT_BIN_PATH}
	fi
done

#
cat <<EOF
#动态链接库(lib*.so.*,lib*.so)所在路径集合如下。添加到系统配置，或LD_LIBRARY_PATH环境变量，并使用生效。
${PJT_LIB_PATH}

#具有执行权限的文件(---x--x--x)所在路径集合如下。添加到系统配置，或PATH环境变量，并使用生效。
${PJT_BIN_PATH}
EOF
