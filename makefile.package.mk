#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
package-devel: package-devel-files package-devel-post package-devel-postun

#
package-devel-files:
	echo ${INSTALL_PREFIX}/lib/libabcdk.so >> ${BUILD_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.a >> ${BUILD_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc >> ${BUILD_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk >> ${BUILD_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk.h >> ${BUILD_PATH}/package.devel.files.txt

#
package-devel-post:
	echo "#abcdk-devel-post-begin" >> ${BUILD_PATH}/package.devel.post.txt
	echo "#abcdk-devel-post-end" >> ${BUILD_PATH}/package.devel.post.txt

#
package-devel-postun:
	echo "#abcdk-devel-postun-begin" >> ${BUILD_PATH}/package.devel.postun.txt
	echo "#abcdk-devel-postun-end" >> ${BUILD_PATH}/package.devel.postun.txt

#
package-runtime: package-runtime-files package-runtime-post package-runtime-postun

#
package-runtime-files:
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR} >> ${BUILD_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}  >> ${BUILD_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/bin/abcdk-tool  >> ${BUILD_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/bin/abcdk-script  >> ${BUILD_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/share/abcdk  >> ${BUILD_PATH}/package.runtime.files.txt

#
package-runtime-post:
	echo "#abcdk-runtime-post-begin" >> ${BUILD_PATH}/package.runtime.post.txt
	echo "echo \"export PATH=\\\$${PATH}:${INSTALL_PREFIX}/bin\" > /etc/profile.d/abcdk.sh" >> ${BUILD_PATH}/package.runtime.post.txt
	echo "chmod 0755 /etc/profile.d/abcdk.sh"  >> ${BUILD_PATH}/package.runtime.post.txt
	echo "echo \"${INSTALL_PREFIX}/lib\" > /etc/ld.so.conf.d/abcdk.conf"  >> ${BUILD_PATH}/package.runtime.post.txt
	echo "ldconfig"  >> ${BUILD_PATH}/package.runtime.post.txt
	echo "#abcdk-runtime-post-end" >> ${BUILD_PATH}/package.runtime.post.txt

#
package-runtime-postun:
	echo "#abcdk-runtime-postun-begin" >> ${BUILD_PATH}/package.runtime.postun.txt
	echo "rm -f /etc/profile.d/abcdk.sh" >> ${BUILD_PATH}/package.runtime.postun.txt
	echo "rm -f /etc/ld.so.conf.d/abcdk.conf" >> ${BUILD_PATH}/package.runtime.postun.txt
	echo "ldconfig" >> ${BUILD_PATH}/package.postun.txt
	echo "#abcdk-runtime-postun-end" >> ${BUILD_PATH}/package.runtime.postun.txt
