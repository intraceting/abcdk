#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
release-devel: release-devel-files release-devel-post release-devel-postun

#
release-devel-files:
	echo ${INSTALL_PREFIX}/lib/libabcdk.so >> ${BUILD_PATH}/release.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.a >> ${BUILD_PATH}/release.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc >> ${BUILD_PATH}/release.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk >> ${BUILD_PATH}/release.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk.h >> ${BUILD_PATH}/release.devel.files.txt

#
release-devel-post:
	echo "#abcdk-devel-post-begin" >> ${BUILD_PATH}/release.devel.post.txt
	echo "#abcdk-devel-post-end" >> ${BUILD_PATH}/release.devel.post.txt

#
release-devel-postun:
	echo "#abcdk-devel-postun-begin" >> ${BUILD_PATH}/release.devel.postun.txt
	echo "#abcdk-devel-postun-end" >> ${BUILD_PATH}/release.devel.postun.txt

#
release-runtime: release-runtime-files release-runtime-post release-runtime-postun

#
release-runtime-files:
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR} >> ${BUILD_PATH}/release.runtime.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}  >> ${BUILD_PATH}/release.runtime.files.txt
	echo ${INSTALL_PREFIX}/bin/abcdk-tool  >> ${BUILD_PATH}/release.runtime.files.txt
	echo ${INSTALL_PREFIX}/bin/abcdk-script  >> ${BUILD_PATH}/release.runtime.files.txt
	echo ${INSTALL_PREFIX}/share/abcdk  >> ${BUILD_PATH}/release.runtime.files.txt

#
release-runtime-post:
	echo "#abcdk-runtime-post-begin" >> ${BUILD_PATH}/release.runtime.post.txt
	echo "echo \"export PATH=\\\$${PATH}:${INSTALL_PREFIX}/bin\" > /etc/profile.d/abcdk.sh" >> ${BUILD_PATH}/release.runtime.post.txt
	echo "chmod 0755 /etc/profile.d/abcdk.sh"  >> ${BUILD_PATH}/release.runtime.post.txt
	echo "echo \"${INSTALL_PREFIX}/lib\" > /etc/ld.so.conf.d/abcdk.conf"  >> ${BUILD_PATH}/release.runtime.post.txt
	echo "ldconfig"  >> ${BUILD_PATH}/release.runtime.post.txt
	echo "#abcdk-runtime-post-end" >> ${BUILD_PATH}/release.runtime.post.txt

#
release-runtime-postun:
	echo "#abcdk-runtime-postun-begin" >> ${BUILD_PATH}/release.runtime.postun.txt
	echo "rm -f /etc/profile.d/abcdk.sh" >> ${BUILD_PATH}/release.runtime.postun.txt
	echo "rm -f /etc/ld.so.conf.d/abcdk.conf" >> ${BUILD_PATH}/release.runtime.postun.txt
	echo "ldconfig" >> ${BUILD_PATH}/release.postun.txt
	echo "#abcdk-runtime-postun-end" >> ${BUILD_PATH}/release.runtime.postun.txt
