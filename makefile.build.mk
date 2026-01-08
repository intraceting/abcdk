#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#
#MAKEFILE_DIR := $(dir $(shell realpath "$(lastword $(MAKEFILE_LIST))"))

#生成LIB安装后执行脚本文件内容.
define LIB_POST_SHELL_CONTEXT
#
endef
export LIB_POST_SHELL_CONTEXT

#生成LIB卸载后执行脚本文件内容.
define LIB_POSTUN_SHELL_CONTEXT
#
endef
export LIB_POSTUN_SHELL_CONTEXT

#生成DEV安装后执行脚本文件内容.
define DEV_POST_SHELL_CONTEXT
#
endef
export DEV_POST_SHELL_CONTEXT

#生成DEV卸载后执行脚本文件内容.
define DEV_POSTUN_SHELL_CONTEXT
#
endef
export DEV_POSTUN_SHELL_CONTEXT

#生成BIN安装后执行脚本文件内容.
define BIN_POST_SHELL_CONTEXT
#
echo "export PATH=${INSTALL_PREFIX}/bin:\$${PATH}" > /etc/profile.d/abcdk-bin.sh
chmod 0755 /etc/profile.d/abcdk-bin.sh
echo "Run 'source /etc/profile' to update PATH, or restart the system for the change to take effect."
#
endef
export BIN_POST_SHELL_CONTEXT

#生成TOOL卸载后执行脚本文件内容.
define BIN_POSTUN_SHELL_CONTEXT
#
rm -f /etc/profile.d/abcdk-bin.sh
#
endef
export BIN_POSTUN_SHELL_CONTEXT

#
SYSROOT_TMP = ${BUILD_PATH}/abcdk.sysroot.tmp/

#
LIB_SYSROOT_TMP = ${SYSROOT_TMP}/lib.sysroot.tmp/
DEV_SYSROOT_TMP = ${SYSROOT_TMP}/dev.sysroot.tmp/
BIN_SYSROOT_TMP = ${SYSROOT_TMP}/bin.sysroot.tmp/
#
LIB_FILE_LIST = ${SYSROOT_TMP}/lib.file.list
DEV_FILE_LIST = ${SYSROOT_TMP}/dev.file.list
BIN_FILE_LIST = ${SYSROOT_TMP}/bin.file.list
#
LIB_POST_SHELL_FILE = ${SYSROOT_TMP}/lib.post.sh
LIB_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/lib.postun.sh
DEV_POST_SHELL_FILE = ${SYSROOT_TMP}/dev.post.sh
DEV_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/dev.postun.sh
BIN_POST_SHELL_FILE = ${SYSROOT_TMP}/bin.post.sh
BIN_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/bin.postun.sh

#
LIB_RPM_SPEC = ${SYSROOT_TMP}/lib.rpm.spec
LIB_DEB_SPEC = ${SYSROOT_TMP}/lib.deb.spec
DEV_RPM_SPEC = ${SYSROOT_TMP}/dev.rpm.spec
DEV_DEB_SPEC = ${SYSROOT_TMP}/dev.deb.spec
BIN_RPM_SPEC = ${SYSROOT_TMP}/bin.rpm.spec
BIN_DEB_SPEC = ${SYSROOT_TMP}/bin.deb.spec 

#
ifeq (${INSTALL_NEEDED},yes)
LIB_RPM_REQUIRE_LIST = "glibc"
LIB_DEB_REQUIRE_LIST = "libc-bin"
DEV_RPM_REQUIRE_LIST = "abcdk-lib = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
DEV_DEB_REQUIRE_LIST = "abcdk-lib (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
BIN_RPM_REQUIRE_LIST = "abcdk-lib = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
BIN_DEB_REQUIRE_LIST = "abcdk-lib (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
else
LIB_RPM_REQUIRE_LIST = "glibc"
LIB_DEB_REQUIRE_LIST = "libc-bin"
DEV_RPM_REQUIRE_LIST = "abcdk-lib = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
DEV_DEB_REQUIRE_LIST = "abcdk-lib (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
BIN_RPM_REQUIRE_LIST = "abcdk-lib = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
BIN_DEB_REQUIRE_LIST = "abcdk-lib (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
endif

#
prepare: prepare-lib prepare-dev prepare-bin

#
prepare-lib:
	rm -rf ${LIB_SYSROOT_TMP}
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${LIB_SYSROOT_TMP}/${INSTALL_PREFIX} install-lib
ifeq (${INSTALL_NEEDED},yes)
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${LIB_SYSROOT_TMP}/${INSTALL_PREFIX} install-needed
endif
	find ${LIB_SYSROOT_TMP}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${LIB_FILE_LIST}
	find ${LIB_SYSROOT_TMP}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${LIB_FILE_LIST}
	printf "%s" "$${LIB_POST_SHELL_CONTEXT}" > ${LIB_POST_SHELL_FILE}
	printf "%s" "$${LIB_POSTUN_SHELL_CONTEXT}" > ${LIB_POSTUN_SHELL_FILE}

#
prepare-dev:
	rm -rf ${DEV_SYSROOT_TMP}
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${DEV_SYSROOT_TMP}/${INSTALL_PREFIX} install-dev
	find ${DEV_SYSROOT_TMP}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${DEV_FILE_LIST}
	find ${DEV_SYSROOT_TMP}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${DEV_FILE_LIST}
#替换PC文件内部的路径为安装路径。
	find ${DEV_SYSROOT_TMP}/${INSTALL_PREFIX} -type f -name "*.pc" -exec sed -i "s#${DEV_SYSROOT_TMP}/${INSTALL_PREFIX}#${INSTALL_PREFIX}#g" {} \;
	printf "%s" "$${DEV_POST_SHELL_CONTEXT}" > ${DEV_POST_SHELL_FILE}
	printf "%s" "$${DEV_POSTUN_SHELL_CONTEXT}" > ${DEV_POSTUN_SHELL_FILE}
	
#
prepare-bin:
	rm -rf ${BIN_SYSROOT_TMP}
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${BIN_SYSROOT_TMP}/${INSTALL_PREFIX} install-bin
	find ${BIN_SYSROOT_TMP}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${BIN_FILE_LIST}
	find ${BIN_SYSROOT_TMP}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${BIN_FILE_LIST}
	printf "%s" "$${BIN_POST_SHELL_CONTEXT}" > ${BIN_POST_SHELL_FILE}
	printf "%s" "$${BIN_POSTUN_SHELL_CONTEXT}" > ${BIN_POSTUN_SHELL_FILE}


#
build-deb-lib: prepare-lib
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.deb.rt.ctl.sh  \
	-d PACK_NAME=abcdk-lib \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${LIB_DEB_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${LIB_FILE_LIST} \
	-d POST_NAME=${LIB_POST_SHELL_FILE} \
	-d POSTUN_NAME=${LIB_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${LIB_DEB_REQUIRE_LIST}
#移动SPEC文件.
	mv ${LIB_DEB_SPEC} ${LIB_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${LIB_SYSROOT_TMP} ${BUILD_PATH}/abcdk-lib-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
#移动SPEC文件.
	mv ${LIB_SYSROOT_TMP}/DEBIAN ${LIB_DEB_SPEC}

#
build-deb-dev: prepare-dev
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.deb.dev.ctl.sh  \
	-d PACK_NAME=abcdk-dev \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${DEV_DEB_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${DEV_FILE_LIST} \
	-d POST_NAME=${DEV_POST_SHELL_FILE} \
	-d POSTUN_NAME=${DEV_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${DEV_DEB_REQUIRE_LIST}
#移动SPEC文件.
	mv ${DEV_DEB_SPEC} ${DEV_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${DEV_SYSROOT_TMP} ${BUILD_PATH}/abcdk-dev-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
#移动SPEC文件.
	mv ${DEV_SYSROOT_TMP}/DEBIAN ${DEV_DEB_SPEC}

#
build-deb-bin: prepare-bin
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.deb.rt.ctl.sh  \
	-d PACK_NAME=abcdk-bin \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${BIN_DEB_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${BIN_FILE_LIST} \
	-d POST_NAME=${BIN_POST_SHELL_FILE} \
	-d POSTUN_NAME=${BIN_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${BIN_DEB_REQUIRE_LIST}
#移动SPEC文件.
	mv ${BIN_DEB_SPEC} ${BIN_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${BIN_SYSROOT_TMP} ${BUILD_PATH}/abcdk-bin-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
#移动SPEC文件.
	mv ${BIN_SYSROOT_TMP}/DEBIAN ${BIN_DEB_SPEC}

#
build-deb: build-deb-lib build-deb-dev build-deb-bin

#
build-rpm-lib: prepare-lib
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.rt.spec.sh \
	-d PACK_NAME=abcdk-lib \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${LIB_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${LIB_FILE_LIST} \
	-d POST_NAME=${LIB_POST_SHELL_FILE} \
	-d POSTUN_NAME=${LIB_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${LIB_RPM_REQUIRE_LIST}
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${LIB_SYSROOT_TMP} \
	-bb ${LIB_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-lib-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"

#
build-rpm-dev: prepare-dev
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.dev.spec.sh \
	-d PACK_NAME=abcdk-dev \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${DEV_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${DEV_FILE_LIST} \
	-d POST_NAME=${DEV_POST_SHELL_FILE} \
	-d POSTUN_NAME=${DEV_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${DEV_RPM_REQUIRE_LIST}
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${DEV_SYSROOT_TMP} \
	-bb ${DEV_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-dev-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"

#
build-rpm-bin: prepare-bin
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.rt.spec.sh \
	-d PACK_NAME=abcdk-bin \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${BIN_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${BIN_FILE_LIST} \
	-d POST_NAME=${BIN_POST_SHELL_FILE} \
	-d POSTUN_NAME=${BIN_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST=${BIN_RPM_REQUIRE_LIST}
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${BIN_SYSROOT_TMP} \
	-bb ${BIN_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-bin-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"


#
build-rpm: build-rpm-lib build-rpm-dev build-rpm-bin

#
clean-build-lib:
	rm -rf ${LIB_SYSROOT_TMP}
	rm -rf ${LIB_FILE_LIST}
	rm -rf ${LIB_POST_SHELL_FILE}
	rm -rf ${LIB_POSTUN_SHELL_FILE}
	rm -rf ${LIB_RPM_SPEC}
	rm -rf ${LIB_DEB_SPEC}
#
clean-build-dev:
	rm -rf ${DEV_SYSROOT_TMP}
	rm -rf ${DEV_FILE_LIST}
	rm -rf ${DEV_POST_SHELL_FILE}
	rm -rf ${DEV_POSTUN_SHELL_FILE}
	rm -rf ${DEV_RPM_SPEC}
	rm -rf ${DEV_DEB_SPEC}

#
clean-build-bin:
	rm -rf ${BIN_SYSROOT_TMP}
	rm -rf ${BIN_FILE_LIST}
	rm -rf ${BIN_POST_SHELL_FILE}
	rm -rf ${BIN_POSTUN_SHELL_FILE}
	rm -rf ${BIN_RPM_SPEC}
	rm -rf ${BIN_DEB_SPEC}
