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

#生成TOOL安装后执行脚本文件内容.
define TOOL_POST_SHELL_CONTEXT
#
echo "export PATH=${INSTALL_PREFIX}/bin:\$${PATH}" > /etc/profile.d/abcdk-tool.sh
chmod 0755 /etc/profile.d/abcdk-tool.sh
echo "Run 'source /etc/profile' to update PATH, or restart the system for the change to take effect."
#
endef
export TOOL_POST_SHELL_CONTEXT

#生成TOOL卸载后执行脚本文件内容.
define TOOL_POSTUN_SHELL_CONTEXT
#
rm -f /etc/profile.d/abcdk-tool.sh
#
endef
export TOOL_POSTUN_SHELL_CONTEXT

#
SYSROOT_TMP = ${BUILD_PATH}/abcdk.sysroot.tmp/

#
LIB_SYSROOT_TMP = ${SYSROOT_TMP}/lib.sysroot.tmp/
DEV_SYSROOT_TMP = ${SYSROOT_TMP}/dev.sysroot.tmp/
TOOL_SYSROOT_TMP = ${SYSROOT_TMP}/tool.sysroot.tmp/
#
LIB_FILE_LIST = ${SYSROOT_TMP}/lib.file.list
DEV_FILE_LIST = ${SYSROOT_TMP}/dev.file.list
TOOL_FILE_LIST = ${SYSROOT_TMP}/tool.file.list
#
LIB_POST_SHELL_FILE = ${SYSROOT_TMP}/lib.post.sh
LIB_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/lib.postun.sh
DEV_POST_SHELL_FILE = ${SYSROOT_TMP}/dev.post.sh
DEV_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/dev.postun.sh
TOOL_POST_SHELL_FILE = ${SYSROOT_TMP}/tool.post.sh
TOOL_POSTUN_SHELL_FILE = ${SYSROOT_TMP}/tool.postun.sh

#
LIB_RPM_SPEC = ${SYSROOT_TMP}/lib.rpm.spec
LIB_DEB_SPEC =  ${SYSROOT_TMP}/lib.deb.spec
DEV_RPM_SPEC = ${SYSROOT_TMP}/dev.rpm.spec
DEV_DEB_SPEC = ${SYSROOT_TMP}/dev.deb.spec
TOOL_RPM_SPEC = ${SYSROOT_TMP}/tool.rpm.spec
TOOL_DEB_SPEC = ${SYSROOT_TMP}/tool.deb.spec 

#
prepare: prepare-lib prepare-dev prepare-tool

#
prepare-lib:
	rm -rf ${LIB_SYSROOT_TMP}
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${LIB_SYSROOT_TMP}/${INSTALL_PREFIX} install-lib
	${SHELLKITS_HOME}/tools/copy-3rdparty-needed.sh ${BUILD_PATH}/abcdk.needed ${LIB_SYSROOT_TMP}/${INSTALL_PREFIX}/lib/
	${SHELLKITS_HOME}/tools/copy-compiler-needed.sh ${CC} ${LIB_SYSROOT_TMP}/${INSTALL_PREFIX}/lib/compat/
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
prepare-tool:
	rm -rf ${TOOL_SYSROOT_TMP}
	$(MAKE) -s -C ${MAKEFILE_DIR} INSTALL_PREFIX=${TOOL_SYSROOT_TMP}/${INSTALL_PREFIX} install-tool
	find ${TOOL_SYSROOT_TMP}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${TOOL_FILE_LIST}
	find ${TOOL_SYSROOT_TMP}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${TOOL_FILE_LIST}
	printf "%s" "$${TOOL_POST_SHELL_CONTEXT}" > ${TOOL_POST_SHELL_FILE}
	printf "%s" "$${TOOL_POSTUN_SHELL_CONTEXT}" > ${TOOL_POSTUN_SHELL_FILE}


#
build-deb-lib: prepare-lib
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.deb.rt.ctl.sh  \
	-d PACK_NAME=abcdk \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${LIB_DEB_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${LIB_FILE_LIST} \
	-d POST_NAME=${LIB_POST_SHELL_FILE} \
	-d POSTUN_NAME=${LIB_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST="libc-bin"
#移动SPEC文件.
	mv ${LIB_DEB_SPEC} ${LIB_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${LIB_SYSROOT_TMP} ${BUILD_PATH}/abcdk-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
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
	-d REQUIRE_LIST="abcdk (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
#移动SPEC文件.
	mv ${DEV_DEB_SPEC} ${DEV_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${DEV_SYSROOT_TMP} ${BUILD_PATH}/abcdk-dev-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
#移动SPEC文件.
	mv ${DEV_SYSROOT_TMP}/DEBIAN ${DEV_DEB_SPEC}

#
build-deb-tool: prepare-tool
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.deb.rt.ctl.sh  \
	-d PACK_NAME=abcdk-tool \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${TOOL_DEB_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${TOOL_FILE_LIST} \
	-d POST_NAME=${TOOL_POST_SHELL_FILE} \
	-d POSTUN_NAME=${TOOL_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST="abcdk (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})"
#移动SPEC文件.
	mv ${TOOL_DEB_SPEC} ${TOOL_SYSROOT_TMP}/DEBIAN
#生成DEB文件.
	dpkg-deb --build ${TOOL_SYSROOT_TMP} ${BUILD_PATH}/abcdk-tool-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.deb
#移动SPEC文件.
	mv ${TOOL_SYSROOT_TMP}/DEBIAN ${TOOL_DEB_SPEC}

#
build-deb: build-deb-lib build-deb-dev build-deb-tool

#
build-rpm-lib: prepare-lib
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.rt.spec.sh \
	-d PACK_NAME=abcdk \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${LIB_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${LIB_FILE_LIST} \
	-d POST_NAME=${LIB_POST_SHELL_FILE} \
	-d POSTUN_NAME=${LIB_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST="glibc"
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${LIB_SYSROOT_TMP} \
	-bb ${LIB_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"

#
build-rpm-dev: prepare-dev
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.dev.spec.sh \
	-d PACK_NAME=abcdk-devel \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${DEV_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${DEV_FILE_LIST} \
	-d POST_NAME=${DEV_POST_SHELL_FILE} \
	-d POSTUN_NAME=${DEV_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST="abcdk = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${DEV_SYSROOT_TMP} \
	-bb ${DEV_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-devel-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"

#
build-rpm-tool: prepare-tool
#生成SPEC文件.
	${SHELLKITS_HOME}/tools/make.rpm.rt.spec.sh \
	-d PACK_NAME=abcdk-tool \
	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
	-d OUTPUT=${TOOL_RPM_SPEC} \
	-d VERSION_MAJOR=${VERSION_MAJOR} \
	-d VERSION_MINOR=${VERSION_MINOR} \
	-d VERSION_RELEASE=${VERSION_PATCH} \
	-d TARGET_PLATFORM=${TARGET_PLATFORM} \
	-d FILES_NAME=${TOOL_FILE_LIST} \
	-d POST_NAME=${TOOL_POST_SHELL_FILE} \
	-d POSTUN_NAME=${TOOL_POSTUN_SHELL_FILE} \
	-d REQUIRE_LIST="abcdk = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}"
#生成RPM文件.
	rpmbuild --noclean \
	--target=${TARGET_PLATFORM} \
	--buildroot ${TOOL_SYSROOT_TMP} \
	-bb ${TOOL_RPM_SPEC} \
	--define="_rpmdir ${BUILD_PATH}" \
	--define="_rpmfilename abcdk-tool-${VERSION_STR_FULL}-${TARGET_MULTIARCH}.rpm"


#
build-rpm: build-rpm-lib build-rpm-dev build-rpm-tool

#
clean-build-lib:
	rm -rf ${LIB_SYSROOT_TMP}

#
clean-build-dev:
	rm -rf ${DEV_SYSROOT_TMP}

#
clean-build-tool:
	rm -rf ${TOOL_SYSROOT_TMP}
