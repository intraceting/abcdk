#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#


#占位预定义，实际会随机生成。
TMP_ROOT_PATH = /tmp/abcdk-build-installer.tmp
#
PACKAGE_PATH = ${BUILD_PACKAGE_PATH}/${VERSION_STR_MAIN}/
#
RUNTIME_PACKAGE_NAME=abcdk-${VERSION_STR_FULL}-${TARGET_PLATFORM}
#
DEVEL_PACKAGE_NAME=abcdk-devel-${VERSION_STR_FULL}-${TARGET_PLATFORM}

#
package-tar: package-runtime-tar package-devel-tar


#
package-runtime-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -cz -f "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/" "."
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -cz -f "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/" "."
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}


#
package-${KIT_NAME}: package-runtime-${KIT_NAME} package-devel-${KIT_NAME}

#
package-runtime-rpm:
#生成SPEC文件。
	$(CURDIR)/make.rpm.rt.spec.sh -d OUTPUT=${BUILD_PATH}/rpm_rt.spec \
		-d INSTALL_PREFIX=${INSTALL_PREFIX} \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_ARCH=${TARGET_ARCH}
# 
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --noclean --buildroot "${TMP_ROOT_PATH}/" -bb ${BUILD_PATH}/rpm_rt.spec --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${RUNTIME_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-rpm:
#生成SPEC文件。
	$(CURDIR)/make.rpm.dev.spec.sh -d OUTPUT=${BUILD_PATH}/rpm_dev.spec \
		-d INSTALL_PREFIX=${INSTALL_PREFIX} \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_ARCH=${TARGET_ARCH}
#
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --noclean --buildroot "${TMP_ROOT_PATH}/" -bb ${BUILD_PATH}/rpm_dev.spec --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${DEVEL_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

#
package-runtime-deb:
#生成CTL文件。
	$(CURDIR)/make.deb.rt.ctl.sh -d OUTPUT=${BUILD_PATH}/deb_rt.ctl \
		-d INSTALL_PREFIX=${INSTALL_PREFIX} \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_ARCH=${TARGET_ARCH}
#
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	cp -rf ${BUILD_PATH}/deb_rt.ctl ${TMP_ROOT_PATH}/DEBIAN
#	创建软链接，因为dpkg-shlibdeps要使用debian/control文件。下同。
	ln -s -f ${TMP_ROOT_PATH}/DEBIAN ${TMP_ROOT_PATH}/debian
#   更新debian/control文件Pre-Depends字段。	
	${DEB_TOOL_ROOT}/dpkg-shlibdeps2control.sh "${TMP_ROOT_PATH}"
#	删除软链接，因为dpkg-deb会把这个当成普通文件复制。下同。
	unlink ${TMP_ROOT_PATH}/debian
	mkdir -p ${PACKAGE_PATH}
	dpkg-deb --build "${TMP_ROOT_PATH}/" "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.deb"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

package-devel-deb:
#生成CTL文件。
	$(CURDIR)/make.deb.dev.ctl.sh -d OUTPUT=${BUILD_PATH}/deb_dev.ctl \
		-d INSTALL_PREFIX=${INSTALL_PREFIX} \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_ARCH=${TARGET_ARCH}
#
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	cp -rf ${BUILD_PATH}/deb_dev.ctl ${TMP_ROOT_PATH}/DEBIAN
	mkdir -p ${PACKAGE_PATH}
	dpkg-deb --build "${TMP_ROOT_PATH}/" "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.deb"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

