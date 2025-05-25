#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#

#
ifneq (${KIT_NAME},rpm)
ifneq (${KIT_NAME},deb)
$(error Unsupported KIT_NAME: ${KIT_NAME}. Only 'rpm' or 'deb' are allowed.)
endif
endif

#打包用的临时目录。
PACK_TMP=${BUILD_PATH}/tmp/pack/

#临时的系统根路径。
RT_SYSROOT_PREFIX=${PACK_TMP}/sysroot.rt/
DEV_SYSROOT_PREFIX=${PACK_TMP}/sysroot.dev/
UTIL_SYSROOT_PREFIX=${PACK_TMP}/sysroot.util/

#
RT_PACKAGE_NAME=abcdk${PACKAGE_SUFFIX}-${VERSION_STR_FULL}-${TARGET_PLATFORM}
DEV_PACKAGE_NAME=abcdk${PACKAGE_SUFFIX}-dev-${VERSION_STR_FULL}-${TARGET_PLATFORM}
UTIL_PACKAGE_NAME=abcdk${PACKAGE_SUFFIX}-util-${VERSION_STR_FULL}-${TARGET_PLATFORM}

#生成安装后运行的脚本文件内容。
define RT_PACKAGE_POST_CONTEXT
#
echo "${INSTALL_PREFIX}/lib" > /etc/ld.so.conf.d/abcdk.conf
ldconfig
#
endef
export RT_PACKAGE_POST_CONTEXT

#生成卸载后运行的脚本文件内容。
define RT_PACKAGE_POSTUN_CONTEXT
#
rm -f /etc/ld.so.conf.d/abcdk.conf
ldconfig
#
endef
export RT_PACKAGE_POSTUN_CONTEXT


#
pack-rt-deb: pack-rt-prepare
#生成CTL文件。
	${DEV_TOOL_HOME}/make.deb.rt.ctl.sh  \
    	-d OUTPUT=${PACK_TMP}/${RT_PACKAGE_NAME}.deb.ctl \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="libc-bin, stapler (>= 5.97.1)"
#复制到临时的系统根路径。
	cp -rf ${PACK_TMP}/${RT_PACKAGE_NAME}.deb.ctl ${RT_SYSROOT_PREFIX}/DEBIAN
#创建软链接，因为dpkg-shlibdeps要使用debian/control文件。下同。
#	ln -s -f ${RT_SYSROOT_PREFIX}/DEBIAN ${RT_SYSROOT_PREFIX}/debian
#更新debian/control文件Pre-Depends字段。	
#	${DEV_TOOL_HOME}/dpkg-shlibdeps2control.sh "${RT_SYSROOT_PREFIX}"
#删除软链接，因为dpkg-deb会把这个当成普通文件复制。下同。
#	unlink ${RT_SYSROOT_PREFIX}/debian
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成DEB格式。
	dpkg-deb --build "${RT_SYSROOT_PREFIX}/" "${PACKAGE_PATH}/${RT_PACKAGE_NAME}.deb"

#
pack-rt-rpm: pack-rt-prepare
#生成SPEC文件。
	${DEV_TOOL_HOME}/make.rpm.rt.spec.sh \
    	-d OUTPUT=${PACK_TMP}/${RT_PACKAGE_NAME}.rpm.spec \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${RT_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="glibc, stapler >= 5.97-1"
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成RPM格式。
	rpmbuild --noclean --buildroot "${RT_SYSROOT_PREFIX}/" -bb ${PACK_TMP}/${RT_PACKAGE_NAME}.rpm.spec --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${RT_PACKAGE_NAME}.rpm"


#
pack-rt-prepare:
#创建不存在的路径。
	mkdir -p ${PACK_TMP}
#创建不存在的路径。
	mkdir -p ${RT_SYSROOT_PREFIX} 
#清理临时的系统路径。
	rm -rf ${RT_SYSROOT_PREFIX}/*
#安装需要的文件到临时的系统路径。
	$(MAKE) install-lib SYSROOT_PREFIX=${RT_SYSROOT_PREFIX}
#生成文件列表。
	find ${RT_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${PACK_TMP}/${RT_PACKAGE_NAME}.filelist.txt
	find ${RT_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${PACK_TMP}/${RT_PACKAGE_NAME}.filelist.txt
#生成安装后和卸载后运行的脚本。
	printf "%s" "$${RT_PACKAGE_POST_CONTEXT}" > ${PACK_TMP}/${RT_PACKAGE_NAME}.post.sh
	printf "%s" "$${RT_PACKAGE_POSTUN_CONTEXT}" > ${PACK_TMP}/${RT_PACKAGE_NAME}.postun.sh




#生成安装后运行的脚本文件内容。
define DEV_PACKAGE_POST_CONTEXT
#
endef
export DEV_PACKAGE_POST_CONTEXT

#生成卸载后运行的脚本文件内容。
define DEV_PACKAGE_POSTUN_CONTEXT
#
endef
export DEV_PACKAGE_POSTUN_CONTEXT


#
pack-dev-deb: pack-dev-prepare
#生成CTL文件。
	${DEV_TOOL_HOME}/make.deb.dev.ctl.sh  \
    	-d OUTPUT=${PACK_TMP}/${DEV_PACKAGE_NAME}.deb.ctl \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk-dev \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="abcdk (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE})"
#复制到临时的系统根路径。
	cp -rf ${PACK_TMP}/${DEV_PACKAGE_NAME}.deb.ctl ${DEV_SYSROOT_PREFIX}/DEBIAN
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成DEB格式。
	dpkg-deb --build "${DEV_SYSROOT_PREFIX}/" "${PACKAGE_PATH}/${DEV_PACKAGE_NAME}.deb"


#
pack-dev-rpm: pack-dev-prepare
#生成SPEC文件。
	${DEV_TOOL_HOME}/make.rpm.dev.spec.sh \
    	-d OUTPUT=${PACK_TMP}/${DEV_PACKAGE_NAME}.rpm.spec \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk-devel \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${DEV_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="abcdk = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_RELEASE}"
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成RPM格式。
	rpmbuild --noclean --buildroot "${DEV_SYSROOT_PREFIX}/" -bb ${PACK_TMP}/${DEV_PACKAGE_NAME}.rpm.spec --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${DEV_PACKAGE_NAME}.rpm"


#
pack-dev-prepare:
#创建不存在的路径。
	mkdir -p ${PACK_TMP}
#创建不存在的路径。
	mkdir -p ${DEV_SYSROOT_PREFIX} 
#清理临时的系统路径。
	rm -rf ${DEV_SYSROOT_PREFIX}/*
#安装需要的文件到临时的系统路径。
	$(MAKE) install-dev SYSROOT_PREFIX=${DEV_SYSROOT_PREFIX}
#生成文件列表。
	find ${DEV_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${PACK_TMP}/${DEV_PACKAGE_NAME}.filelist.txt
	find ${DEV_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${PACK_TMP}/${DEV_PACKAGE_NAME}.filelist.txt
#生成安装后和卸载后运行的脚本。
	printf "%s" "$${DEV_PACKAGE_POST_CONTEXT}" > ${PACK_TMP}/${DEV_PACKAGE_NAME}.post.sh
	printf "%s" "$${DEV_PACKAGE_POSTUN_CONTEXT}" > ${PACK_TMP}/${DEV_PACKAGE_NAME}.postun.sh


#生成安装后运行的脚本文件内容。
define UTIL_PACKAGE_POST_CONTEXT
#
echo "export PATH=\$${PATH}:${INSTALL_PREFIX}/bin" > /etc/profile.d/abcdk-util.sh
chmod 0755 /etc/profile.d/abcdk-util.sh
#
endef
export UTIL_PACKAGE_POST_CONTEXT

#生成卸载后运行的脚本文件内容。
define UTIL_PACKAGE_POSTUN_CONTEXT
#
rm -f /etc/profile.d/abcdk-util.sh
#
endef
export UTIL_PACKAGE_POSTUN_CONTEXT

#
pack-util-deb: pack-util-prepare
#生成CTL文件。
	${DEV_TOOL_HOME}/make.deb.rt.ctl.sh  \
    	-d OUTPUT=${PACK_TMP}/${UTIL_PACKAGE_NAME}.deb.ctl \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk-util \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="libc-bin, stapler (>= 5.97.1)"
#复制到临时的系统根路径。
	cp -rf ${PACK_TMP}/${UTIL_PACKAGE_NAME}.deb.ctl ${UTIL_SYSROOT_PREFIX}/DEBIAN
#创建软链接，因为dpkg-shlibdeps要使用debian/control文件。下同。
#	ln -s -f ${UTIL_SYSROOT_PREFIX}/DEBIAN ${UTIL_SYSROOT_PREFIX}/debian
#更新debian/control文件Pre-Depends字段。	
#	${DEV_TOOL_HOME}/dpkg-shlibdeps2control.sh "${UTIL_SYSROOT_PREFIX}"
#删除软链接，因为dpkg-deb会把这个当成普通文件复制。下同。
#	unlink ${UTIL_SYSROOT_PREFIX}/debian
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成DEB格式。
	dpkg-deb --build "${UTIL_SYSROOT_PREFIX}/" "${PACKAGE_PATH}/${UTIL_PACKAGE_NAME}.deb"

#
pack-util-rpm: pack-util-prepare
#生成SPEC文件。
	${DEV_TOOL_HOME}/make.rpm.rt.spec.sh \
    	-d OUTPUT=${PACK_TMP}/${UTIL_PACKAGE_NAME}.rpm.spec \
    	-d VENDOR_NAME=INTRACETING\(traceting@gmail.com\) \
    	-d PACK_NAME=abcdk-util \
		-d VERSION_MAJOR=${VERSION_MAJOR} \
		-d VERSION_MINOR=${VERSION_MINOR} \
		-d VERSION_RELEASE=${VERSION_RELEASE} \
		-d TARGET_PLATFORM=${TARGET_PLATFORM} \
		-d FILES_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.filelist.txt \
		-d POST_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.post.sh \
		-d POSTUN_NAME=${PACK_TMP}/${UTIL_PACKAGE_NAME}.postun.sh \
		-d REQUIRE_LIST="glibc, stapler >= 5.97-1"
#创建不存在的路径。
	mkdir -p ${PACKAGE_PATH}
#打包成RPM格式。
	rpmbuild --noclean --buildroot "${UTIL_SYSROOT_PREFIX}/" -bb ${PACK_TMP}/${UTIL_PACKAGE_NAME}.rpm.spec --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${UTIL_PACKAGE_NAME}.rpm"


#
pack-util-prepare:
#创建不存在的路径。
	mkdir -p ${PACK_TMP}
#创建不存在的路径。
	mkdir -p ${UTIL_SYSROOT_PREFIX} 
#清理临时的系统路径。
	rm -rf ${UTIL_SYSROOT_PREFIX}/*
#安装需要的文件到临时的系统路径。
	$(MAKE) install-tool install-script SYSROOT_PREFIX=${UTIL_SYSROOT_PREFIX}
#生成文件列表。
	find ${UTIL_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type f -printf "${INSTALL_PREFIX}/%P\n" > ${PACK_TMP}/${UTIL_PACKAGE_NAME}.filelist.txt
	find ${UTIL_SYSROOT_PREFIX}/${INSTALL_PREFIX} -type l -printf "${INSTALL_PREFIX}/%P\n" >> ${PACK_TMP}/${UTIL_PACKAGE_NAME}.filelist.txt
#生成安装后和卸载后运行的脚本。
	printf "%s" "$${UTIL_PACKAGE_POST_CONTEXT}" > ${PACK_TMP}/${UTIL_PACKAGE_NAME}.post.sh
	printf "%s" "$${UTIL_PACKAGE_POSTUN_CONTEXT}" > ${PACK_TMP}/${UTIL_PACKAGE_NAME}.postun.sh