#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#
#MAKEFILE_DIR := $(dir $(shell realpath "$(lastword $(MAKEFILE_LIST))"))

#生成PC文件内容.
define LIB_PKGCONFIG_CONTEXT
prefix=${INSTALL_PREFIX}
libdir=$${prefix}/lib
includedir=$${prefix}/include

Name: ABCDK
Version: ${VERSION_STR_FULL}
Description: ABCDK library
Requires:
Libs: -labcdk -L$${libdir}
Cflags: -I$${includedir}
endef
export LIB_PKGCONFIG_CONTEXT

#生成RT0文件内容.
define BIN_RT0_CONTEXT
#!/bin/bash
#
# This file is part of ABCDK.
#
# Automatically generated, do not modify.
#
SHELLNAME=$$(basename $${0})
SHELLDIR=$$(cd `dirname $${0}`; pwd)

#Export the necessary environment variables.
export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH}:$${SHELLDIR}:$${SHELLDIR}/../lib:$${SHELLDIR}/../lib/abcdk.compat

#Start the executable program.
$${0}.exe $$@
exit $$?
endef
export BIN_RT0_CONTEXT

#
install-bin:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/bin
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	chmod 0755 ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	printf "%s" "$${BIN_RT0_CONTEXT}" > ${INSTALL_PREFIX}/bin/abcdk-tool
	chmod 0755 ${INSTALL_PREFIX}/bin/abcdk-tool

#
uninstall-bin:
#
	rm -f ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	rm -f ${INSTALL_PREFIX}/bin/abcdk-tool

#
install-lib:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/lib
	cp -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PREFIX}/lib/
	chmod 0755 ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}
	ln -sf libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN}
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/locale/
	cp -f $(BUILD_PATH)/abcdk.pot ${INSTALL_PREFIX}/share/abcdk/locale/


#
uninstall-lib:
#
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN}
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}
#
	rm -f ${INSTALL_PREFIX}/share/abcdk/locale/abcdk.pot


#
install-dev:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/lib
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PREFIX}/lib/
	chmod 0755 ${INSTALL_PREFIX}/lib/libabcdk.a
	ln -sf libabcdk.so.${VERSION_STR_MAIN} ${INSTALL_PREFIX}/lib/libabcdk.so
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/lib/pkgconfig
	printf "%s" "$${LIB_PKGCONFIG_CONTEXT}" > ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc
	chmod 0644 ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/include
	cp -f $(MAKEFILE_DIR)/src/lib/include/abcdk.h ${INSTALL_PREFIX}/include/
	chmod 0644 ${INSTALL_PREFIX}/include/abcdk.h
	mkdir -p -m 0755 ${INSTALL_PREFIX}/include/abcdk
	cp -rfP $(MAKEFILE_DIR)/src/lib/include/abcdk/. ${INSTALL_PREFIX}/include/abcdk/
	find ${INSTALL_PREFIX}/include/abcdk -type d -exec chmod 0755 {} \;
	find ${INSTALL_PREFIX}/include/abcdk -type f -exec chmod 0644 {} \;
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/lib/
	cp -rfP $(MAKEFILE_DIR)/share/abcdk/lib/. ${INSTALL_PREFIX}/share/abcdk/lib/
	find ${INSTALL_PREFIX}/share/abcdk/lib -type d -exec chmod 0755 {} \;
	find ${INSTALL_PREFIX}/share/abcdk/lib -type f -exec chmod 0644 {} \;	

#
uninstall-dev:
#
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.a
	rm -f ${INSTALL_PREFIX}/include/abcdk.h
	rm -rf ${INSTALL_PREFIX}/include/abcdk
	rm -f ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc
	rm -rf ${INSTALL_PREFIX}/share/abcdk/lib

#
install-needed:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/lib/abcdk.compat
	${SHELLKITS_HOME}/tools/copy-3rdparty-needed.sh ${BUILD_PATH}/abcdk.needed ${INSTALL_PREFIX}/lib/abcdk.compat/
	${SHELLKITS_HOME}/tools/copy-compiler-needed.sh ${CC} ${INSTALL_PREFIX}/lib/abcdk.compat/

#
uninstall-needed:
#
	rm -rf ${INSTALL_PREFIX}/lib/abcdk.compat