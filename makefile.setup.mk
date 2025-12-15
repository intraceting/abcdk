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

#
install-tool:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/bin
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	chmod 0755 ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	cp -f ${SHELLKITS_HOME}/tools/rt0.sh ${INSTALL_PREFIX}/bin/abcdk-tool
	chmod 0755 ${INSTALL_PREFIX}/bin/abcdk-tool
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/sample/abcdk-tool/
	cp -rfP $(MAKEFILE_DIR)/share/abcdk/sample/abcdk-tool/. ${INSTALL_PREFIX}/share/abcdk/sample/abcdk-tool/
	find ${INSTALL_PREFIX}/share/abcdk/sample/abcdk-tool -type d -exec chmod 0755 {} \;
	find ${INSTALL_PREFIX}/share/abcdk/sample/abcdk-tool -type f -exec chmod 0644 {} \;
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/locale/
	cp -f $(BUILD_PATH)/abcdk-tool.pot ${INSTALL_PREFIX}/share/abcdk/locale/
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES
	cp -f $(MAKEFILE_DIR)/share/locale/en_US/LC_MESSAGES/abcdk-tool.mo ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/
	chmod 0644 ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/abcdk-tool.mo

#
install-lib:
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/lib
	cp -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PREFIX}/lib/
	chmod 0755 ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}
	ln -sf libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN}
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/locale/
	cp -f $(BUILD_PATH)/libabcdk.pot ${INSTALL_PREFIX}/share/abcdk/locale/
#
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES
	cp -f $(MAKEFILE_DIR)/share/locale/en_US/LC_MESSAGES/libabcdk.mo ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/
	chmod 0644 ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/libabcdk.mo

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
	mkdir -p -m 0755 ${INSTALL_PREFIX}/share/abcdk/protocol/libabcdk/
	cp -rfP $(MAKEFILE_DIR)/share/abcdk/protocol/libabcdk/. ${INSTALL_PREFIX}/share/abcdk/protocol/libabcdk/
	find ${INSTALL_PREFIX}/share/abcdk/protocol/libabcdk -type d -exec chmod 0755 {} \;
	find ${INSTALL_PREFIX}/share/abcdk/protocol/libabcdk -type f -exec chmod 0644 {} \;	

#
uninstall-tool:
#
	rm -f ${INSTALL_PREFIX}/bin/abcdk-tool.exe
	rm -f ${INSTALL_PREFIX}/bin/abcdk-tool
	rm -f ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/abcdk-tool.mo
	rm -f ${INSTALL_PREFIX}/share/abcdk/locale/abcdk-tool.pot
	rm -rf ${INSTALL_PREFIX}/share/abcdk/sample/abcdk-tool

#
uninstall-lib:
#
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN}
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}
	rm -f ${INSTALL_PREFIX}/share/locale/en_US/LC_MESSAGES/libabcdk.mo
	rm -f ${INSTALL_PREFIX}/share/abcdk/locale/libabcdk.pot
	
#
uninstall-dev:
#
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.so
	rm -f ${INSTALL_PREFIX}/lib/libabcdk.a
	rm -f ${INSTALL_PREFIX}/include/abcdk.h
	rm -rf ${INSTALL_PREFIX}/include/abcdk
	rm -f ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc
	rm -rf ${INSTALL_PREFIX}/share/abcdk/protocol/libabcdk

	

