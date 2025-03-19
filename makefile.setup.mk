#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
INSTALL_PATH=${INSTALL_PREFIX}
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)
INSTALL_PATH_DOC = $(abspath ${INSTALL_PATH}/share/)


#
install-tool:
#
	mkdir -p ${INSTALL_PATH_BIN}/
	mkdir -p ${INSTALL_PATH_DOC}/abcdk/tool/
#
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}/
	cp -rf $(CURDIR)/share/tool/. ${INSTALL_PATH_DOC}/abcdk/tool/
#
	chmod 0755 ${INSTALL_PATH_BIN}/abcdk-tool
	find ${INSTALL_PATH_DOC}/abcdk/tool/ -type f -exec chmod 0644 {} \;

#
install-script:
#
	mkdir -p ${INSTALL_PATH_BIN}/abcdk-script/
	mkdir -p ${INSTALL_PATH_DOC}/abcdk/script/
#
	cp -rf $(CURDIR)/src/script/. ${INSTALL_PATH_BIN}/abcdk-script/
	cp -rf $(CURDIR)/share/script/. ${INSTALL_PATH_DOC}/abcdk/script/
#
	find ${INSTALL_PATH_BIN}/abcdk-script/ -type f -name *.sh -exec chmod 0755 {} \;


#
install-lib:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_DOC}/abcdk/lib/
#
	cp -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PATH_LIB}/
	cp -rf $(CURDIR)/share/lib/. ${INSTALL_PATH_DOC}/abcdk/lib/
#	
	chmod 0755 ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_FULL}
	cd ${INSTALL_PATH_LIB} ; ln -sf libabcdk.so.${VERSION_STR_FULL} libabcdk.so.${VERSION_STR_MAIN} ;
#
	find ${INSTALL_PATH_DOC}/abcdk/lib/ -type f -exec chmod 0644 {} \;

#
install-dev:
#
	mkdir -p ${INSTALL_PATH_LIB}/pkgconfig/
	mkdir -p ${INSTALL_PATH_INC}
#
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
	cp  -rf $(CURDIR)/src/lib/include/abcdk ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/src/lib/include/abcdk.h ${INSTALL_PATH_INC}/
	
#生成PC文件。
	echo "prefix=${INSTALL_PREFIX}" 		> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "libdir=\$${prefix}/lib" 			>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "includedir=\$${prefix}/include" 	>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "" 								>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Name: ABCDK" 						>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Version: ${VERSION_STR_FULL}" 	>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Description: ABCDK library" 		>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Requires:" 						>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Libs: -labcdk -L\$${libdir}" 		>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
	echo "Cflags: -I\$${includedir}" 		>> ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc

#
	chmod 0755 ${INSTALL_PATH_LIB}/libabcdk.a
	cd ${INSTALL_PATH_LIB} ; ln -sf libabcdk.so.${VERSION_STR_MAIN} libabcdk.so ;
	find ${INSTALL_PATH_INC}/abcdk/ -type f -exec chmod 0644 {} \;
	chmod 0644 ${INSTALL_PATH_INC}/abcdk.h
	chmod 0644 ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc

#
uninstall-tool:
#
	rm -f ${INSTALL_PATH_BIN}/abcdk-tool
	rm -rf $(INSTALL_PATH_DOC)/abcdk/tool


uninstall-script:
#
	rm -rf ${INSTALL_PATH_BIN}/abcdk-script
	rm -rf $(INSTALL_PATH_DOC)/abcdk/script

#
uninstall-lib:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_MAIN}
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_FULL}
	rm -rf $(INSTALL_PATH_DOC)/abcdk/lib
	
#
uninstall-dev:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
	rm -rf ${INSTALL_PATH_INC}/abcdk
	rm -f ${INSTALL_PATH_INC}/abcdk.h
	rm -f ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc

	

