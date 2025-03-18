#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
INSTALL_PATH=${ROOT_PATH}/${INSTALL_PREFIX}
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)
INSTALL_PATH_DOC = $(abspath ${INSTALL_PATH}/share/)

#
install-runtime:
#
	mkdir -p ${INSTALL_PATH_BIN}
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_DOC}/abcdk/
#
	cp -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}/
	cp -rf $(CURDIR)/share/abcdk/. ${INSTALL_PATH_DOC}/abcdk/

#	
	chmod 0755 ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_FULL}
	cd ${INSTALL_PATH_LIB} ; ln -sf libabcdk.so.${VERSION_STR_FULL} libabcdk.so.${VERSION_STR_MAIN} ;
	chmod 0755 ${INSTALL_PATH_BIN}/abcdk-tool
	find ${INSTALL_PATH_DOC}/abcdk/ -type f -exec chmod 0644 {} \;

#
install-runtime-package: install-runtime
#
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_MAIN} 	>> ${INSTALL_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.so.${VERSION_STR_FULL}  >> ${INSTALL_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/bin/abcdk-tool  						>> ${INSTALL_PATH}/package.runtime.files.txt
	echo ${INSTALL_PREFIX}/share/abcdk  						>> ${INSTALL_PATH}/package.runtime.files.txt
#
	echo "#abcdk-runtime-post-begin" 														>> ${INSTALL_PATH}/package.runtime.post.txt
	echo "echo \"export PATH=\\\$${PATH}:${INSTALL_PREFIX}/bin\" > /etc/profile.d/abcdk.sh" >> ${INSTALL_PATH}/package.runtime.post.txt
	echo "chmod 0755 /etc/profile.d/abcdk.sh"  												>> ${INSTALL_PATH}/package.runtime.post.txt
	echo "echo \"${INSTALL_PREFIX}/lib\" > /etc/ld.so.conf.d/abcdk.conf"  					>> ${INSTALL_PATH}/package.runtime.post.txt
	echo "ldconfig"  																		>> ${INSTALL_PATH}/package.runtime.post.txt
	echo "#abcdk-runtime-post-end" 															>> ${INSTALL_PATH}/package.runtime.post.txt
#
	echo "#abcdk-runtime-postun-begin" 				>> ${INSTALL_PATH}/package.runtime.postun.txt
	echo "rm -f /etc/profile.d/abcdk.sh" 			>> ${INSTALL_PATH}/package.runtime.postun.txt
	echo "rm -f /etc/ld.so.conf.d/abcdk.conf" 		>> ${INSTALL_PATH}/package.runtime.postun.txt
	echo "ldconfig" 								>> ${INSTALL_PATH}/package.runtime.postun.txt
	echo "#abcdk-runtime-postun-end" 				>> ${INSTALL_PATH}/package.runtime.postun.txt	

#
install-devel:
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
install-devel-package: install-devel
#
	echo ${INSTALL_PREFIX}/lib/libabcdk.so 			>> ${INSTALL_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/libabcdk.a 			>> ${INSTALL_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/lib/pkgconfig/abcdk.pc 	>> ${INSTALL_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk 			>> ${INSTALL_PATH}/package.devel.files.txt
	echo ${INSTALL_PREFIX}/include/abcdk.h 			>> ${INSTALL_PATH}/package.devel.files.txt
#
	echo "#abcdk-devel-post-begin" >> ${INSTALL_PATH}/package.devel.post.txt
	echo "#abcdk-devel-post-end" >> ${INSTALL_PATH}/package.devel.post.txt
#
	echo "#abcdk-devel-postun-begin" >> ${INSTALL_PATH}/package.devel.postun.txt
	echo "#abcdk-devel-postun-end" >> ${INSTALL_PATH}/package.devel.postun.txt


#
uninstall-runtime:
#
	unlink ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_MAIN}
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_FULL}
	rm -f ${INSTALL_PATH_BIN}/abcdk-tool
	rm -rf ${INSTALL_PATH_BIN}/abcdk-script
	rm -rf $(INSTALL_PATH_DOC)/abcdk

	
#
uninstall-devel:
#
	unlink ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
	rm -rf ${INSTALL_PATH_INC}/abcdk
	rm -f ${INSTALL_PATH_INC}/abcdk.h
	rm -f ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc

	

