#
# This file is part of ABCDK.
#
# MIT License
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/build/makefile.conf)

#加载配置项。
include ${MAKE_CONF}

#
SOLUTION_NAME ?= abcdk

#
ifeq (${VERSION_MAJOR},)
VERSION_MAJOR = 1
else ifeq (${VERSION_MAJOR},"")
VERSION_MAJOR = 1
endif

#
ifeq (${VERSION_MINOR},)
VERSION_MINOR = 0
else ifeq (${VERSION_MINOR},"")
VERSION_MINOR = 0
endif

#
ifeq (${INSTALL_PREFIX},)
INSTALL_PREFIX = /usr/local/${SOLUTION_NAME}/
else ifeq (${INSTALL_PREFIX},"")
INSTALL_PREFIX = /usr/local/${SOLUTION_NAME}/
endif

#
ifeq (${ROOT_PATH},)
ROOT_PATH = /
else ifeq (${ROOT_PATH},"")
ROOT_PATH = /
endif



#
all: abcdkutil abcdkcomm tools tests

#
abcdkutil: abcdkutil-clean
	make -C $(CURDIR)/abcdkutil/

#
abcdkcomm: abcdkcomm-clean
	make -C $(CURDIR)/abcdkcomm/

#
tools: tools-clean
	make -C $(CURDIR)/tools/

#
tests: tests-clean
	make -C $(CURDIR)/tests/

#
clean: abcdkutil-clean abcdkcomm-clean tools-clean tests-clean

#
abcdkutil-clean: 
	make -C $(CURDIR)/abcdkutil/ clean

#
abcdkcomm-clean: 
	make -C $(CURDIR)/abcdkcomm/ clean

#
tools-clean: 
	make -C $(CURDIR)/tools/ clean

#
tests-clean:
	make -C $(CURDIR)/tests/ clean

#
LDC_PATH = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/lib/)
LDC_FILE = $(abspath ${LDC_PATH}/ldconfig.sh)
PKG_PATH = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/pkgconfig/)
PKG_FILE = $(abspath ${PKG_PATH}/${SOLUTION_NAME}.pc)

#
install: abcdkutil-install abcdkcomm-install tools-install
#
	mkdir -p ${PKG_PATH}
	echo "prefix=${INSTALL_PREFIX}" > ${PKG_FILE}
	echo "libdir=\$${prefix}/lib/" >> ${PKG_FILE}
	echo "incdir=\$${prefix}/include/" >> ${PKG_FILE}
	echo "" >> ${PKG_FILE}
	echo "Name: ${SOLUTION_NAME}" >> ${PKG_FILE}
	echo "Description: A better c development kit. " >> ${PKG_FILE}
	echo "Version: ${VERSION_MAJOR}.${VERSION_MINOR}" >> ${PKG_FILE}
	echo "Cflags: -I\$${incdir}" >> ${PKG_FILE}
	echo "Libs: -labcdk-commn -labcdk-util -L\$${libdir}" >> ${PKG_FILE}
	echo "Libs.private: ${DEPEND_LIBS}" >> ${PKG_FILE}
#
	mkdir -p ${LDC_PATH}
	echo "#!/bin/sh" > ${LDC_FILE}
	echo "SHELL_PWD=\$$(cd \`dirname \$$0\`; pwd)" >> ${LDC_FILE}
	echo "[ \$$UID != 0 ] &&  echo \"System configuration requires root privileges. you are not root.\" && exit" >> ${LDC_FILE}
	echo "echo \"\$${SHELL_PWD}/\" > /etc/ld.so.conf.d/${SOLUTION_NAME}.conf" >> ${LDC_FILE}
	echo "ldconfig"  >> ${LDC_FILE}
	chmod 755 ${LDC_FILE}
	
#
abcdkutil-install: 
	make -C $(CURDIR)/abcdkutil/ install

#
abcdkcomm-install: 
	make -C $(CURDIR)/abcdkcomm/ install

#
tools-install:
	make -C $(CURDIR)/tools/ install

#
uninstall: abcdkutil-uninstall abcdkcomm-uninstall tools-uninstall
#
	rm -f ${PKG_FILE}
	rm -f ${LDC_FILE}

#
abcdkutil-uninstall: 
	make -C $(CURDIR)/abcdkutil/ uninstall

#
abcdkcomm-uninstall: 
	make -C $(CURDIR)/abcdkcomm/ uninstall

#
tools-uninstall: 
	make -C $(CURDIR)/tools/ uninstall

#
TMP_ROOT_PATH = /tmp/${SOLUTION_NAME}-build-installer.tmp/
PACKGE_TAR_NAME = ${SOLUTION_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}-${TARGET_PLATFORM}.tar.gz

#
package: package-tar

#
package-tar: 
#
	make -C $(CURDIR)
	make -C $(CURDIR) install ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${BUILD_PATH}/${PACKGE_TAR_NAME}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "abcdk"
	make -C $(CURDIR) uninstall ROOT_PATH=${TMP_ROOT_PATH}
	make -C $(CURDIR) clean
#
	@echo "\n"
	@echo "${BUILD_PATH}/${PACKGE_TAR_NAME}"

