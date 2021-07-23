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
PKG_NAME = ${SOLUTION_NAME}.pc

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
INSTALL_PATH_LIB = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/lib/)
INSTALL_PATH_PKG = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/pkgconfig/)

#
install: abcdkutil-install abcdkcomm-install tools-install
#
	mkdir -p ${INSTALL_PATH_PKG}
	echo "prefix=${INSTALL_PREFIX}" > ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "libdir=\$${prefix}/lib/" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "incdir=\$${prefix}/include/" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Name: ${SOLUTION_NAME}" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Description: A better c development kit. " >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Version: ${VERSION_MAJOR}.${VERSION_MINOR}" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Cflags: -I\$${incdir}" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Libs: -labcdk-commn -labcdk-util -L\$${libdir}" >> ${INSTALL_PATH_PKG}/${PKG_NAME}
	echo "Libs.private: ${DEPEND_LIBS}" >> ${INSTALL_PATH_PKG}/${PKG_NAME}

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
	rm -f ${INSTALL_PATH_PKG}/${PKG_NAME}

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
PACK_TAR_NAME = ${SOLUTION_NAME}-${VERSION_MAJOR}.${VERSION_MINOR}-${TARGET_PLATFORM}.tar.gz
#
package: package-tar

#
package-tar: 
#
	make -C $(CURDIR)
	make -C $(CURDIR) install ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${BUILD_PATH}/${PACK_TAR_NAME}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "abcdk"
	make -C $(CURDIR) uninstall ROOT_PATH=${TMP_ROOT_PATH}
	make -C $(CURDIR) clean
#
	@echo "\n"
	@echo "${BUILD_PATH}/${PACK_TAR_NAME}"

