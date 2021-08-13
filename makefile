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
VERSION_STR = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_RELEASE}

#
UTIL_NAME = libabcdk-util.so
UTIL_REALNAME = ${UTIL_NAME}.${VERSION_STR}

#
MT_REALNAME = abcdk-mt.exe
MT_NAME = abcdk-mt

#
MTX_REALNAME = abcdk-mtx.exe
MTX_NAME = abcdk-mtx

#
RELEASE_REALNAME = abcdk-release.exe
RELEASE_NAME = abcdk-release

#
ODBC_REALNAME = abcdk-odbc.exe
ODBC_NAME = abcdk-odbc

#
HTML_REALNAME = abcdk-html.exe
HTML_NAME = abcdk-html

#
ROBOTS_REALNAME = abcdk-robots.exe
ROBOTS_NAME = abcdk-robots

#
HEXDUMP_REALNAME = abcdk-hexdump.exe
HEXDUMP_NAME = abcdk-hexdump

#
MUX_TESTNAME = mux_test.exe

#
UTIL_TESTNAME = util_test.exe

#Compiler
CCC = gcc

#可能在交叉编译环中。
ifneq ($(TARGET_PLATFORM),$(HOST_PLATFORM))
    CCC = $(TARGET_PLATFORM)-linux-gnu-gcc
endif

#Standard
CCC_STD = -std=c11

#
LINK_FLAGS += ${DEPEND_LIBS}
LINK_FLAGS += -Wl,--as-needed -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"

#
CCC_FLAGS += ${DEPEND_FLAGS}
CCC_FLAGS += -fPIC -Wno-unused-result
CCC_FLAGS += -DVERSION_MAJOR=${VERSION_MAJOR} 
CCC_FLAGS += -DVERSION_MINOR=${VERSION_MINOR} 
CCC_FLAGS += -DVERSION_RELEASE=${VERSION_RELEASE} 
CCC_FLAGS += -DBUILD_TIME=\"${BUILD_TIME}\"

#
ifeq (${BUILD_TYPE},debug)
CCC_FLAGS += -g
else 
CCC_FLAGS += -s -O2
endif

#
CCC_FLAGS += -I$(CURDIR)/include/
 
#
LINK_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#
UTIL_SRC_FILES = $(wildcard util/*.c)
UTIL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${UTIL_SRC_FILES}))

#
TOOL_SRC_FILES = $(wildcard tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#
all: util tool test

util: ${UTIL_NAME}

#
${UTIL_REALNAME}: $(UTIL_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${UTIL_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${UTIL_REALNAME} $^ -Wl,--soname,${UTIL_NAME}  $(LINK_FLAGS) -shared

#
${UTIL_NAME}:${UTIL_REALNAME}
	ln -f -s ${UTIL_REALNAME} $(BUILD_PATH)/${UTIL_NAME}

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

tool: ${MTX_NAME} ${MT_NAME} ${RELEASE_NAME} ${ODBC_NAME} ${HTML_NAME} ${ROBOTS_NAME} ${HEXDUMP_NAME} 

#
${MTX_REALNAME}:${UTIL_NAME} ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MTX_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${MTX_REALNAME} ${OBJ_PATH}/tool/mtx.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${MTX_NAME}:${MTX_REALNAME}
	ln -f -s ${MTX_REALNAME} $(BUILD_PATH)/${MTX_NAME}

#
${MT_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MT_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${MT_REALNAME} ${OBJ_PATH}/tool/mt.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${MT_NAME}:${MT_REALNAME}
	ln -f -s ${MT_REALNAME} $(BUILD_PATH)/${MT_NAME}

#
${RELEASE_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	rm -f $(BUILD_PATH)/${RELEASE_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${RELEASE_REALNAME} ${OBJ_PATH}/tool/release.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${RELEASE_NAME}: ${RELEASE_REALNAME}
	ln -f -s ${RELEASE_REALNAME} $(BUILD_PATH)/${RELEASE_NAME}

#
${ODBC_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	rm -f $(BUILD_PATH)/${ODBC_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${ODBC_REALNAME} ${OBJ_PATH}/tool/odbc.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${ODBC_NAME}: ${ODBC_REALNAME}
	ln -f -s ${ODBC_REALNAME} $(BUILD_PATH)/${ODBC_NAME}

#
${HTML_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	rm -f $(BUILD_PATH)/${HTML_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${HTML_REALNAME} ${OBJ_PATH}/tool/html.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${HTML_NAME}: ${HTML_REALNAME}
	ln -f -s ${HTML_REALNAME} $(BUILD_PATH)/${HTML_NAME}

#
${ROBOTS_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	rm -f $(BUILD_PATH)/${ROBOTS_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${ROBOTS_REALNAME} ${OBJ_PATH}/tool/robots.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${ROBOTS_NAME}: ${ROBOTS_REALNAME}
	ln -f -s ${ROBOTS_REALNAME} $(BUILD_PATH)/${ROBOTS_NAME}

#
${HEXDUMP_REALNAME}: ${UTIL_NAME} ${TOOL_OBJ_FILES}
	rm -f $(BUILD_PATH)/${HEXDUMP_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${HEXDUMP_REALNAME} ${OBJ_PATH}/tool/hexdump.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${HEXDUMP_NAME}: ${HEXDUMP_REALNAME}
	ln -f -s ${HEXDUMP_REALNAME} $(BUILD_PATH)/${HEXDUMP_NAME}

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

test: ${MUX_TESTNAME} ${UTIL_TESTNAME}

#
${MUX_TESTNAME}: ${UTIL_NAME} ${TEST_OBJ_FILES}
	rm -f $(BUILD_PATH)/${MUX_TESTNAME}
	$(CCC) -o $(BUILD_PATH)/${MUX_TESTNAME} ${OBJ_PATH}/test/mux_test.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
${UTIL_TESTNAME}: ${UTIL_NAME} ${TEST_OBJ_FILES}
	rm -f $(BUILD_PATH)/${UTIL_TESTNAME}
	$(CCC) -o $(BUILD_PATH)/${UTIL_TESTNAME} ${OBJ_PATH}/test/util_test.o -l:${UTIL_NAME} $(LINK_FLAGS)

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean: clean-util clean-tool clean-test
	rm -rf ${OBJ_PATH}

#
clean-util:
	rm -f $(BUILD_PATH)/${UTIL_REALNAME}
	rm -f $(BUILD_PATH)/${UTIL_NAME}

#
clean-tool:
	rm -f $(BUILD_PATH)/${MTX_REALNAME}
	rm -f $(BUILD_PATH)/${MTX_NAME}
	rm -f $(BUILD_PATH)/${MT_REALNAME}
	rm -f $(BUILD_PATH)/${MT_NAME}
	rm -f $(BUILD_PATH)/${RELEASE_REALNAME}
	rm -f $(BUILD_PATH)/${RELEASE_NAME}
	rm -f $(BUILD_PATH)/${ODBC_REALNAME}
	rm -f $(BUILD_PATH)/${ODBC_NAME}
	rm -f $(BUILD_PATH)/${HTML_REALNAME}
	rm -f $(BUILD_PATH)/${HTML_NAME}
	rm -f $(BUILD_PATH)/${ROBOTS_REALNAME}
	rm -f $(BUILD_PATH)/${ROBOTS_NAME}
	rm -f $(BUILD_PATH)/${HEXDUMP_REALNAME}
	rm -f $(BUILD_PATH)/${HEXDUMP_NAME}

#
clean-test:
	rm -f $(BUILD_PATH)/${MUX_TESTNAME}
	rm -f $(BUILD_PATH)/${UTIL_TESTNAME}

#
INSTALL_PATH_INC = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/include/)
INSTALL_PATH_LIB = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/lib/)
INSTALL_PATH_BIN = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/bin/)

#
LDC_PATH = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/lib/)
LDC_FILE = $(abspath ${LDC_PATH}/ldconfig.sh)
PKG_PATH = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/pkgconfig/)
PKG_FILE = $(abspath ${PKG_PATH}/${SOLUTION_NAME}.pc)

#
install: install-util install-tool install-ldc install-pkg

#
install-util:
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/${UTIL_REALNAME} ${INSTALL_PATH_LIB}/
	ln -f -s ${UTIL_REALNAME} ${INSTALL_PATH_LIB}/${UTIL_NAME}
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/${SOLUTION_NAME}-util ${INSTALL_PATH_INC}/

#
install-tool:
	mkdir -p ${INSTALL_PATH_BIN}
	cp -f -f $(BUILD_PATH)/${MTX_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${MTX_REALNAME} $(INSTALL_PATH_BIN)/${MTX_NAME}
	cp -f -f $(BUILD_PATH)/${MT_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${MT_REALNAME} $(INSTALL_PATH_BIN)/${MT_NAME}
	cp -f -f $(BUILD_PATH)/${RELEASE_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${RELEASE_REALNAME} $(INSTALL_PATH_BIN)/${RELEASE_NAME}
	cp -f -f $(BUILD_PATH)/${ODBC_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${ODBC_REALNAME} $(INSTALL_PATH_BIN)/${ODBC_NAME}
	cp -f -f $(BUILD_PATH)/${HTML_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${HTML_REALNAME} $(INSTALL_PATH_BIN)/${HTML_NAME}
	cp -f -f $(BUILD_PATH)/${ROBOTS_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${ROBOTS_REALNAME} $(INSTALL_PATH_BIN)/${ROBOTS_NAME}
	cp -f -f $(BUILD_PATH)/${HEXDUMP_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${HEXDUMP_REALNAME} $(INSTALL_PATH_BIN)/${HEXDUMP_NAME}

#
install-ldc:
	mkdir -p ${PKG_PATH}
	echo "prefix=${INSTALL_PREFIX}" > ${PKG_FILE}
	echo "libdir=\$${prefix}/lib/" >> ${PKG_FILE}
	echo "incdir=\$${prefix}/include/" >> ${PKG_FILE}
	echo "" >> ${PKG_FILE}
	echo "Name: ${SOLUTION_NAME}" >> ${PKG_FILE}
	echo "Description: A better c development kit. " >> ${PKG_FILE}
	echo "Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}" >> ${PKG_FILE}
	echo "Cflags: -I\$${incdir}" >> ${PKG_FILE}
	echo "Libs: -l:${UTIL_NAME} -L\$${libdir}" >> ${PKG_FILE}
	echo "Libs.private: ${DEPEND_LIBS}" >> ${PKG_FILE}

#
install-pkg:
	mkdir -p ${LDC_PATH}
	echo "#!/bin/sh" > ${LDC_FILE}
	echo "SHELL_PWD=\$$(cd \`dirname \$$0\`; pwd)" >> ${LDC_FILE}
	echo "[ \$$UID != 0 ] &&  echo \"you are not root.\" && exit" >> ${LDC_FILE}
	echo "echo \"\$${SHELL_PWD}/\" > /etc/ld.so.conf.d/${SOLUTION_NAME}.conf" >> ${LDC_FILE}
	echo "ldconfig"  >> ${LDC_FILE}
	chmod 755 ${LDC_FILE}

#
uninstall: uninstall-util uninstall-tool uninstall-ldc uninstall-pkg

#
uninstall-util:
	rm -f ${INSTALL_PATH_LIB}/${UTIL_REALNAME}
	rm -f ${INSTALL_PATH_LIB}/${UTIL_NAME}
	rm -rf ${INSTALL_PATH_INC}/${SOLUTION_NAME}-util

#
uninstall-tool:
	rm -f $(INSTALL_PATH_BIN)/${MTX_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${MTX_NAME}
	rm -f $(INSTALL_PATH_BIN)/${MT_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${MT_NAME}
	rm -f $(INSTALL_PATH_BIN)/${RELEASE_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${RELEASE_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ODBC_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${ODBC_NAME}
	rm -f $(INSTALL_PATH_BIN)/${HTML_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${HTML_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ROBOTS_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${ROBOTS_NAME}
	rm -f $(INSTALL_PATH_BIN)/${HEXDUMP_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${HEXDUMP_NAME}

#
uninstall-ldc:
	rm -f ${LDC_FILE}

#
uninstall-pkg:
	rm -f ${PKG_FILE}
	
#
TMP_ROOT_PATH = /tmp/${SOLUTION_NAME}-build-installer.tmp

#
TAR_FILE = $(CURDIR)/package/${SOLUTION_NAME}-${VERSION_STR}-${TARGET_PLATFORM}.tar.gz

#
SPEC_FILE=$(BUILD_PATH)/${SOLUTION_NAME}.spec
RPM_PATH=$(CURDIR)/package

#
DEB_ARCH=$(shell dpkg-architecture |grep "DEB_TARGET_ARCH=" |cut -d '=' -f 2)
DEB_FILE=$(CURDIR)/package/${SOLUTION_NAME}-${VERSION_STR}.${DEB_ARCH}.deb
CTRL_FILE="${TMP_ROOT_PATH}/DEBIAN/control"
CLOG_FILE="${TMP_ROOT_PATH}/DEBIAN/changelog"

#
package: package-${KIT_NAME}

#
package-tar: clean
	make -C $(CURDIR)
	make -C $(CURDIR) install ROOT_PATH=${TMP_ROOT_PATH}
#	
	tar -czv -f "${TAR_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}"
#
	make -C $(CURDIR) uninstall ROOT_PATH=${TMP_ROOT_PATH}

#
package-rpm: clean
	make -C $(CURDIR)
	make -C $(CURDIR) install ROOT_PATH=${TMP_ROOT_PATH}
#
	echo "BuildRoot: ${TMP_ROOT_PATH}" > ${SPEC_FILE}
	echo 'Vendor: intraceting<intraceting@outlook.com>' >> ${SPEC_FILE}
	echo "Name: ${SOLUTION_NAME}" >> ${SPEC_FILE}
	echo "Version: ${VERSION_MAJOR}.${VERSION_MINOR}" >> ${SPEC_FILE}
	echo "Release: ${VERSION_RELEASE}" >> ${SPEC_FILE}
	echo 'Group: Development/Libraries' >> ${SPEC_FILE}
	echo 'License: MIT' >> ${SPEC_FILE}
	echo "Summary: A better c development kit." >> ${SPEC_FILE}
	echo '' >> ${SPEC_FILE}
	echo '%description' >> ${SPEC_FILE}
	echo '${SOLUTION_NAME} is a toolkit for simplifying the use of C as a development language.' >> ${SPEC_FILE}
	echo '' >> ${SPEC_FILE}
	echo '%files' >> ${SPEC_FILE}
	echo "${INSTALL_PREFIX}" >> ${SPEC_FILE}
	echo '' >> ${SPEC_FILE}
#	echo '%changelog' >> ${SPEC_FILE}
#	cat $(CURDIR)/CHANGELOG >> ${SPEC_FILE}
#	echo '' >> ${SPEC_FILE}
#
	rpmbuild --rmspec --buildroot "${TMP_ROOT_PATH}"  -bb "${SPEC_FILE}" --define="_rpmdir ${RPM_PATH}"
#
	make -C $(CURDIR) uninstall ROOT_PATH=${TMP_ROOT_PATH}

#
package-deb: clean
	make -C $(CURDIR)
	make -C $(CURDIR) install ROOT_PATH=${TMP_ROOT_PATH}
#
	mkdir -p ${TMP_ROOT_PATH}/DEBIAN/
	echo "Source: ${SOLUTION_NAME}" > ${CTRL_FILE}
	echo "Maintainer: intraceting<intraceting@outlook.com>" >> ${CTRL_FILE}
	echo "Package: ${SOLUTION_NAME}" >> ${CTRL_FILE}
	echo "Version: ${VERSION_STR}" >> ${CTRL_FILE}
	echo "Section: Development/Libraries" >> ${CTRL_FILE}
	echo "Priority: optional" >> ${CTRL_FILE}
	echo "Architecture: ${DEB_ARCH}" >> ${CTRL_FILE}
#	echo "Depends: \$${shlibs:Depends}" >> ${CTRL_FILE}
	echo "Description: ${SOLUTION_NAME} is a toolkit for simplifying the use of C as a development language." >> ${CTRL_FILE}
#	cat $(CURDIR)/CHANGELOG > ${CLOG_FILE}
#
	dpkg-deb --build "${TMP_ROOT_PATH}/" "${DEB_FILE}"
#
	make -C $(CURDIR) uninstall ROOT_PATH=${TMP_ROOT_PATH}
