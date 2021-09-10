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
UTIL_NAME = abcdk-util

#
MP4_NAME = abcdk-mp4

#
AUTH_NAME = abcdk-auth

#
MT_NAME = abcdk-mt
MTX_NAME = abcdk-mtx
RELEASE_NAME = abcdk-release
ODBC_NAME = abcdk-odbc
HTML_NAME = abcdk-html
ROBOTS_NAME = abcdk-robots
HEXDUMP_NAME = abcdk-hexdump
MP4DUMP_NAME = abcdk-mp4dump
MP4JUICER_NAME = abcdk-mp4juicer

#
EPOLLEX_TESTNAME = epollex_test
UTIL_TESTNAME = util_test

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
LINK_FLAGS += -g
else 
CCC_FLAGS += -O2
LINK_FLAGS += -s
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
MP4_SRC_FILES = $(wildcard mp4/*.c)
MP4_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${MP4_SRC_FILES}))

#
AUTH_SRC_FILES = $(wildcard auth/*.c)
AUTH_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${AUTH_SRC_FILES}))


#
TOOL_SRC_FILES = $(wildcard tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#
all: util mp4 auth tool test

util: ${UTIL_NAME}

#
${UTIL_NAME}: $(UTIL_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${UTIL_NAME}
	$(CCC) -o $(BUILD_PATH)/lib${UTIL_NAME}.so $^ $(LINK_FLAGS) -shared
	ar -cr $(BUILD_PATH)/lib${UTIL_NAME}.a $^

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

mp4: ${MP4_NAME}

#
${MP4_NAME}: $(MP4_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MP4_NAME}
	$(CCC) -o $(BUILD_PATH)/lib${MP4_NAME}.so $^ -l${UTIL_NAME} $(LINK_FLAGS) -shared
	ar -cr $(BUILD_PATH)/lib${MP4_NAME}.a $^

#
$(OBJ_PATH)/mp4/%.o: mp4/%.c
	mkdir -p $(OBJ_PATH)/mp4/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

auth: ${AUTH_NAME}

#
${AUTH_NAME}: $(AUTH_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${AUTH_NAME}
	$(CCC) -o $(BUILD_PATH)/lib${AUTH_NAME}.so $^ -l${UTIL_NAME} $(LINK_FLAGS) -shared
	ar -cr $(BUILD_PATH)/lib${AUTH_NAME}.a $^

#
$(OBJ_PATH)/auth/%.o: auth/%.c
	mkdir -p $(OBJ_PATH)/auth/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"


tool: ${MTX_NAME} ${MT_NAME} ${RELEASE_NAME} ${ODBC_NAME} ${HTML_NAME} ${ROBOTS_NAME} ${HEXDUMP_NAME} ${MP4DUMP_NAME} ${MP4JUICER_NAME}

#
${MTX_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MTX_NAME}
	$(CCC) -o $(BUILD_PATH)/${MTX_NAME} ${OBJ_PATH}/tool/mtx.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${MT_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MT_NAME}
	$(CCC) -o $(BUILD_PATH)/${MT_NAME} ${OBJ_PATH}/tool/mt.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${RELEASE_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${RELEASE_NAME}
	$(CCC) -o $(BUILD_PATH)/${RELEASE_NAME} ${OBJ_PATH}/tool/release.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${ODBC_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${ODBC_NAME}
	$(CCC) -o $(BUILD_PATH)/${ODBC_NAME} ${OBJ_PATH}/tool/odbc.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${HTML_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${HTML_NAME}
	$(CCC) -o $(BUILD_PATH)/${HTML_NAME} ${OBJ_PATH}/tool/html.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${ROBOTS_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${ROBOTS_NAME}
	$(CCC) -o $(BUILD_PATH)/${ROBOTS_NAME} ${OBJ_PATH}/tool/robots.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${HEXDUMP_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${HEXDUMP_NAME}
	$(CCC) -o $(BUILD_PATH)/${HEXDUMP_NAME} ${OBJ_PATH}/tool/hexdump.o -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${MP4DUMP_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MP4DUMP_NAME}
	$(CCC) -o $(BUILD_PATH)/${MP4DUMP_NAME} ${OBJ_PATH}/tool/mp4dump.o -l:lib${MP4_NAME}.a -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
${MP4JUICER_NAME}: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${MP4JUICER_NAME}
	$(CCC) -o $(BUILD_PATH)/${MP4JUICER_NAME} ${OBJ_PATH}/tool/mp4juicer.o -l:lib${MP4_NAME}.a -l:lib${UTIL_NAME}.a $(LINK_FLAGS)

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

test: ${EPOLLEX_TESTNAME} ${UTIL_TESTNAME}

#
${EPOLLEX_TESTNAME}: ${TEST_OBJ_FILES}
	rm -f $(BUILD_PATH)/${EPOLLEX_TESTNAME}
	$(CCC) -o $(BUILD_PATH)/${EPOLLEX_TESTNAME} ${OBJ_PATH}/test/epollex_test.o -l${UTIL_NAME} $(LINK_FLAGS)

#
${UTIL_TESTNAME}: ${TEST_OBJ_FILES}
	rm -f $(BUILD_PATH)/${UTIL_TESTNAME}
	$(CCC) -o $(BUILD_PATH)/${UTIL_TESTNAME} ${OBJ_PATH}/test/util_test.o -l${AUTH_NAME} -l${MP4_NAME} -l${UTIL_NAME}  $(LINK_FLAGS)

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean: clean-util clean-mp4 clean-auth clean-tool clean-test
	rm -rf ${OBJ_PATH}

#
clean-util:
	rm -f $(BUILD_PATH)/lib${UTIL_NAME}.so
	rm -f $(BUILD_PATH)/lib${UTIL_NAME}.a

#
clean-mp4:
	rm -f $(BUILD_PATH)/lib${MP4_NAME}.so
	rm -f $(BUILD_PATH)/lib${MP4_NAME}.a

#
clean-auth:
	rm -f $(BUILD_PATH)/lib${AUTH_NAME}.so
	rm -f $(BUILD_PATH)/lib${AUTH_NAME}.a

#
clean-tool:
	rm -f $(BUILD_PATH)/${MTX_NAME}
	rm -f $(BUILD_PATH)/${MT_NAME}
	rm -f $(BUILD_PATH)/${RELEASE_NAME}
	rm -f $(BUILD_PATH)/${ODBC_NAME}
	rm -f $(BUILD_PATH)/${HTML_NAME}
	rm -f $(BUILD_PATH)/${ROBOTS_NAME}
	rm -f $(BUILD_PATH)/${HEXDUMP_NAME}
	rm -f $(BUILD_PATH)/${MP4DUMP_NAME}
	rm -f $(BUILD_PATH)/${MP4JUICER_NAME}

#
clean-test:
	rm -f $(BUILD_PATH)/${EPOLLEX_TESTNAME}
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
install: install-util install-mp4 install-auth install-tool install-ldc install-pkg

#
install-util:
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/lib${UTIL_NAME}.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/lib${UTIL_NAME}.a ${INSTALL_PATH_LIB}/
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/${SOLUTION_NAME}-util ${INSTALL_PATH_INC}/

#
install-mp4:
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/lib${MP4_NAME}.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/lib${MP4_NAME}.a ${INSTALL_PATH_LIB}/
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/${SOLUTION_NAME}-mp4 ${INSTALL_PATH_INC}/

#
install-auth:
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/lib${AUTH_NAME}.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/lib${AUTH_NAME}.a ${INSTALL_PATH_LIB}/
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/${SOLUTION_NAME}-auth ${INSTALL_PATH_INC}/
#
install-tool:
	mkdir -p ${INSTALL_PATH_BIN}
	cp -f $(BUILD_PATH)/${MTX_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${MT_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${RELEASE_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${ODBC_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${HTML_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${ROBOTS_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${HEXDUMP_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${MP4DUMP_NAME} ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/${MP4JUICER_NAME} ${INSTALL_PATH_BIN}/

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
	echo "Libs: -l${AUTH_NAME} -l${MP4_NAME} -l${UTIL_NAME} -L\$${libdir}" >> ${PKG_FILE}
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
uninstall: uninstall-util uninstall-mp4 uninstall-auth uninstall-tool uninstall-ldc uninstall-pkg

#
uninstall-util:
	rm -f ${INSTALL_PATH_LIB}/lib${UTIL_NAME}.so
	rm -f ${INSTALL_PATH_LIB}/lib${UTIL_NAME}.a
	rm -rf ${INSTALL_PATH_INC}/${SOLUTION_NAME}-util

#
uninstall-mp4:
	rm -f ${INSTALL_PATH_LIB}/lib${MP4_NAME}.so
	rm -f ${INSTALL_PATH_LIB}/lib${MP4_NAME}.a
	rm -rf ${INSTALL_PATH_INC}/${SOLUTION_NAME}-mp4

#
uninstall-auth:
	rm -f ${INSTALL_PATH_LIB}/lib${AUTH_NAME}.so
	rm -f ${INSTALL_PATH_LIB}/lib${AUTH_NAME}.a
	rm -rf ${INSTALL_PATH_INC}/${SOLUTION_NAME}-auth
#
uninstall-tool:
	rm -f $(INSTALL_PATH_BIN)/${MTX_NAME}
	rm -f $(INSTALL_PATH_BIN)/${MT_NAME}
	rm -f $(INSTALL_PATH_BIN)/${RELEASE_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ODBC_NAME}
	rm -f $(INSTALL_PATH_BIN)/${HTML_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ROBOTS_NAME}
	rm -f $(INSTALL_PATH_BIN)/${HEXDUMP_NAME}
	rm -f $(INSTALL_PATH_BIN)/${MP4DUMP_NAME}
	rm -f $(INSTALL_PATH_BIN)/${MP4JUICER_NAME}

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
