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

#Compiler
CC = gcc
AR = ar

#可能在交叉编译环中。
ifneq ($(TARGET_PLATFORM),$(HOST_PLATFORM))
CC = $(TARGET_PLATFORM)-linux-gnu-gcc
AR = $(TARGET_PLATFORM)-linux-gnu-ar
endif

#Standard
CCC_STD = -std=c11

#
LINK_FLAGS += ${DEPEND_LIBS}
LINK_FLAGS += -Wl,--as-needed -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"

#
CCC_FLAGS += ${DEPEND_FLAGS}
CCC_FLAGS += -fPIC 
CCC_FLAGS += -Wno-unused-result 
CCC_FLAGS += -Wno-unused-variable 
CCC_FLAGS += -Wno-pointer-sign 
CCC_FLAGS += -Wno-unused-but-set-variable 
CCC_FLAGS += -Wno-unused-label
CCC_FLAGS += -Wno-strict-aliasing
CCC_FLAGS += -Wno-unused-function
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
all: lib tool test

#
lib: $(UTIL_OBJ_FILES) $(MP4_OBJ_FILES) $(AUTH_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/libabcdk.so $^ $(LINK_FLAGS) -shared
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $^

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/mp4/%.o: mp4/%.c
	mkdir -p $(OBJ_PATH)/mp4/
	rm -f $@
	$(CC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/auth/%.o: auth/%.c
	mkdir -p $(OBJ_PATH)/auth/
	rm -f $@
	$(CC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"


tool: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-mtx ${OBJ_PATH}/tool/mtx.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-mt ${OBJ_PATH}/tool/mt.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-lsb ${OBJ_PATH}/tool/release.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-odbc ${OBJ_PATH}/tool/odbc.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-html ${OBJ_PATH}/tool/html.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-robots ${OBJ_PATH}/tool/robots.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-hexdump ${OBJ_PATH}/tool/hexdump.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-mp4dump ${OBJ_PATH}/tool/mp4dump.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-mp4juicer ${OBJ_PATH}/tool/mp4juicer.o -l:libabcdk.a $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/abcdk-mklicence ${OBJ_PATH}/tool/mklicence.o -l:libabcdk.a $(LINK_FLAGS)

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

test: ${TEST_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/epollex_test ${OBJ_PATH}/test/epollex_test.o -l:libabcdk.so $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/util_test ${OBJ_PATH}/test/util_test.o -l:libabcdk.so  $(LINK_FLAGS)

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean: clean-lib clean-tool clean-test
	rm -rf ${OBJ_PATH}

#
clean-lib:
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-tool:
	rm -f $(BUILD_PATH)/abcdk-mtx
	rm -f $(BUILD_PATH)/abcdk-mt
	rm -f $(BUILD_PATH)/abcdk-lsb
	rm -f $(BUILD_PATH)/abcdk-odbc
	rm -f $(BUILD_PATH)/abcdk-html
	rm -f $(BUILD_PATH)/abcdk-robots
	rm -f $(BUILD_PATH)/abcdk-hexdump
	rm -f $(BUILD_PATH)/abcdk-mp4dump
	rm -f $(BUILD_PATH)/abcdk-mp4juicer
	rm -f $(BUILD_PATH)/abcdk-mklicence

#
clean-test:
	rm -f $(BUILD_PATH)/epollex_test
	rm -f $(BUILD_PATH)/util_test

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
install: install-lib install-tool install-ldc install-pkg

#
install-lib:
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/libabcdk.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/abcdk-util ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/abcdk-mp4 ${INSTALL_PATH_INC}/
	cp  -rf $(CURDIR)/include/abcdk-auth ${INSTALL_PATH_INC}/
#
install-tool:
	mkdir -p ${INSTALL_PATH_BIN}
	cp -f $(BUILD_PATH)/abcdk-mtx ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-mt ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-lsb ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-odbc ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-html ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-robots ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-hexdump ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-mp4dump ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-mp4juicer ${INSTALL_PATH_BIN}/
	cp -f $(BUILD_PATH)/abcdk-mklicence ${INSTALL_PATH_BIN}/

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
	echo "Libs: -labcdk -L\$${libdir}" >> ${PKG_FILE}
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
uninstall: uninstall-lib uninstall-tool uninstall-ldc uninstall-pkg

#
uninstall-lib:
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
	rm -rf ${INSTALL_PATH_INC}/abcdk-util
	rm -rf ${INSTALL_PATH_INC}/abcdk-mp4
	rm -rf ${INSTALL_PATH_INC}/abcdk-auth
#
uninstall-tool:
	rm -f $(INSTALL_PATH_BIN)/abcdk-mtx
	rm -f $(INSTALL_PATH_BIN)/abcdk-mt
	rm -f $(INSTALL_PATH_BIN)/abcdk-lsb
	rm -f $(INSTALL_PATH_BIN)/abcdk-odbc
	rm -f $(INSTALL_PATH_BIN)/abcdk-html
	rm -f $(INSTALL_PATH_BIN)/abcdk-robots
	rm -f $(INSTALL_PATH_BIN)/abcdk-hexdump
	rm -f $(INSTALL_PATH_BIN)/abcdk-mp4dump
	rm -f $(INSTALL_PATH_BIN)/abcdk-mp4juicer
	rm -f $(INSTALL_PATH_BIN)/abcdk-mklicence

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
