#
# This file is part of ABCDK.
#
# MIT License
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/build/makefile.conf)

# 加载配置项。
include ${MAKE_CONF}

#
VERSION_STR = ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}

# C Standard
CC_STD = -std=c11

#
LINK_FLAGS += -fPIC
LINK_FLAGS += -Wl,--as-needed 
LINK_FLAGS += -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"
LINK_FLAGS += ${DEPEND_LIBS}

#
CC_FLAGS += -fPIC 
CC_FLAGS += -Wno-unused-result 
CC_FLAGS += -Wno-unused-variable 
CC_FLAGS += -Wno-pointer-sign 
CC_FLAGS += -Wno-unused-but-set-variable 
CC_FLAGS += -Wno-unused-label
CC_FLAGS += -Wno-strict-aliasing
CC_FLAGS += -Wno-unused-function
CC_FLAGS += -Wno-sizeof-pointer-memaccess
CC_FLAGS += -DVERSION_MAJOR=${VERSION_MAJOR} 
CC_FLAGS += -DVERSION_MINOR=${VERSION_MINOR} 
CC_FLAGS += -DVERSION_RELEASE=${VERSION_RELEASE} 
CC_FLAGS += -DBUILD_TIME=\"${BUILD_TIME}\"
CC_FLAGS += ${DEPEND_FLAGS}

#
ifeq (${BUILD_TYPE},debug)
CC_FLAGS += -g
LINK_FLAGS += -g
else 
CC_FLAGS += -O2
LINK_FLAGS += -s
endif

#
CC_FLAGS += -I$(CURDIR)
 
#
LINK_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#
BASE_SRC_FILES += $(wildcard util/*.c)
BASE_SRC_FILES += $(wildcard shell/*.c)
BASE_SRC_FILES += $(wildcard mp4/*.c)
BASE_SRC_FILES += $(wildcard comm/*.c)
BASE_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${BASE_SRC_FILES}))

#
TOOL_SRC_FILES = $(wildcard tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#
VMTX_SRC_FILES = $(wildcard vmtx/*.c)
VMTX_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${VMTX_SRC_FILES}))

#
all: base tool test vmtx

#
base: base-src
#
base-src: $(BASE_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/libabcdk.so $^ $(LINK_FLAGS) -shared
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $^

#
tool: base tool-src
#
tool-src: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-tool $^  -l:libabcdk.a $(LINK_FLAGS)

#
test: base test-src
#
test-src: ${TEST_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/epollex_test ${OBJ_PATH}/test/epollex_test.o  -l:libabcdk.so $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/util_test ${OBJ_PATH}/test/util_test.o -l:libabcdk.so $(LINK_FLAGS)

#
vmtx: base vmtx-src
#
vmtx-src: ${VMTX_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-vmtx $^  -l:libabcdk.a $(LINK_FLAGS)

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"
#
$(OBJ_PATH)/shell/%.o: shell/%.c
	mkdir -p $(OBJ_PATH)/shell/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/mp4/%.o: mp4/%.c
	mkdir -p $(OBJ_PATH)/mp4/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/comm/%.o: comm/%.c
	mkdir -p $(OBJ_PATH)/comm/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
$(OBJ_PATH)/vmtx/%.o: vmtx/%.c
	mkdir -p $(OBJ_PATH)/vmtx/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o "$@"

#
clean: clean-base clean-tool clean-test clean-vmtx
	rm -rf ${OBJ_PATH}

#
clean-base:
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-tool:
	rm -f $(BUILD_PATH)/abcdk-tool

#
clean-test:
	rm -f $(BUILD_PATH)/epollex_test
	rm -f $(BUILD_PATH)/util_test

#
clean-vmtx:
	rm -f $(BUILD_PATH)/abcdk-vmtx

#
INSTALL_PATH=${ROOT_PATH}/${INSTALL_PREFIX}
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)

#
INSTALL_LDC_PATH = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_LDC_FILE = $(abspath ${INSTALL_LDC_PATH}/ldconfig.sh)
INSTALL_PKG_PATH = $(abspath ${INSTALL_PATH}/pkgconfig/)
INSTALL_PKG_FILE = $(abspath ${INSTALL_PKG_PATH}/${SOLUTION_NAME}.pc)

#
install: install-runtime install-devel

#
install-runtime:
#
	mkdir -p ${INSTALL_PATH_LIB}
#
	cp -f $(BUILD_PATH)/libabcdk.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
#
	mkdir -p ${INSTALL_LDC_PATH}
	echo "#!/bin/bash" > ${INSTALL_LDC_FILE}
	echo "SHELL_PWD=\$$(cd \`dirname \$$0\`; pwd)" >> ${INSTALL_LDC_FILE}
	echo "[ \$$UID -ne 0 ] &&  echo \"you are not root.\" && exit" >> ${INSTALL_LDC_FILE}
	echo "echo \"\$${SHELL_PWD}/\" > /etc/ld.so.conf.d/${SOLUTION_NAME}.conf" >> ${INSTALL_LDC_FILE}
	echo "ldconfig"  >> ${INSTALL_LDC_FILE}
	chmod 755 ${INSTALL_LDC_FILE}
#
	mkdir -p ${INSTALL_PATH_BIN}
#
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}
	cp -f $(BUILD_PATH)/abcdk-vmtx ${INSTALL_PATH_BIN}

#
install-devel:
#
	mkdir -p ${INSTALL_PATH_INC}/util
	mkdir -p ${INSTALL_PATH_INC}/shell
	mkdir -p ${INSTALL_PATH_INC}/mp4
	mkdir -p ${INSTALL_PATH_INC}/comm
#
	cp  -f $(CURDIR)/util/*.h ${INSTALL_PATH_INC}/util/
	cp  -f $(CURDIR)/shell/*.h ${INSTALL_PATH_INC}/shell/
	cp  -f $(CURDIR)/mp4/*.h ${INSTALL_PATH_INC}/mp4/
	cp  -f $(CURDIR)/comm/*.h ${INSTALL_PATH_INC}/comm/
#
	mkdir -p ${INSTALL_PKG_PATH}
	echo "prefix=${INSTALL_PREFIX}" > ${INSTALL_PKG_FILE}
	echo "libdir=\$${prefix}/lib/" >> ${INSTALL_PKG_FILE}
	echo "incdir=\$${prefix}/include/" >> ${INSTALL_PKG_FILE}
	echo "" >> ${INSTALL_PKG_FILE}
	echo "Name: ${SOLUTION_NAME}" >> ${INSTALL_PKG_FILE}
	echo "Description: A bad c development kit. " >> ${INSTALL_PKG_FILE}
	echo "Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}" >> ${INSTALL_PKG_FILE}
	echo "Cflags: -I\$${incdir}" >> ${INSTALL_PKG_FILE}
	echo "Libs: -labcdk -L\$${libdir}" >> ${INSTALL_PKG_FILE}
	echo "Libs.private: ${DEPEND_LIBS}" >> ${INSTALL_PKG_FILE}


#
uninstall: uninstall-runtime uninstall-devel

#
uninstall-runtime:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
#
	rm -f ${INSTALL_LDC_FILE}
#
	rm -f $(INSTALL_PATH_BIN)/abcdk-tool
	rm -f $(INSTALL_PATH_BIN)/abcdk-vmtx

#
uninstall-devel:
#
	rm -rf ${INSTALL_PATH_INC}/util
	rm -rf ${INSTALL_PATH_INC}/shell
	rm -rf ${INSTALL_PATH_INC}/mp4
	rm -rf ${INSTALL_PATH_INC}/comm
#
	rm -f ${INSTALL_PKG_FILE}
	
#
TMP_ROOT_PATH = /tmp/${SOLUTION_NAME}-build-installer.tmp
#
RUNTIME_PACKAGE_FILE = $(CURDIR)/package/${SOLUTION_NAME}-${VERSION_STR}-${OS_ID}-${OS_VER}-${TARGET_PLATFORM}.tar.gz
#
DEVEL_PACKAGE_FILE = $(CURDIR)/package/${SOLUTION_NAME}-devel-${VERSION_STR}.tar.gz

#
package: package-runtime package-devel

#
package-runtime:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${RUNTIME_PACKAGE_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}"
#	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
#
package-devel:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${DEVEL_PACKAGE_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}"
#	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}


help:
	@echo "make"
	@echo "make clean"
	@echo "make install"
	@echo "make install-runtime"
	@echo "make install-devel"
	@echo "make uninstall"
	@echo "make uninstall-runtime"
	@echo "make uninstall-devel"
	@echo "make package"
	@echo "make package-runtime"
	@echo "make package-devel"