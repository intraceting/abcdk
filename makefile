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

# C Standard
CC_STD = -std=c11

#
ifeq (${BUILD_TYPE},debug)
CC_FLAGS += -g 
LINK_FLAGS += -g
endif

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
CC_FLAGS += -I$(CURDIR)/src/
 
#
LINK_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#
BASE_SRC_FILES += $(wildcard src/util/*.c)
BASE_SRC_FILES += $(wildcard src/shell/*.c)
BASE_SRC_FILES += $(wildcard src/mp4/*.c)
BASE_SRC_FILES += $(wildcard src/comm/*.c)
BASE_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${BASE_SRC_FILES}))

#
TOOL_SRC_FILES = $(wildcard src/tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard src/test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#
VMTX_SRC_FILES = $(wildcard src/vmtx/*.c)
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
	$(CC) -o $(BUILD_PATH)/epollex_test ${OBJ_PATH}/src/test/epollex_test.o  -l:libabcdk.so $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/util_test ${OBJ_PATH}/src/test/util_test.o -l:libabcdk.so $(LINK_FLAGS)

#
vmtx: base vmtx-src
#
vmtx-src: ${VMTX_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-vmtx $^  -l:libabcdk.a $(LINK_FLAGS)


#
$(OBJ_PATH)/src/util/%.o: src/util/%.c
	mkdir -p $(OBJ_PATH)/src/util/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@
#
$(OBJ_PATH)/src/shell/%.o: src/shell/%.c
	mkdir -p $(OBJ_PATH)/src/shell/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/mp4/%.o: src/mp4/%.c
	mkdir -p $(OBJ_PATH)/src/mp4/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/comm/%.o: src/comm/%.c
	mkdir -p $(OBJ_PATH)/src/comm/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/tool/%.o: src/tool/%.c
	mkdir -p $(OBJ_PATH)/src/tool/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/test/%.o: src/test/%.c
	mkdir -p $(OBJ_PATH)/src/test/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/vmtx/%.o: src/vmtx/%.c
	mkdir -p $(OBJ_PATH)/src/vmtx/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@


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
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}/
#
	cp -f $(BUILD_PATH)/abcdk-vmc ${INSTALL_PATH_BIN}/

#
install-devel:
#
	mkdir -p ${INSTALL_PATH_INC}/util
	mkdir -p ${INSTALL_PATH_INC}/shell
	mkdir -p ${INSTALL_PATH_INC}/mp4
	mkdir -p ${INSTALL_PATH_INC}/comm
#
	cp  -f $(CURDIR)/src/util/*.h ${INSTALL_PATH_INC}/util/
	cp  -f $(CURDIR)/src/shell/*.h ${INSTALL_PATH_INC}/shell/
	cp  -f $(CURDIR)/src/mp4/*.h ${INSTALL_PATH_INC}/mp4/
	cp  -f $(CURDIR)/src/comm/*.h ${INSTALL_PATH_INC}/comm/

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
#
	rm -f $(INSTALL_PATH_BIN)/abcdk-vmc

#
uninstall-devel:
#
	rm -rf ${INSTALL_PATH_INC}/util
	rm -rf ${INSTALL_PATH_INC}/shell
	rm -rf ${INSTALL_PATH_INC}/mp4
	rm -rf ${INSTALL_PATH_INC}/comm
#
TMP_ROOT_PATH = /tmp/${SOLUTION_NAME}-${VERSION_STR}-build-installer.tmp
#
RUNTIME_PACKAGE_FILE = $(CURDIR)/package/${SOLUTION_NAME}-${VERSION_STR}-${TARGET_PLATFORM}.tar.gz
#
DEVEL_PACKAGE_FILE = $(CURDIR)/package/${SOLUTION_NAME}-devel-${VERSION_STR}.tar.gz

#
package: package-runtime package-devel

#
package-runtime:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${RUNTIME_PACKAGE_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}-${VERSION_STR}"
#	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
#
package-devel:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${DEVEL_PACKAGE_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}-${VERSION_STR}"
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