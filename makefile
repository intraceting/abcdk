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
CC_FLAGS += -g -O2
LINK_FLAGS += -g
else 
CC_FLAGS += -O2
LINK_FLAGS += -s
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
CC_FLAGS += -I$(CURDIR)/
 
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
all: base tool test

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
	$(CC) -o $(BUILD_PATH)/abcdk $^  -l:libabcdk.a $(LINK_FLAGS)

#
test: base test-src
#
test-src: ${TEST_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/epollex_test ${OBJ_PATH}/test/epollex_test.o  -l:libabcdk.so $(LINK_FLAGS)
	$(CC) -o $(BUILD_PATH)/util_test ${OBJ_PATH}/test/util_test.o -l:libabcdk.so $(LINK_FLAGS)

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@
#
$(OBJ_PATH)/shell/%.o: shell/%.c
	mkdir -p $(OBJ_PATH)/shell/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/mp4/%.o: mp4/%.c
	mkdir -p $(OBJ_PATH)/mp4/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/comm/%.o: comm/%.c
	mkdir -p $(OBJ_PATH)/comm/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC) $(CC_STD) $(CC_FLAGS) -c $< -o $@

#
clean: clean-base clean-tool clean-test

#
clean-base:
	rm -rf ${OBJ_PATH}/util
	rm -rf ${OBJ_PATH}/mp4
	rm -rf ${OBJ_PATH}/comm
	rm -rf ${OBJ_PATH}/shell
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-tool:
	rm -rf ${OBJ_PATH}/tool
	rm -f $(BUILD_PATH)/abcdk

#
clean-test:
	rm -rf ${OBJ_PATH}/test
	rm -f $(BUILD_PATH)/epollex_test
	rm -f $(BUILD_PATH)/util_test

#
INSTALL_PATH=${ROOT_PATH}/${INSTALL_PREFIX}
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)

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
	mkdir -p ${INSTALL_PATH_BIN}
#
	cp -f $(BUILD_PATH)/abcdk ${INSTALL_PATH_BIN}/
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
uninstall: uninstall-runtime uninstall-devel

#
uninstall-runtime:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
#
	rm -f $(INSTALL_PATH_BIN)/abcdk

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
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
#
package-devel:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	tar -czv -f "${DEVEL_PACKAGE_FILE}" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}-${VERSION_STR}"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
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