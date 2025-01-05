#
# This file is part of ABCDK.
#
# Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/build/makefile.conf)

# 加载配置项。
include ${MAKE_CONF}

#gcc临时文件目录。
export TMPDIR=${BUILD_PATH}/

#
CC_FLAGS += -std=${CSTD}
CC_FLAGS += -fPIC 
CC_FLAGS += -Wno-unused-result
CC_FLAGS += -Wno-unused-variable 
CC_FLAGS += -Wno-pointer-sign 
CC_FLAGS += -Wno-unused-but-set-variable 
CC_FLAGS += -Wno-unused-label
CC_FLAGS += -Wno-strict-aliasing
CC_FLAGS += -Wno-unused-function
#CC_FLAGS += -Wno-sizeof-pointer-memaccess
CC_FLAGS += -Wno-deprecated-declarations
#CC_FLAGS += -Wint-to-pointer-cast
CC_FLAGS += -Wno-attributes
CC_FLAGS += -Wno-format
CC_FLAGS += -DABCDK_VERSION_MAJOR=${VERSION_MAJOR} 
CC_FLAGS += -DABCDK_VERSION_MINOR=${VERSION_MINOR} 
CC_FLAGS += -DABCDK_VERSION_RELEASE=${VERSION_RELEASE}
CC_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#
LINK_FLAGS += -Wl,--as-needed

#
ifeq (${BUILD_TYPE},debug)
CC_FLAGS += -g
LINK_FLAGS += -g
else 
LINK_FLAGS += -s
endif

#
ifeq (${BUILD_OPTIMIZE},yes)
ifeq (${OPTIMIZE_LEVEL},1)
CC_FLAGS += -O1
endif
ifeq (${OPTIMIZE_LEVEL},2)
CC_FLAGS += -O2
endif
ifeq (${OPTIMIZE_LEVEL},3)
CC_FLAGS += -O3
endif
ifeq (${OPTIMIZE_LEVEL},s)
CC_FLAGS += -Os
endif
ifeq (${OPTIMIZE_LEVEL},fast)
CC_FLAGS += -Ofast
endif
endif

#
ifdef SYSROOT_PREFIX
CC_FLAGS += --sysroot="${SYSROOT_PREFIX}"
LINK_FLAGS += --sysroot="${SYSROOT_PREFIX}"
endif 

#
LINK_FLAGS += -Wl,-rpath="./"
LINK_FLAGS += ${DEPEND_LINKS}

#
ifeq (${SYSROOT_RELEASE},linux-gnu)
CC_FLAGS += -D_GNU_SOURCE
LINK_FLAGS += -ldl -pthread -lc -lm -lrt
endif
ifeq (${SYSROOT_RELEASE},android)
LINK_FLAGS += -ldl -pthread -lc -lm
endif

#
CC_FLAGS += ${DEPEND_FLAGS}

#
CC_FLAGS += -I$(CURDIR)/lib/include/

#
LINK_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#
LIB_SRC_FILES += $(wildcard lib/source/util/*.c)
LIB_SRC_FILES += $(wildcard lib/source/shell/*.c)
LIB_SRC_FILES += $(wildcard lib/source/mp4/*.c)
LIB_SRC_FILES += $(wildcard lib/source/log/*.c)
LIB_SRC_FILES += $(wildcard lib/source/rtp/*.c)
LIB_SRC_FILES += $(wildcard lib/source/ffmpeg/*.c)
LIB_SRC_FILES += $(wildcard lib/source/audio/*.c)
LIB_SRC_FILES += $(wildcard lib/source/database/*.c)
LIB_SRC_FILES += $(wildcard lib/source/http/*.c)
LIB_SRC_FILES += $(wildcard lib/source/json/*.c)
LIB_SRC_FILES += $(wildcard lib/source/sdp/*.c)
LIB_SRC_FILES += $(wildcard lib/source/rtsp/*.c)
LIB_SRC_FILES += $(wildcard lib/source/openssl/*.c)
LIB_SRC_FILES += $(wildcard lib/source/video/*.c)
LIB_SRC_FILES += $(wildcard lib/source/image/*.c)
LIB_SRC_FILES += $(wildcard lib/source/curl/*.c)
LIB_SRC_FILES += $(wildcard lib/source/net/*.c)
LIB_SRC_FILES += $(wildcard lib/source/enigma/*.c)
LIB_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LIB_SRC_FILES}))

#
TOOL_SRC_FILES = $(wildcard tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: lib tool test

#
all: lib tool test

#
lib: lib-src
	mkdir -p $(BUILD_PATH)
	$(CC) -shared -o $(BUILD_PATH)/libabcdk.so $(LIB_OBJ_FILES) $(LINK_FLAGS)
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
lib-src: $(LIB_OBJ_FILES)

#
tool: tool-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-tool ${TOOL_OBJ_FILES} -l:libabcdk.so $(LINK_FLAGS)

#
tool-src: ${TOOL_OBJ_FILES} 

#
test: test-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-test ${TEST_OBJ_FILES} -l:libabcdk.a $(LINK_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

#
$(OBJ_PATH)/lib/source/util/%.o: lib/source/util/%.c
	mkdir -p $(OBJ_PATH)/lib/source/util/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/log/%.o: lib/source/log/%.c
	mkdir -p $(OBJ_PATH)/lib/source/log/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/shell/%.o: lib/source/shell/%.c
	mkdir -p $(OBJ_PATH)/lib/source/shell/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/mp4/%.o: lib/source/mp4/%.c
	mkdir -p $(OBJ_PATH)/lib/source/mp4/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/rtp/%.o: lib/source/rtp/%.c
	mkdir -p $(OBJ_PATH)/lib/source/rtp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/ffmpeg/%.o: lib/source/ffmpeg/%.c
	mkdir -p $(OBJ_PATH)/lib/source/ffmpeg/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/audio/%.o: lib/source/audio/%.c
	mkdir -p $(OBJ_PATH)/lib/source/audio/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/database/%.o: lib/source/database/%.c
	mkdir -p $(OBJ_PATH)/lib/source/database/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/image/%.o: lib/source/image/%.c
	mkdir -p $(OBJ_PATH)/lib/source/image/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/json/%.o: lib/source/json/%.c
	mkdir -p $(OBJ_PATH)/lib/source/json/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/http/%.o: lib/source/http/%.c
	mkdir -p $(OBJ_PATH)/lib/source/http/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/sdp/%.o: lib/source/sdp/%.c
	mkdir -p $(OBJ_PATH)/lib/source/sdp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/rtsp/%.o: lib/source/rtsp/%.c
	mkdir -p $(OBJ_PATH)/lib/source/rtsp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/openssl/%.o: lib/source/openssl/%.c
	mkdir -p $(OBJ_PATH)/lib/source/openssl/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/video/%.o: lib/source/video/%.c
	mkdir -p $(OBJ_PATH)/lib/source/video/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/curl/%.o: lib/source/curl/%.c
	mkdir -p $(OBJ_PATH)/lib/source/curl/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/net/%.o: lib/source/net/%.c
	mkdir -p $(OBJ_PATH)/lib/source/net/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/enigma/%.o: lib/source/enigma/%.c
	mkdir -p $(OBJ_PATH)/lib/source/enigma/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/ssl/%.o: lib/source/ssl/%.c
	mkdir -p $(OBJ_PATH)/lib/source/ssl/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
clean: clean-lib clean-tool clean-test

#
clean-lib:
	rm -rf ${OBJ_PATH}/lib
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-tool:
	rm -rf ${OBJ_PATH}/tool
	rm -f $(BUILD_PATH)/abcdk-tool

#
clean-test:
	rm -rf ${OBJ_PATH}/test
	rm -f $(BUILD_PATH)/abcdk-test

#
INSTALL_PATH=${ROOT_PATH}/${INSTALL_PREFIX}
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)
INSTALL_PATH_DOC = $(abspath ${INSTALL_PATH}/doc/)
INSTALL_PATH_PKG = $(abspath ${INSTALL_PATH}/pkgconfig/)
INSTALL_PATH_SPT = $(abspath ${INSTALL_PATH}/script/)


#
install: install-runtime install-devel

#
install-runtime:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_BIN}
	mkdir -p ${INSTALL_PATH_DOC}
	mkdir -p ${INSTALL_PATH_SPT}
#
	cp -f $(BUILD_PATH)/libabcdk.so ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}/
#
	cp -rf $(CURDIR)/script/. ${INSTALL_PATH_SPT}/
	cp -rf $(CURDIR)/doc/. ${INSTALL_PATH_DOC}/
	
#
install-devel:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_INC}
	mkdir -p ${INSTALL_PATH_PKG}
#
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
#
	cp  -rf $(CURDIR)/lib/include/abcdk ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/lib/include/abcdk.h ${INSTALL_PATH_INC}/
#
	cp  -f ${PKG_PC} ${INSTALL_PATH_PKG}/abcdk.pc

#
uninstall: uninstall-runtime uninstall-devel

#
uninstall-runtime:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
	rm -f ${INSTALL_PATH_BIN}/abcdk-tool
#
	rm -rf $(INSTALL_PATH_SPT)/*
	rm -rf $(INSTALL_PATH_DOC)/*
	
#
uninstall-devel:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
#
	rm -rf ${INSTALL_PATH_INC}/abcdk
	rm -f ${INSTALL_PATH_INC}/abcdk.h
#
	rm -f  ${INSTALL_PATH_PKG}/abcdk.pc

#占位预定义，实际会随机生成。
TMP_ROOT_PATH = /tmp/abcdk-build-installer.tmp
#
PACKAGE_PATH = ${BUILD_PACKAGE_PATH}/${VERSION_MAJOR}.${VERSION_MINOR}/
#
RUNTIME_PACKAGE_NAME=abcdk-${VERSION_STR}-${TARGET_PLATFORM}
#
DEVEL_PACKAGE_NAME=abcdk-devel-${VERSION_STR}-${TARGET_PLATFORM}


#
package: package-tar package-${KIT_NAME}

#
package-tar: package-runtime-tar package-devel-tar


#
package-runtime-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -czv -f "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "abcdk"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -czv -f "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "abcdk"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}


#
package-${KIT_NAME}: package-runtime-${KIT_NAME} package-devel-${KIT_NAME}

#
package-runtime-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_RT_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${RUNTIME_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_DEV_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${DEVEL_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

#
package-runtime-deb:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	cp -rf ${DEB_RT_CTL} ${TMP_ROOT_PATH}/DEBIAN
#	创建软链接，因为dpkg-shlibdeps要使用debian/control文件。下同。
	ln -s -f ${TMP_ROOT_PATH}/DEBIAN ${TMP_ROOT_PATH}/debian
	${DEB_TOOL_ROOT}/dpkg-shlibdeps2control.sh "${TMP_ROOT_PATH}"
#	删除软链接，因为dpkg-deb会把这个当成普通文件复制。下同。
	unlink ${TMP_ROOT_PATH}/debian
	mkdir -p ${PACKAGE_PATH}
	dpkg-deb --build "${TMP_ROOT_PATH}/" "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.deb"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

package-devel-deb:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	cp -rf ${DEB_DEV_CTL} ${TMP_ROOT_PATH}/DEBIAN
	mkdir -p ${PACKAGE_PATH}
	dpkg-deb --build "${TMP_ROOT_PATH}/" "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.deb"
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
	@echo "make package-tar"
	@echo "make package-${KIT_NAME}"
