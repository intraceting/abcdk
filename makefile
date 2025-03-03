#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/build/makefile.conf)

# 加载配置项。
include ${MAKE_CONF}

#gcc临时文件目录。
export TMPDIR=${BUILD_PATH}/

#
C_FLAGS += -fPIC 
C_FLAGS += -Wno-unused-result
C_FLAGS += -Wno-unused-variable 
C_FLAGS += -Wno-pointer-sign 
C_FLAGS += -Wno-unused-but-set-variable 
C_FLAGS += -Wno-unused-label
C_FLAGS += -Wno-strict-aliasing
C_FLAGS += -Wno-unused-function
C_FLAGS += -Wno-sizeof-pointer-memaccess
C_FLAGS += -Wno-deprecated-declarations
C_FLAGS += -Wint-to-pointer-cast
C_FLAGS += -Wno-attributes
C_FLAGS += -Wno-format
C_FLAGS += -DABCDK_VERSION_MAJOR=${VERSION_MAJOR} 
C_FLAGS += -DABCDK_VERSION_MINOR=${VERSION_MINOR} 
C_FLAGS += -DABCDK_VERSION_RELEASE=${VERSION_RELEASE}
C_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#
CXX_FLAGS += -fPIC 
CXX_FLAGS += -Wno-unused-result
CXX_FLAGS += -Wno-unused-variable 
#CXX_FLAGS += -Wno-pointer-sign 
CXX_FLAGS += -Wno-unused-but-set-variable 
CXX_FLAGS += -Wno-unused-label
CXX_FLAGS += -Wno-strict-aliasing
CXX_FLAGS += -Wno-unused-function
CXX_FLAGS += -Wno-sizeof-pointer-memaccess
CXX_FLAGS += -Wno-deprecated-declarations
CXX_FLAGS += -Wint-to-pointer-cast
CXX_FLAGS += -Wno-attributes
CXX_FLAGS += -Wno-format
CXX_FLAGS += -Wno-overloaded-virtual
CXX_FLAGS += -Wno-sign-conversion
CXX_FLAGS += -DABCDK_VERSION_MAJOR=${VERSION_MAJOR} 
CXX_FLAGS += -DABCDK_VERSION_MINOR=${VERSION_MINOR} 
CXX_FLAGS += -DABCDK_VERSION_RELEASE=${VERSION_RELEASE}
CXX_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#绑定C++编译器。
NVCC_FLAGS += -ccbin ${CC}
#抑制“未识别的属性”的诊断消息输出，让编译日志更简洁。
NVCC_FLAGS += -Xcudafe --diag_suppress=unrecognized_attribute

#
LD_FLAGS += -Wl,--as-needed

#
ifeq (${BUILD_TYPE},debug)
C_FLAGS += -g
CXX_FLAGS += -g
#生成调试信息。
NVCC_FLAGS += -g -G
endif
ifeq (${BUILD_TYPE},release)
#仅保留用于性能分性的调式信息。
NVCC_FLAGS += -lineinfo
LD_FLAGS += -s
endif

#
ifeq (${OPTIMIZE_LEVEL},1)
C_FLAGS += -O1
CXX_FLAGS += -O1
endif
ifeq (${OPTIMIZE_LEVEL},2)
C_FLAGS += -O2
CXX_FLAGS += -O2
endif
ifeq (${OPTIMIZE_LEVEL},3)
C_FLAGS += -O3
CXX_FLAGS += -O3
endif
ifeq (${OPTIMIZE_LEVEL},s)
C_FLAGS += -Os
CXX_FLAGS += -Os
endif
ifeq (${OPTIMIZE_LEVEL},fast)
C_FLAGS += -Ofast
CXX_FLAGS += -Ofast
endif

#
ifeq (${LSB_RELEASE},linux-gnu)
C_FLAGS += -D_GNU_SOURCE
CXX_FLAGS += -D_GNU_SOURCE
LD_FLAGS += -ldl -pthread -lc -lm -lrt -lstdc++ -lstdc++fs 
endif
ifeq (${LSB_RELEASE},android)
LD_FLAGS += -ldl -pthread -lc -lm -lstdc++ -lstdc++fs 
endif

#
C_FLAGS += ${DEPEND_FLAGS}
CXX_FLAGS += ${DEPEND_FLAGS}

#
C_FLAGS += -I$(CURDIR)/lib/include/
CXX_FLAGS += -I$(CURDIR)/lib/include/

#C++编译选项绑定到CUDA编译选项。
NVCC_FLAGS += $(addprefix -Xcompiler ,${CXX_FLAGS})

#
LD_FLAGS += ${DEPEND_LINKS}

#
LD_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#C
LIB_SRC_FILES += $(wildcard lib/source/util/*.c)
LIB_SRC_FILES += $(wildcard lib/source/system/*.c)
LIB_SRC_FILES += $(wildcard lib/source/mp4/*.c)
LIB_SRC_FILES += $(wildcard lib/source/net/*.c)
LIB_SRC_FILES += $(wildcard lib/source/ffmpeg/*.c)
LIB_SRC_FILES += $(wildcard lib/source/redis/*.c)
LIB_SRC_FILES += $(wildcard lib/source/sqlite/*.c)
LIB_SRC_FILES += $(wildcard lib/source/odbc/*.c)
LIB_SRC_FILES += $(wildcard lib/source/json/*.c)
LIB_SRC_FILES += $(wildcard lib/source/lz4/*.c)
LIB_SRC_FILES += $(wildcard lib/source/openssl/*.c)
LIB_SRC_FILES += $(wildcard lib/source/torch/*.c)
LIB_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LIB_SRC_FILES}))

#C++
LIB_SRC_CXX_FILES += $(wildcard lib/source/opencv/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard lib/source/torch/*.cpp)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cpp,%.cpp.o,${LIB_SRC_CXX_FILES}))

#CUDA是可选项，可能未启用。
ifneq ($(strip $(NVCC)),)
LIB_SRC_CU_FILES += $(wildcard lib/source/nvidia/*.cu)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cu,%.cu.o,${LIB_SRC_CU_FILES}))
endif

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
	$(CC) -shared -o $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} $(LIB_OBJ_FILES) $(LD_FLAGS) -Wl,-soname,libabcdk.so.${VERSION_STR_MAIN}
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
lib-src: $(LIB_OBJ_FILES)

#
tool: tool-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-tool ${TOOL_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
tool-src: ${TOOL_OBJ_FILES} 

#
test: test-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-test ${TEST_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

#
$(OBJ_PATH)/lib/source/util/%.o: lib/source/util/%.c
	mkdir -p $(OBJ_PATH)/lib/source/util/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/system/%.o: lib/source/system/%.c
	mkdir -p $(OBJ_PATH)/lib/source/system/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/mp4/%.o: lib/source/mp4/%.c
	mkdir -p $(OBJ_PATH)/lib/source/mp4/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/net/%.o: lib/source/net/%.c
	mkdir -p $(OBJ_PATH)/lib/source/net/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@


#
$(OBJ_PATH)/lib/source/ffmpeg/%.o: lib/source/ffmpeg/%.c
	mkdir -p $(OBJ_PATH)/lib/source/ffmpeg/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/redis/%.o: lib/source/redis/%.c
	mkdir -p $(OBJ_PATH)/lib/source/redis/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/sqlite/%.o: lib/source/sqlite/%.c
	mkdir -p $(OBJ_PATH)/lib/source/sqlite/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/odbc/%.o: lib/source/odbc/%.c
	mkdir -p $(OBJ_PATH)/lib/source/odbc/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/openssl/%.o: lib/source/openssl/%.c
	mkdir -p $(OBJ_PATH)/lib/source/openssl/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/lz4/%.o: lib/source/lz4/%.c
	mkdir -p $(OBJ_PATH)/lib/source/lz4/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/json/%.o: lib/source/json/%.c
	mkdir -p $(OBJ_PATH)/lib/source/json/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@
	
#
$(OBJ_PATH)/lib/source/opencv/%.cpp.o: lib/source/opencv/%.cpp
	mkdir -p $(OBJ_PATH)/lib/source/opencv/
	rm -f $@
	$(CC) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/torch/%.o: lib/source/torch/%.c
	mkdir -p $(OBJ_PATH)/lib/source/torch/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/torch/%.cpp.o: lib/source/torch/%.cpp
	mkdir -p $(OBJ_PATH)/lib/source/torch/
	rm -f $@
	$(CC) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/lib/source/nvidia/%.cu.o: lib/source/nvidia/%.cu
	mkdir -p $(OBJ_PATH)/lib/source/nvidia/
	rm -f $@
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -Xcompiler -std=c++11  -c $< -o $@

#
$(OBJ_PATH)/tool/%.o: tool/%.c
	mkdir -p $(OBJ_PATH)/tool/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@

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
INSTALL_PATH_DOC = $(abspath ${INSTALL_PATH}/share/)

#
install: install-runtime install-devel

#
install-runtime:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_BIN}/abcdk-script/
	mkdir -p ${INSTALL_PATH_DOC}/abcdk/
#
	cp -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} ${INSTALL_PATH_LIB}/
	cp -f $(BUILD_PATH)/abcdk-tool ${INSTALL_PATH_BIN}/
	cp -rf $(CURDIR)/script/. ${INSTALL_PATH_BIN}/abcdk-script/
	cp -rf $(CURDIR)/share/. ${INSTALL_PATH_DOC}/abcdk/
#	
	chmod 0555 ${INSTALL_PATH_LIB}/libabcdk.so.${VERSION_STR_FULL}
	cd ${INSTALL_PATH_LIB} ; ln -sf libabcdk.so.${VERSION_STR_FULL} libabcdk.so.${VERSION_STR_MAIN} ;
	chmod 0555 ${INSTALL_PATH_BIN}/abcdk-tool
	find ${INSTALL_PATH_BIN}/abcdk-script/ -type f -name "*.sh" -exec chmod 0555 {} \;
	find ${INSTALL_PATH_DOC}/abcdk/ -type f -exec chmod 0444 {} \;

#
install-devel:
#
	mkdir -p ${INSTALL_PATH_LIB}/pkgconfig/
	mkdir -p ${INSTALL_PATH_INC}
#
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
	cp  -rf $(CURDIR)/lib/include/abcdk ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/lib/include/abcdk.h ${INSTALL_PATH_INC}/
	cp  -f ${PKG_PC} ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc
#
	chmod 0555 ${INSTALL_PATH_LIB}/libabcdk.a
	cd ${INSTALL_PATH_LIB} ; ln -sf libabcdk.so.${VERSION_STR_MAIN} libabcdk.so ;
	find ${INSTALL_PATH_INC}/abcdk/ -type f -exec chmod 0444 {} \;
	chmod 0444 ${INSTALL_PATH_INC}/abcdk.h
	chmod 0444 ${INSTALL_PATH_LIB}/pkgconfig/abcdk.pc

#
uninstall: uninstall-runtime uninstall-devel

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

#占位预定义，实际会随机生成。
TMP_ROOT_PATH = /tmp/abcdk-build-installer.tmp
#
PACKAGE_PATH = ${BUILD_PACKAGE_PATH}/${VERSION_STR_MAIN}/
#
RUNTIME_PACKAGE_NAME=abcdk-${VERSION_STR_FULL}-${TARGET_PLATFORM}
#
DEVEL_PACKAGE_NAME=abcdk-devel-${VERSION_STR_FULL}-${TARGET_PLATFORM}


#
package: package-tar package-${KIT_NAME}

#
package-tar: package-runtime-tar package-devel-tar


#
package-runtime-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -cz -f "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/" "."
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -cz -f "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/" "."
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}


#
package-${KIT_NAME}: package-runtime-${KIT_NAME} package-devel-${KIT_NAME}

#
package-runtime-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --noclean --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_RT_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${RUNTIME_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --noclean --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_DEV_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${DEVEL_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

#
package-runtime-deb:
	$(eval TMP_ROOT_PATH := $(shell mktemp -p ${BUILD_PATH} -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	cp -rf ${DEB_RT_CTL} ${TMP_ROOT_PATH}/DEBIAN
#	创建软链接，因为dpkg-shlibdeps要使用debian/control文件。下同。
	ln -s -f ${TMP_ROOT_PATH}/DEBIAN ${TMP_ROOT_PATH}/debian
#   更新debian/control文件Pre-Depends字段。	
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
