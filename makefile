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

#gcc临时文件目录。
export TMPDIR=${BUILD_PATH}/

#
ifeq (${BUILD_TYPE},debug)
CC_FLAGS += -g
LINK_FLAGS += -g
else 
LINK_FLAGS += -s
endif

#
CC_OPLV = -O2
ifeq (${OPTIMIZE_LEVEL},1)
CC_OPLV = -O1
endif
ifeq (${OPTIMIZE_LEVEL},3)
CC_OPLV = -O3
endif
ifeq (${OPTIMIZE_LEVEL},s)
CC_OPLV = -Os
endif
ifeq (${OPTIMIZE_LEVEL},fast)
CC_OPLV = -Ofast
endif


#
ifeq (${BUILD_OPTIMIZE},yes)
CC_FLAGS += ${CC_OPLV}
endif

#
ifdef SYSROOT_PATH
CC_FLAGS += --sysroot="${SYSROOT_PATH}"
LINK_FLAGS += --sysroot="${SYSROOT_PATH}"
endif 

#
LINK_FLAGS += -Wl,--as-needed
LINK_FLAGS += -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"
LINK_FLAGS += -ldl -pthread -lrt -lc -lm
LINK_FLAGS += ${DEPEND_LINKS}

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
CC_FLAGS += -D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 
CC_FLAGS += ${DEPEND_FLAGS}

#
CC_FLAGS += -I$(CURDIR)/abcdk/include/

#
LINK_FLAGS += -L${BUILD_PATH}

#
OBJ_PATH = ${BUILD_PATH}/tmp

#
LIB_SRC_FILES += $(wildcard abcdk/source/util/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/shell/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/mp4/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/log/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/rtp/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/ffmpeg/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/audio/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/database/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/http/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/json/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/sdp/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/rtsp/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/openssl/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/video/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/image/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/curl/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/net/*.c)
LIB_SRC_FILES += $(wildcard abcdk/source/enigma/*.c)
LIB_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LIB_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: abcdk test

#
all: abcdk test

#
abcdk: abcdk-src
	mkdir -p $(BUILD_PATH)
	$(CC) -shared -o $(BUILD_PATH)/libabcdk.so $(LIB_OBJ_FILES) $(LINK_FLAGS)
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
abcdk-src: $(LIB_OBJ_FILES)
	
#
test: test-src abcdk
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/test ${TEST_OBJ_FILES} -l:libabcdk.so $(LINK_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

#
$(OBJ_PATH)/abcdk/source/util/%.o: abcdk/source/util/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/util/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/log/%.o: abcdk/source/log/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/log/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/shell/%.o: abcdk/source/shell/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/shell/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/mp4/%.o: abcdk/source/mp4/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/mp4/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/rtp/%.o: abcdk/source/rtp/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/rtp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/ffmpeg/%.o: abcdk/source/ffmpeg/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/ffmpeg/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/audio/%.o: abcdk/source/audio/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/audio/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/database/%.o: abcdk/source/database/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/database/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/image/%.o: abcdk/source/image/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/image/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/json/%.o: abcdk/source/json/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/json/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/http/%.o: abcdk/source/http/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/http/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/sdp/%.o: abcdk/source/sdp/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/sdp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/rtsp/%.o: abcdk/source/rtsp/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/rtsp/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/openssl/%.o: abcdk/source/openssl/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/openssl/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/video/%.o: abcdk/source/video/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/video/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/curl/%.o: abcdk/source/curl/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/curl/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/net/%.o: abcdk/source/net/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/net/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/abcdk/source/enigma/%.o: abcdk/source/enigma/%.c
	mkdir -p $(OBJ_PATH)/abcdk/source/enigma/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@


#
$(OBJ_PATH)/test/%.o: test/%.c
	mkdir -p $(OBJ_PATH)/test/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
clean: clean-abcdk clean-test

#
clean-abcdk:
	rm -rf ${OBJ_PATH}/abcdk
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-test:
	rm -rf ${OBJ_PATH}/test
	rm -f $(BUILD_PATH)/test

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
	cp  -rf $(CURDIR)/abcdk/include/abcdk ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/abcdk/include/abcdk.h ${INSTALL_PATH_INC}/
#
	cp  -f ${PKG_PC} ${INSTALL_PATH_PKG}/abcdk.pc

#
uninstall: uninstall-runtime uninstall-devel

#
uninstall-runtime:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
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
