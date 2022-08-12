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
ifeq (${BUILD_TYPE},debug)
CC_FLAGS += -g
LINK_FLAGS += -g
else 
LINK_FLAGS += -s
endif

#
ifeq (${BUILD_OPTIMIZE},yes)
CC_FLAGS += -O2
endif

#
LINK_FLAGS += -Wl,--as-needed 
LINK_FLAGS += -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"
LINK_FLAGS += ${DEPEND_LIBS}

#
CC_FLAGS += -std=c11
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
BASE_SRC_FILES += $(wildcard log/*.c)
BASE_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${BASE_SRC_FILES}))

#
TOOL_SRC_FILES = $(wildcard tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
LOGD_SRC_FILES = $(wildcard logd/*.c)
LOGD_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LOGD_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))


#
all: base tool logd test

#
base: base-src
#
base-src: $(BASE_OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	$(CC) -shared -o $(BUILD_PATH)/libabcdk.so $^ $(LINK_FLAGS)
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $^

#
tool: base tool-src
#
tool-src: ${TOOL_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk $^ -l:libabcdk.a $(LINK_FLAGS)

#
logd: base logd-src
#
logd-src: ${LOGD_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-logd $^ -l:libabcdk.a $(LINK_FLAGS)

#
test: base test-src
#
test-src: ${TEST_OBJ_FILES}
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/test ${OBJ_PATH}/test/test.o -l:libabcdk.so $(LINK_FLAGS)

#
$(OBJ_PATH)/util/%.o: util/%.c
	mkdir -p $(OBJ_PATH)/util/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@
#
$(OBJ_PATH)/shell/%.o: shell/%.c
	mkdir -p $(OBJ_PATH)/shell/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/mp4/%.o: mp4/%.c
	mkdir -p $(OBJ_PATH)/mp4/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/comm/%.o: comm/%.c
	mkdir -p $(OBJ_PATH)/comm/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/log/%.o: log/%.c
	mkdir -p $(OBJ_PATH)/log/
	rm -f $@
	$(CC)  $(CC_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/logd/%.o: logd/%.c
	mkdir -p $(OBJ_PATH)/logd/
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
clean: clean-base clean-tool clean-logd clean-test

#
clean-base:
	rm -rf ${OBJ_PATH}/util
	rm -rf ${OBJ_PATH}/mp4
	rm -rf ${OBJ_PATH}/comm
	rm -rf ${OBJ_PATH}/shell
	rm -rf ${OBJ_PATH}/log
	rm -f $(BUILD_PATH)/libabcdk.so
	rm -f $(BUILD_PATH)/libabcdk.a

#
clean-tool:
	rm -rf ${OBJ_PATH}/tool
	rm -f $(BUILD_PATH)/abcdk

#
clean-logd:
	rm -rf ${OBJ_PATH}/logd
	rm -f $(BUILD_PATH)/abcdk-logd

#
clean-test:
	rm -rf ${OBJ_PATH}/test
	rm -f $(BUILD_PATH)/test

#
INSTALL_PATH=${ROOT_PATH}/${INSTALL_PREFIX}
INSTALL_PATH_3PARTY = $(abspath ${INSTALL_PATH}/3party/)
INSTALL_PATH_INC = $(abspath ${INSTALL_PATH}/include/)
INSTALL_PATH_LIB = $(abspath ${INSTALL_PATH}/lib/)
INSTALL_PATH_BIN = $(abspath ${INSTALL_PATH}/bin/)
INSTALL_PATH_PC = $(abspath ${INSTALL_PATH}/pkgconfig/)


#
install: install-runtime install-devel

#
install-runtime:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_BIN}
#
	cp -f $(BUILD_PATH)/libabcdk.so ${INSTALL_PATH_LIB}/
#
	cp -f $(BUILD_PATH)/abcdk ${INSTALL_PATH_BIN}/
	
#
install-devel:
#
	mkdir -p ${INSTALL_PATH_LIB}
	mkdir -p ${INSTALL_PATH_INC}/util
	mkdir -p ${INSTALL_PATH_INC}/shell
	mkdir -p ${INSTALL_PATH_INC}/mp4
	mkdir -p ${INSTALL_PATH_INC}/comm
	mkdir -p ${INSTALL_PATH_INC}/log
	mkdir -p ${INSTALL_PATH_PC}/
#
	cp -f $(BUILD_PATH)/libabcdk.a ${INSTALL_PATH_LIB}/
#
	cp  -f $(CURDIR)/util/*.h ${INSTALL_PATH_INC}/util/
	cp  -f $(CURDIR)/shell/*.h ${INSTALL_PATH_INC}/shell/
	cp  -f $(CURDIR)/mp4/*.h ${INSTALL_PATH_INC}/mp4/
	cp  -f $(CURDIR)/comm/*.h ${INSTALL_PATH_INC}/comm/
	cp  -f $(CURDIR)/log/*.h ${INSTALL_PATH_INC}/log/
#  
	cp  -f ${PKG_PC} ${INSTALL_PATH_PC}/abcdk.pc

#
uninstall: uninstall-runtime uninstall-devel

#
uninstall-runtime:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.so
#
	rm -f $(INSTALL_PATH_BIN)/abcdk
	rm -f $(INSTALL_PATH_BIN)/abcdk-logd
	
#
uninstall-devel:
#
	rm -f ${INSTALL_PATH_LIB}/libabcdk.a
#
	rm -rf ${INSTALL_PATH_INC}/util
	rm -rf ${INSTALL_PATH_INC}/shell
	rm -rf ${INSTALL_PATH_INC}/mp4
	rm -rf ${INSTALL_PATH_INC}/comm
	rm -rf ${INSTALL_PATH_INC}/log
#
	rm -f  ${INSTALL_PATH_PC}/abcdk.pc

#占位预定义，实际会随机生成。
TMP_ROOT_PATH = /tmp/${SOLUTION_NAME}-build-installer.tmp
#
PACKAGE_PATH = $(CURDIR)/package/
#
RUNTIME_PACKAGE_NAME=${SOLUTION_NAME}-${VERSION_STR}-${TARGET_PLATFORM}
#
DEVEL_PACKAGE_NAME=${SOLUTION_NAME}-devel-${VERSION_STR}-${TARGET_PLATFORM}


#
package: package-tar package-${KIT_NAME}

#
package-tar: package-runtime-tar package-devel-tar


#
package-runtime-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -czv -f "${PACKAGE_PATH}/${RUNTIME_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-tar:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	tar -czv -f "${PACKAGE_PATH}/${DEVEL_PACKAGE_NAME}.tar.gz" -C "${TMP_ROOT_PATH}/${INSTALL_PREFIX}/../" "${SOLUTION_NAME}"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}


#
package-${KIT_NAME}: package-runtime-${KIT_NAME} package-devel-${KIT_NAME}

#
package-runtime-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-runtime ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_RT_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${RUNTIME_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-runtime ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}
	
#
package-devel-rpm:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
	make -C $(CURDIR) install-devel ROOT_PATH=${TMP_ROOT_PATH}
	mkdir -p ${PACKAGE_PATH}
	rpmbuild --buildroot "${TMP_ROOT_PATH}/" -bb ${RPM_DEV_SPEC} --define="_rpmdir ${PACKAGE_PATH}" --define="_rpmfilename ${DEVEL_PACKAGE_NAME}.rpm"
	make -C $(CURDIR) uninstall-devel ROOT_PATH=${TMP_ROOT_PATH}
	rm -rf ${TMP_ROOT_PATH}

#
package-runtime-deb:
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
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
	$(eval TMP_ROOT_PATH := $(shell mktemp -d))
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
