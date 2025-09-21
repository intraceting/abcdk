#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#


#C
LIB_SRC_FILES += $(wildcard src/lib/source/util/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/system/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/mp4/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/net/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/ffmpeg/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/redis/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/sqlite/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/odbc/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/json/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/lz4/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/openssl/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/curl/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/rtsp/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/qrcode/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/torch/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/torch_host/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/torch_cuda/*.c)
LIB_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LIB_SRC_FILES}))

#C++
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/rtsp/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch_host/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch_host/bytetrack/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch_cuda/*.cpp)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cpp,%.o,${LIB_SRC_CXX_FILES}))

#CUDA是可选项，可能未启用。
ifeq (${HAVE_CUDA},yes)
LIB_SRC_CU_FILES += $(wildcard src/lib/source/torch_cuda/*.cu)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cu,%.o,${LIB_SRC_CU_FILES}))
endif

#
TOOL_SRC_FILES = $(wildcard src/tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#
TEST_SRC_FILES = $(wildcard src/test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: lib tool test xgettext

#
lib: lib-src
	mkdir -p $(BUILD_PATH)
	$(CXX) -shared -o $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} $(LIB_OBJ_FILES) $(LD_FLAGS) -Wl,-soname,libabcdk.so.${VERSION_STR_MAIN}
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
lib-src: $(LIB_OBJ_FILES)

#
clean-lib:
	rm -rf ${OBJ_PATH}/src/lib
	rm -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL}
	rm -f $(BUILD_PATH)/libabcdk.a


#
tool: tool-src lib
	mkdir -p $(BUILD_PATH)
	$(CXX) -o $(BUILD_PATH)/abcdk-tool ${TOOL_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
tool-src: ${TOOL_OBJ_FILES} 

#
clean-tool:
	rm -rf ${OBJ_PATH}/src/tool
	rm -f $(BUILD_PATH)/abcdk-tool

#
test: test-src lib
	mkdir -p $(BUILD_PATH)
	$(CXX) -o $(BUILD_PATH)/abcdk-test ${TEST_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

#
clean-test:
	rm -rf ${OBJ_PATH}/src/test
	rm -f $(BUILD_PATH)/abcdk-test

#
xgettext: xgettext-lib xgettext-tool

#把POT文件从share目录复制到build目录进行更新。
xgettext-lib:
	cp -f $(CURDIR)/share/locale/en_US/gettext/libabcdk.pot $(BUILD_PATH)/libabcdk.en_US.pot
	${SHELL_TOOLS_HOME}/xgettext.sh ABCDK ${VERSION_STR_FULL} TT $(CURDIR)/src/lib/ $(BUILD_PATH)/libabcdk.en_US.pot
	echo "'$(BUILD_PATH)/libabcdk.en_US.pot' Update completed."

#把POT文件从share目录复制到build目录进行更新。
xgettext-tool:
	cp -f $(CURDIR)/share/locale/en_US/gettext/abcdk-tool.pot $(BUILD_PATH)/abcdk-tool.en_US.pot
	${SHELL_TOOLS_HOME}/xgettext.sh ABCDK ${VERSION_STR_FULL} TT $(CURDIR)/src/tool/ $(BUILD_PATH)/abcdk-tool.en_US.pot
	echo "'$(BUILD_PATH)/abcdk-tool.en_US.pot' Update completed."

# $@: 目标文件
# $<: 源文件
# 自动匹配多级路径.
$(OBJ_PATH)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -std=c99  $(C_FLAGS)  -c $< -o $@

#
$(OBJ_PATH)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/%.o: %.cu
	mkdir -p $(dir $@)
	rm -f $@
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -Xcompiler -std=c++11  -c $< -o $@