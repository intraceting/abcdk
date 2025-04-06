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
LIB_SRC_FILES += $(wildcard src/lib/source/torch_host/*.c)
LIB_SRC_FILES += $(wildcard src/lib/source/torch_cuda/*.c)
LIB_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${LIB_SRC_FILES}))

#C++
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/rtsp/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch_host/*.cpp)
LIB_SRC_CXX_FILES += $(wildcard src/lib/source/torch_cuda/*.cpp)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cpp,%.cpp.o,${LIB_SRC_CXX_FILES}))

#CUDA是可选项，可能未启用。
ifneq ($(strip $(NVCC)),)
LIB_SRC_CU_FILES += $(wildcard src/lib/source/torch_cuda/*.cu)
LIB_OBJ_FILES += $(addprefix ${OBJ_PATH}/,$(patsubst %.cu,%.cu.o,${LIB_SRC_CU_FILES}))
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
	$(CC) -shared -o $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} $(LIB_OBJ_FILES) $(LD_FLAGS) -Wl,-soname,libabcdk.so.${VERSION_STR_MAIN}
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
lib-src: $(LIB_OBJ_FILES)


#
$(OBJ_PATH)/src/lib/source/util/%.o: src/lib/source/util/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/util/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/system/%.o: src/lib/source/system/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/system/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/mp4/%.o: src/lib/source/mp4/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/mp4/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/net/%.o: src/lib/source/net/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/net/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@


#
$(OBJ_PATH)/src/lib/source/ffmpeg/%.o: src/lib/source/ffmpeg/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/ffmpeg/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/redis/%.o: src/lib/source/redis/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/redis/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/sqlite/%.o: src/lib/source/sqlite/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/sqlite/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/odbc/%.o: src/lib/source/odbc/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/odbc/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/openssl/%.o: src/lib/source/openssl/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/openssl/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/lz4/%.o: src/lib/source/lz4/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/lz4/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/json/%.o: src/lib/source/json/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/json/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/curl/%.o: src/lib/source/curl/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/curl/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@
	
#
$(OBJ_PATH)/src/lib/source/rtsp/%.cpp.o: src/lib/source/rtsp/%.cpp
	mkdir -p $(OBJ_PATH)/src/lib/source/rtsp/
	rm -f $@
	$(CC) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/torch_host/%.o: src/lib/source/torch_host/%.c
	mkdir -p $(OBJ_PATH)/src/lib/source/torch_host/
	rm -f $@
	$(CC) -std=c99  $(C_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/torch_host/%.cpp.o: src/lib/source/torch_host/%.cpp
	mkdir -p $(OBJ_PATH)/src/lib/source/torch_host/
	rm -f $@
	$(CC) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/torch_cuda/%.cu.o: src/lib/source/torch_cuda/%.cu
	mkdir -p $(OBJ_PATH)/src/lib/source/torch_cuda/
	rm -f $@
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -Xcompiler -std=c++11  -c $< -o $@

#
$(OBJ_PATH)/src/lib/source/torch_cuda/%.cpp.o: src/lib/source/torch_cuda/%.cpp
	mkdir -p $(OBJ_PATH)/src/lib/source/torch_cuda/
	rm -f $@
	$(CC) -std=c++11 $(CXX_FLAGS) -c $< -o $@

#
clean-lib:
	rm -rf ${OBJ_PATH}/src/lib
	rm -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL}
	rm -f $(BUILD_PATH)/libabcdk.a


#
tool: tool-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-tool ${TOOL_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
tool-src: ${TOOL_OBJ_FILES} 

#
$(OBJ_PATH)/src/tool/%.o: src/tool/%.c
	mkdir -p $(OBJ_PATH)/src/tool/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@

#
clean-tool:
	rm -rf ${OBJ_PATH}/src/tool
	rm -f $(BUILD_PATH)/abcdk-tool


#
test: test-src lib
	mkdir -p $(BUILD_PATH)
	$(CC) -o $(BUILD_PATH)/abcdk-test ${TEST_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

#
$(OBJ_PATH)/src/test/%.o: src/test/%.c
	mkdir -p $(OBJ_PATH)/src/test/
	rm -f $@
	$(CC) -std=c99 $(C_FLAGS) -c $< -o $@
#
clean-test:
	rm -rf ${OBJ_PATH}/src/test
	rm -f $(BUILD_PATH)/abcdk-test

#
xgettext: xgettext-lib xgettext-tool

#把POT文件从share目录复制到build目录进行更新。
xgettext-lib:
	@if [ -x "${XGETTEXT}" ]; then \
		cp -f $(CURDIR)/share/lib/locale/en_US/gettext/lib.pot $(BUILD_PATH)/lib.en_US.pot ; \
		find $(CURDIR)/src/lib/ -iname "*.c" -o -iname "*.cpp" -o -iname "*.hxx" -o -iname "*.cu" > $(BUILD_PATH)/lib.gettext.filelist.txt ; \
		${XGETTEXT} --force-po --no-wrap --no-location --join-existing --package-name=ABCDK --package-version=${VERSION_STR_FULL} -o $(BUILD_PATH)/lib.en_US.pot --from-code=UTF-8 --keyword=TT -f $(BUILD_PATH)/lib.gettext.filelist.txt -L c++ ; \
		rm -f $(BUILD_PATH)/lib.gettext.filelist.txt ; \
		echo "'$(BUILD_PATH)/lib.en_US.pot' Update completed." ; \
	fi


#把POT文件从share目录复制到build目录进行更新。
xgettext-tool:
	@if [ -x "${XGETTEXT}" ]; then \
		cp -f $(CURDIR)/share/tool/locale/en_US/gettext/tool.pot $(BUILD_PATH)/tool.en_US.pot ; \
		find $(CURDIR)/src/tool/ -iname "*.c" -o -iname "*.cpp" -o -iname "*.hxx" > $(BUILD_PATH)/tool.gettext.filelist.txt ; \
		${XGETTEXT} --force-po --no-wrap --no-location --join-existing --package-name=ABCDK --package-version=${VERSION_STR_FULL} -o $(BUILD_PATH)/tool.en_US.pot --from-code=UTF-8 --keyword=TT -f $(BUILD_PATH)/tool.gettext.filelist.txt -L c++ ; \
		rm -f $(BUILD_PATH)/tool.gettext.filelist.txt ; \
		echo "'$(BUILD_PATH)/tool.en_US.pot' Update completed." ; \
	fi