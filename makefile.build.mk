#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#
# Makefile 所在目录（绝对路径）
MAKEFILE_DIRNAME := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

#
SRC_DIR := ${MAKEFILE_DIRNAME}/src/

#C
LIB_SRC_FILES += $(wildcard $(SRC_DIR)/lib/source/*/*.c)

#C++
LIB_SRC_CXX_FILES += $(wildcard $(SRC_DIR)/lib/source/*/*.cpp)

#CUDA是可选项，可能未启用。
ifeq (${HAVE_CUDA},yes)
LIB_SRC_CU_FILES += $(wildcard $(SRC_DIR)/lib/source/*/*.cu)
endif

#
TOOL_SRC_FILES = $(wildcard $(SRC_DIR)/tool/*.c)

#
TEST_SRC_FILES = $(wildcard $(SRC_DIR)/test/*.c)

#
LIB_OBJ_FILES := $(patsubst $(SRC_DIR)/lib/%, $(OBJ_PATH)/lib/%, $(LIB_SRC_FILES:.c=.o) $(LIB_SRC_CXX_FILES:.cpp=.o) $(LIB_SRC_CU_FILES:.cu=.o))
LIB_OBJ_DEPS += $(LIB_OBJ_FILES:.o=.d)

#
TOOL_OBJ_FILES := $(patsubst $(SRC_DIR)/tool/%, $(OBJ_PATH)/tool/%, $(TOOL_SRC_FILES:.c=.o))
TOOL_OBJ_DEPS += $(TOOL_OBJ_FILES:.o=.d)

#
TEST_OBJ_FILES := $(patsubst $(SRC_DIR)/test/%, $(OBJ_PATH)/test/%, $(TEST_SRC_FILES:.c=.o))
TEST_OBJ_DEPS += $(TEST_OBJ_FILES:.o=.d)


#
lib: lib-src
	mkdir -p $(BUILD_PATH)
	$(CXX) -shared -o $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL} $(LIB_OBJ_FILES) $(LD_FLAGS) -Wl,-soname,libabcdk.so.${VERSION_STR_MAIN}
	$(AR) -cr $(BUILD_PATH)/libabcdk.a $(LIB_OBJ_FILES)

#
lib-src: $(LIB_OBJ_FILES)

#
tool: tool-src lib
	mkdir -p $(BUILD_PATH)
	$(CXX) -o $(BUILD_PATH)/abcdk-tool ${TOOL_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
tool-src: ${TOOL_OBJ_FILES} 


#
test: test-src lib
	mkdir -p $(BUILD_PATH)
	$(CXX) -o $(BUILD_PATH)/abcdk-test ${TEST_OBJ_FILES} -l:libabcdk.a $(LD_FLAGS)

#
test-src: ${TEST_OBJ_FILES} 

# $@: 目标文件
# $<: 源文件
# 自动匹配多级路径.
$(OBJ_PATH)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

#
$(OBJ_PATH)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

#
$(OBJ_PATH)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(dir $@)
	rm -f $@
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@


#包含依赖文件(不能晚于此处).
-include $(LIB_OBJ_DEPS) 
-include $(TOOL_OBJ_DEPS)
-include $(TEST_OBJ_DEPS)

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


#
clean-lib:
	rm -rf ${OBJ_PATH}/lib
	rm -f $(BUILD_PATH)/libabcdk.so.${VERSION_STR_FULL}
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
clean-xgettext:
	rm -rf $(BUILD_PATH)/libabcdk.en_US.pot
	rm -rf $(BUILD_PATH)/abcdk-tool.en_US.pot

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: lib tool test 