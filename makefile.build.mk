#
# This file is part of ABCDK.
#
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
#
#
#MAKEFILE_DIR := $(dir $(shell realpath "$(lastword $(MAKEFILE_LIST))"))

#
SRC_DIR := $(MAKEFILE_DIR)/src/

#C
LIB_SRC_FILES += $(wildcard $(SRC_DIR)/lib/source/*/*.c)

#C++
LIB_SRC_CXX_FILES += $(wildcard $(SRC_DIR)/lib/source/*/*.cpp)

#
TOOL_SRC_FILES = $(wildcard $(SRC_DIR)/tool/*.c)

#
TEST_SRC_FILES = $(wildcard $(SRC_DIR)/test/*.c)

#
LIB_OBJ_FILES := $(patsubst $(SRC_DIR)/lib/%, $(OBJ_PATH)/lib/%, $(LIB_SRC_FILES:.c=.o) $(LIB_SRC_CXX_FILES:.cpp=.o))
LIB_OBJ_DEPS += $(LIB_OBJ_FILES:.o=.d)

#
TOOL_OBJ_FILES := $(patsubst $(SRC_DIR)/tool/%, $(OBJ_PATH)/tool/%, $(TOOL_SRC_FILES:.c=.o))
TOOL_OBJ_DEPS += $(TOOL_OBJ_FILES:.o=.d)

#
TEST_OBJ_FILES := $(patsubst $(SRC_DIR)/test/%, $(OBJ_PATH)/test/%, $(TEST_SRC_FILES:.c=.o))
TEST_OBJ_DEPS += $(TEST_OBJ_FILES:.o=.d)


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
	$(NVCC) $(NVCC_FLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@


#包含依赖文件(不能晚于此处).
-include $(LIB_OBJ_DEPS) 
-include $(TOOL_OBJ_DEPS)
-include $(TEST_OBJ_DEPS)

#
xgettext: xgettext-lib xgettext-tool

#
xgettext-lib:
	${SHELLKITS_HOME}/tools/xgettext.sh ABCDK ${VERSION_STR_FULL} ABCDK_GETTEXT $(MAKEFILE_DIR)/src/lib/ $(BUILD_PATH)/libabcdk.pot

#
xgettext-tool:
	${SHELLKITS_HOME}/tools/xgettext.sh ABCDK ${VERSION_STR_FULL} ABCDK_GETTEXT $(MAKEFILE_DIR)/src/tool/ $(BUILD_PATH)/abcdk-tool.pot

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
	rm -rf $(BUILD_PATH)/libabcdk.pot
	rm -rf $(BUILD_PATH)/abcdk-tool.pot
