#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
TEST_SRC_FILES = $(wildcard src/test/*.c)
TEST_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TEST_SRC_FILES}))

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: test test-src

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
