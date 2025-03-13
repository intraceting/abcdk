#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
TOOL_SRC_FILES = $(wildcard src/tool/*.c)
TOOL_OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,${TOOL_SRC_FILES}))

#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: tool-xgettext tool too-src

#把POT文件从share目录复制到build目录进行更新。
tool-xgettext:
	@if [ -x "${XGETTEXT}" ]; then \
		cp -f $(CURDIR)/share/locale/en_US/gettext/tool.pot $(BUILD_PATH)/tool.en_US.pot ; \
		find $(CURDIR)/src/tool/ -iname "*.c" -o -iname "*.cpp" > $(BUILD_PATH)/tool.gettext.filelist.txt ; \
		${XGETTEXT} --force-po --no-wrap --no-location --join-existing --package-name=ABCDK --package-version=${VERSION_STR_FULL} -o $(BUILD_PATH)/tool.en_US.pot --from-code=UTF-8 --keyword=TT -f $(BUILD_PATH)/tool.gettext.filelist.txt -L c++ ; \
		rm -f $(BUILD_PATH)/tool.gettext.filelist.txt ; \
		echo "'$(BUILD_PATH)/tool.en_US.pot' Update completed." ; \
	fi

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

