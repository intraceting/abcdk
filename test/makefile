#
# This file is part of ABCDK.
#
# MIT License
#
#


#
MAKE_CONF ?= $(abspath $(CURDIR)/../build/makefile.conf)

#
PROJECT_NAME = tests

#加载配置项。
include ${MAKE_CONF}

#
ifeq (${BUILD_PATH},)
BUILD_PATH = $(abspath $(CURDIR)/../build/)
else ifeq (${BUILD_PATH},"")
BUILD_PATH = $(abspath $(CURDIR)/../build/)
endif

#
#Compiler
CCC = gcc

# 
CCC_STD = -std=c11

#-Werror -Wall -Wextra
CCC_FLAGS += -fPIC -Wno-unused-result

#
LINK_FLAGS += -Wl,--as-needed -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"

#
CCC_FLAGS += -DBUILD_VERSION_MAJOR=${VERSION_MAJOR} -DBUILD_VERSION_MINOR=${VERSION_MINOR} -DBUILD_VERSION_DATETIME=\"${VERSION_DATETIME}\"

#
ifeq ($(BUILD_TYPE),debug)
CCC_FLAGS += -g
else 
CCC_FLAGS += -s -O2
endif

#可能在交叉编译环中。
ifneq ($(TARGET_PLATFORM),$(HOST_PLATFORM))
    CCC = $(TARGET_PLATFORM)-linux-gnu-gcc
endif

#
CCC_FLAGS += -I$(abspath $(CURDIR)/../) ${DEPEND_FLAGS}

#
LINK_FLAGS += -labcdk-comm -labcdk-util -L${BUILD_PATH} -Wl,-rpath-link=${BUILD_PATH} ${DEPEND_LIBS}
#LINK_FLAGS += -l:libabcdkutil.a -L${BUILD_PATH} -Wl,-rpath-link=${BUILD_PATH} ${DEPEND_LIBS}

#
OBJ_PATH = ${BUILD_PATH}/${PROJECT_NAME}.o.tmp/

#
all: ${PROJECT_NAME}

#
${PROJECT_NAME}: \
	$(BUILD_PATH)/util_test.exe \
	$(BUILD_PATH)/comm_test.exe

#
$(BUILD_PATH)/%.exe: ${OBJ_PATH}/%.o
	mkdir -p $(BUILD_PATH)
	rm -f $@
	$(CCC) -o "$@" $^ $(LINK_FLAGS)

#
$(OBJ_PATH)/%.o: %.c
	mkdir -p $(OBJ_PATH)
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean:
	rm -rf $(OBJ_PATH)
	rm -f $(BUILD_PATH)/util_test.exe
	rm -f $(BUILD_PATH)/comm_test.exe
	

