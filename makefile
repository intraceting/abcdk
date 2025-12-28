#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#
MAKEFILE_DIR := $(dir $(shell realpath "$(lastword $(MAKEFILE_LIST))"))

#
CONF_FILE ?= $(MAKEFILE_DIR)/build/makefile.conf

#加载配置项.
include ${CONF_FILE}

#
VERSION_MAJOR = 3
VERSION_MINOR = 7
VERSION_PATCH = 2

#
VERSION_STR_MAIN = ${VERSION_MAJOR}.${VERSION_MINOR}
VERSION_STR_FULL = ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}

#
C_FLAGS += -std=${C_STD} 
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
C_FLAGS += -DABCDK_VERSION_PATCH=${VERSION_PATCH}
C_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#
CXX_FLAGS += -std=${CXX_STD}
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
CXX_FLAGS += -DABCDK_VERSION_PATCH=${VERSION_PATCH}
CXX_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#在GCC中, 链接器按照从左到右的顺序解析库, 因此想让这个生效, 必须写在链接参数的第一个.
LD_FLAGS += -Wl,--as-needed

#
ifeq (${BUILD_TYPE},debug)
C_FLAGS += -g
CXX_FLAGS += -g
#生成调试信息.
NVCC_FLAGS += -g -G
endif
ifeq (${BUILD_TYPE},release)
C_FLAGS += -O2
CXX_FLAGS += -O2
LD_FLAGS += -s
endif

#
ifeq (${LSB_RELEASE},linux-gnu)
C_FLAGS += -D_GNU_SOURCE
CXX_FLAGS += -D_GNU_SOURCE
LD_FLAGS += -ldl -pthread -lc -lm -lrt
endif
ifeq (${LSB_RELEASE},android)
LD_FLAGS += -ldl -pthread -lc -lm
endif


#
C_FLAGS += -I$(MAKEFILE_DIR)/src/lib/include/ 
C_FLAGS += ${EXTRA_C_FLAGS}
#
CXX_FLAGS += -I$(MAKEFILE_DIR)/src/lib/include/ 
CXX_FLAGS += ${EXTRA_C_FLAGS} ${EXTRA_CXX_FLAGS}

#
LD_FLAGS += -L${BUILD_PATH}
LD_FLAGS += ${EXTRA_LD_FLAGS}

#把依赖组件路径拆分, 加上前缀用于搜所深度依赖项.
# aaa/bbb:ccc/ddd:eee/fff
# -Wl,-rpath-link=aaa/bbb -Wl,-rpath-link=ccc/ddd -Wl,-rpath-link=eee/fff
LD_FLAGS += $(foreach ONE_PATH,$(subst :, ,$(EXTRA_RPATH)),-Wl,-rpath-link=$(ONE_PATH))

 

#
OBJ_PATH = ${BUILD_PATH}/abcdk.obj.tmp/


#gcc临时文件目录.
export TMPDIR=${OBJ_PATH}/

#伪目标, 告诉make这些都是标志, 而不是实体目录.
#因为如果标签和目录同名, 而目录内的文件没有更新的情况下, 编译和链接会跳过.如："XXX is up to date".
.PHONY: compile clean install uninstall build help

#
all: compile

#加载子项目.
#顺序不能更换.
include $(MAKEFILE_DIR)/makefile.compile.mk

#
compile: lib tool test xgettext needed

#
clean: clean-lib clean-tool clean-test clean-xgettext clean-needed


#加载子项目.
#顺序不能更换.
include $(MAKEFILE_DIR)/makefile.setup.mk

#
install: install-lib install-dev install-tool install-pot

#
uninstall: uninstall-lib uninstall-dev uninstall-tool uninstall-pot


#加载子项目.
#顺序不能更换.
include $(MAKEFILE_DIR)/makefile.build.mk

#
build: build-deb build-rpm

#
help:
	@echo "make"
	@echo "make all"
	@echo "make clean"
	@echo "make install"
	@echo "make install-lib"
	@echo "make install-dev"
	@echo "make install-tool"
	@echo "make install-pot"
	@echo "make uninstall"
	@echo "make uninstall-lib"
	@echo "make uninstall-dev"
	@echo "make uninstall-tool"
	@echo "make uninstall-pot"
	@echo "make build"
	@echo "make build-deb"
	@echo "make build-rpm"

