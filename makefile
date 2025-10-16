#
# This file is part of ABCDK.
#
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/build/makefile.conf)

# 加载配置项。
include ${MAKE_CONF}

#主版本
VERSION_MAJOR = 3

#副版本
VERSION_MINOR = 6

#发行版本
VERSION_RELEASE = 1

#
VERSION_STR_MAIN = ${VERSION_MAJOR}.${VERSION_MINOR}
VERSION_STR_FULL = ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}


#gcc临时文件目录。
export TMPDIR=${BUILD_PATH}/

#
C_FLAGS += -std=${STD_C} 
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
C_FLAGS += -DABCDK_VERSION_RELEASE=${VERSION_RELEASE}
C_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#
CXX_FLAGS += -std=${STD_CXX}
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
CXX_FLAGS += -DABCDK_VERSION_RELEASE=${VERSION_RELEASE}
CXX_FLAGS += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

#
C_FLAGS += -I${INSTALL_PREFIX}/include
C_FLAGS += ${DEPEND_FLAGS}
#
CXX_FLAGS += -I${INSTALL_PREFIX}/include
CXX_FLAGS += ${DEPEND_FLAGS}

#
NVCC_FLAGS += -std=${STD_CXX}
NVCC_FLAGS += -ccbin ${CXX}
#抑制“未识别的属性”的诊断消息输出，让编译日志更简洁。
NVCC_FLAGS += -Xcudafe --diag_suppress=unrecognized_attribute

#在GCC中，链接器按照从左到右的顺序解析库，因此想让这个生效，必须写在链接参数的第一个。
LD_FLAGS += -Wl,--as-needed
#
LD_FLAGS += -L${INSTALL_PREFIX}/lib -Wl,-rpath-link=${INSTALL_PREFIX}/lib
#
LD_FLAGS += ${DEPEND_LINKS}

#
ifeq (${BUILD_TYPE},debug)
C_FLAGS += -g
CXX_FLAGS += -g
#生成调试信息。
NVCC_FLAGS += -g -G
endif
ifeq (${BUILD_TYPE},release)
#仅保留用于性能分性的调式信息。
NVCC_FLAGS += -lineinfo
LD_FLAGS += -s
endif

#
ifeq (${OPTIMIZE_LEVEL},1)
C_FLAGS += -O1
CXX_FLAGS += -O1
endif
ifeq (${OPTIMIZE_LEVEL},2)
C_FLAGS += -O2
CXX_FLAGS += -O2
endif
ifeq (${OPTIMIZE_LEVEL},3)
C_FLAGS += -O3
CXX_FLAGS += -O3
endif
ifeq (${OPTIMIZE_LEVEL},s)
C_FLAGS += -Os
CXX_FLAGS += -Os
endif
ifeq (${OPTIMIZE_LEVEL},fast)
C_FLAGS += -Ofast
CXX_FLAGS += -Ofast
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

#在使用对应的编译器链接时，下面的动态链接库会自动进行链接。
#-lstdc++ -lstdgcc_s

#
C_FLAGS += -I$(CURDIR)/src/lib/include/ 
C_FLAGS += ${EXTRA_C_FLAGS}
#
CXX_FLAGS += -I$(CURDIR)/src/lib/include/ 
CXX_FLAGS += ${EXTRA_CXX_FLAGS}

#C++编译选项绑定到CUDA编译选项。
NVCC_FLAGS += $(addprefix -Xcompiler ,${CXX_FLAGS})

#
LD_FLAGS += -L${BUILD_PATH}
#把依赖组件路径拆分，加上前缀用于搜所深度依赖项。
# aaa/bbb:ccc/ddd:eee/fff
# -Wl,-rpath-link=aaa/bbb -Wl,-rpath-link=ccc/ddd -Wl,-rpath-link=eee/fff
LD_FLAGS += $(foreach DE_LIB_PATH,$(subst :, ,$(DEPEND_LIB_PATH)),-Wl,-rpath-link=$(DE_LIB_PATH))
#
LD_FLAGS += ${EXTRA_LD_FLAGS}
 

#
OBJ_PATH = ${BUILD_PATH}/obj/



#
all: build

#加载子项目。
#顺序不能更换。
include $(CURDIR)/makefile.build.mk

#
build: lib tool test xgettext

#
clean: clean-lib clean-tool clean-test clean-xgettext


#加载子项目。
#顺序不能更换。
include $(CURDIR)/makefile.setup.mk

#
install: install-lib install-dev install-tool

#
uninstall: uninstall-lib uninstall-dev uninstall-tool

#
help:
	@echo "make"
	@echo "make all"
	@echo "make clean"
	@echo "make install"
	@echo "make install-lib"
	@echo "make install-dev"
	@echo "make install-tool"
	@echo "make uninstall"
	@echo "make uninstall-lib"
	@echo "make uninstall-dev"
	@echo "make uninstall-tool"


#伪目标，告诉make这些都是标志，而不是实体目录。
#因为如果标签和目录同名，而目录内的文件没有更新的情况下，编译和链接会跳过。如："XXX is up to date"。
.PHONY: build