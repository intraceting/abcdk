#
# This file is part of ABCDK.
#
# MIT License
#
#

#
MAKE_CONF ?= $(abspath $(CURDIR)/../build/makefile.conf)

#加载配置项。
include ${MAKE_CONF}

#
SOLUTION_NAME ?= abcdk

#
PROJECT_NAME = abcdkcomm

#
ifeq (${VERSION_MAJOR},)
VERSION_MAJOR = 1
else ifeq (${VERSION_MAJOR},"")
VERSION_MAJOR = 1
endif

#
ifeq (${VERSION_MINOR},)
VERSION_MINOR = 0
else ifeq (${VERSION_MINOR},"")
VERSION_MINOR = 0
endif

#
ifeq (${BUILD_PATH},)
BUILD_PATH = $(abspath $(CURDIR)/../build/)
else ifeq (${BUILD_PATH},"")
BUILD_PATH = $(abspath $(CURDIR)/../build/)
endif

#
ifeq (${INSTALL_PREFIX},)
INSTALL_PREFIX = /usr/local/${SOLUTION_NAME}/
else ifeq (${INSTALL_PREFIX},"")
INSTALL_PREFIX = /usr/local/${SOLUTION_NAME}/
endif

#
ifeq (${ROOT_PATH},)
ROOT_PATH = /
else ifeq (${ROOT_PATH},"")
ROOT_PATH = /
endif

#
A_NAME = libabcdk-comm.a
A_REALNAME = ${A_NAME}.${VERSION_MAJOR}.${VERSION_MINOR}

#
SO_NAME = libabcdk-comm.so
SO_REALNAME = ${SO_NAME}.${VERSION_MAJOR}.${VERSION_MINOR}


#Archive
ARC = ar

#Compiler
CCC = gcc

#Standard
CCC_STD = -std=c11

#-Werror -Wall -Wextra
CCC_FLAGS += -fPIC -Wno-unused-result

#
LINK_FLAGS += -Wl,--as-needed -Wl,-rpath="./" -Wl,-rpath="${INSTALL_PREFIX}/lib/"

#
LINK_FLAGS += -Wl,--soname,${SO_NAME}

#
CCC_FLAGS += -DBUILD_VERSION_MAJOR=${VERSION_MAJOR} -DBUILD_VERSION_MINOR=${VERSION_MINOR} -DBUILD_VERSION_DATETIME=\"${VERSION_DATETIME}\"

#
ifeq (${BUILD_TYPE},debug)
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
LINK_FLAGS += -labcdk-util -L${BUILD_PATH} -Wl,-rpath-link=${BUILD_PATH} ${DEPEND_LIBS}

#
OBJ_PATH = ${BUILD_PATH}/${PROJECT_NAME}.o.tmp/

#
OBJ_FILES = \
	${OBJ_PATH}/mux.o


#
all: ${PROJECT_NAME}

#
${PROJECT_NAME}: $(OBJ_FILES)
	mkdir -p $(BUILD_PATH)
	rm -f $(BUILD_PATH)/${SO_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${SO_REALNAME} $^ $(LINK_FLAGS) -shared
	ln -f -s ${SO_REALNAME} $(BUILD_PATH)/${SO_NAME}
	rm -f $(BUILD_PATH)/${A_REALNAME}
	$(ARC) rcs $(BUILD_PATH)/${A_REALNAME} $^
	ln -f -s ${A_REALNAME} $(BUILD_PATH)/${A_NAME}
#
$(OBJ_PATH)/%.o: %.c
	mkdir -p $(OBJ_PATH)
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean:
	rm -rf $(OBJ_PATH)
	rm -f $(BUILD_PATH)/${SO_REALNAME}
	rm -f $(BUILD_PATH)/${SO_NAME}
	rm -f $(BUILD_PATH)/${A_REALNAME}
	rm -f $(BUILD_PATH)/${A_NAME}


#
INSTALL_PATH_INC = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/include/${PROJECT_NAME}/)
INSTALL_PATH_LIB = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/lib/)
INSTALL_PATH_BIN = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/bin/)

#
install:
#
	mkdir -p ${INSTALL_PATH_LIB}
	cp -f $(BUILD_PATH)/${SO_REALNAME} ${INSTALL_PATH_LIB}/
	ln -f -s ${SO_REALNAME} ${INSTALL_PATH_LIB}/${SO_NAME}
	cp -f $(BUILD_PATH)/${A_REALNAME} ${INSTALL_PATH_LIB}/
	ln -f -s ${A_REALNAME} ${INSTALL_PATH_LIB}/${A_NAME}

#
	mkdir -p ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/mux.h ${INSTALL_PATH_INC}/

#
uninstall:
#
	rm -f ${INSTALL_PATH_LIB}/${SO_REALNAME}
	rm -f ${INSTALL_PATH_LIB}/${SO_NAME}
	rm -f ${INSTALL_PATH_LIB}/${A_REALNAME}
	rm -f ${INSTALL_PATH_LIB}/${A_NAME}

#
	rm -f ${INSTALL_PATH_INC}/mux.h
