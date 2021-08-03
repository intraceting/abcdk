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
PROJECT_NAME = tools

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
MT_REALNAME = abcdk-mt.exe
MT_NAME = abcdk-mt

#
MTX_REALNAME = abcdk-mtx.exe
MTX_NAME = abcdk-mtx

#
LSB_REALNAME = abcdk-lsb_release.exe
LSB_NAME = abcdk-lsb_release

#
ODBC_REALNAME = abcdk-odbc_login.exe
ODBC_NAME = abcdk-odbc_login

#
CRAWLER_REALNAME = abcdk-crawler.exe
CRAWLER_NAME = abcdk-crawler

#
ROBOTS_REALNAME = abcdk-robots.exe
ROBOTS_NAME = abcdk-robots


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
LINK_FLAGS += -labcdk-util -L${BUILD_PATH} -Wl,-rpath-link=${BUILD_PATH} ${DEPEND_LIBS}

#
OBJ_PATH = ${BUILD_PATH}/${PROJECT_NAME}.o.tmp/

#
all: ${PROJECT_NAME}

#
OBJ_FILES = \
	${OBJ_PATH}/mtx.o \
	${OBJ_PATH}/mt.o \
	${OBJ_PATH}/lsb_release.o \
	${OBJ_PATH}/odbc_login.o \
	${OBJ_PATH}/crawler.o \
	${OBJ_PATH}/robots.o 

#
${PROJECT_NAME}:${OBJ_FILES}
	#
	mkdir -p $(BUILD_PATH)
	
	#
	rm -f $(BUILD_PATH)/${MTX_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${MTX_REALNAME} ${OBJ_PATH}/mtx.o $(LINK_FLAGS)
	ln -f -s ${MTX_REALNAME} $(BUILD_PATH)/${MTX_NAME}

	#
	rm -f $(BUILD_PATH)/${MT_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${MT_REALNAME} ${OBJ_PATH}/mt.o $(LINK_FLAGS)
	ln -f -s ${MT_REALNAME} $(BUILD_PATH)/${MT_NAME}

	#
	rm -f $(BUILD_PATH)/${LSB_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${LSB_REALNAME} ${OBJ_PATH}/lsb_release.o $(LINK_FLAGS)
	ln -f -s ${LSB_REALNAME} $(BUILD_PATH)/${LSB_NAME}

	#
	rm -f $(BUILD_PATH)/${ODBC_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${ODBC_REALNAME} ${OBJ_PATH}/odbc_login.o $(LINK_FLAGS)
	ln -f -s ${ODBC_REALNAME} $(BUILD_PATH)/${ODBC_NAME}

	#
	rm -f $(BUILD_PATH)/${CRAWLER_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${CRAWLER_REALNAME} ${OBJ_PATH}/crawler.o $(LINK_FLAGS)
	ln -f -s ${CRAWLER_REALNAME} $(BUILD_PATH)/${CRAWLER_NAME}

	#
	rm -f $(BUILD_PATH)/${ROBOTS_REALNAME}
	$(CCC) -o $(BUILD_PATH)/${ROBOTS_REALNAME} ${OBJ_PATH}/robots.o $(LINK_FLAGS)
	ln -f -s ${ROBOTS_REALNAME} $(BUILD_PATH)/${ROBOTS_NAME}

#
$(OBJ_PATH)/%.o: %.c
	mkdir -p $(OBJ_PATH)
	rm -f $@
	$(CCC) $(CCC_STD) $(CCC_FLAGS) -c $< -o "$@"

#
clean:
	rm -rf $(OBJ_PATH)
	rm -f $(BUILD_PATH)/${MTX_REALNAME}
	rm -f $(BUILD_PATH)/${MTX_NAME}
	rm -f $(BUILD_PATH)/${MT_REALNAME}
	rm -f $(BUILD_PATH)/${MT_NAME}
	rm -f $(BUILD_PATH)/${LSB_REALNAME}
	rm -f $(BUILD_PATH)/${LSB_NAME}
	rm -f $(BUILD_PATH)/${ODBC_REALNAME}
	rm -f $(BUILD_PATH)/${ODBC_NAME}
	rm -f $(BUILD_PATH)/${CRAWLER_REALNAME}
	rm -f $(BUILD_PATH)/${CRAWLER_NAME}
	rm -f $(BUILD_PATH)/${ROBOTS_REALNAME}
	rm -f $(BUILD_PATH)/${ROBOTS_NAME}

#
INSTALL_PATH_BIN = $(abspath ${ROOT_PATH}/${INSTALL_PREFIX}/bin/)


#
install:
#
	mkdir -p ${INSTALL_PATH_BIN}
	cp -f -f $(BUILD_PATH)/${MTX_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${MTX_REALNAME} $(INSTALL_PATH_BIN)/${MTX_NAME}
	cp -f -f $(BUILD_PATH)/${MT_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${MT_REALNAME} $(INSTALL_PATH_BIN)/${MT_NAME}
	cp -f -f $(BUILD_PATH)/${LSB_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${LSB_REALNAME} $(INSTALL_PATH_BIN)/${LSB_NAME}
	cp -f -f $(BUILD_PATH)/${ODBC_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${ODBC_REALNAME} $(INSTALL_PATH_BIN)/${ODBC_NAME}
	cp -f -f $(BUILD_PATH)/${CRAWLER_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${CRAWLER_REALNAME} $(INSTALL_PATH_BIN)/${CRAWLER_NAME}
	cp -f -f $(BUILD_PATH)/${ROBOTS_REALNAME} ${INSTALL_PATH_BIN}/
	ln -f -s ${ROBOTS_REALNAME} $(INSTALL_PATH_BIN)/${ROBOTS_NAME}

#
uninstall:
#
	rm -f $(INSTALL_PATH_BIN)/${MTX_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${MTX_NAME}
	rm -f $(INSTALL_PATH_BIN)/${MT_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${MT_NAME}
	rm -f $(INSTALL_PATH_BIN)/${LSB_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${LSB_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ODBC_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${ODBC_NAME}
	rm -f $(INSTALL_PATH_BIN)/${CRAWLER_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${CRAWLER_NAME}
	rm -f $(INSTALL_PATH_BIN)/${ROBOTS_REALNAME}
	rm -f $(INSTALL_PATH_BIN)/${ROBOTS_NAME}
