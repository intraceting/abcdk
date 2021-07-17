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
PROJECT_NAME = abcdkutil

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
A_NAME = libabcdk-util.a
A_REALNAME = ${A_NAME}.${VERSION_MAJOR}.${VERSION_MINOR}

#
SO_NAME = libabcdk-util.so
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
LINK_FLAGS += -Wl,--as-needed -Wl,-rpath="./"  -Wl,-rpath="${INSTALL_PREFIX}/lib/"

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
CCC_FLAGS += ${DEPEND_FLAGS}

#
LINK_FLAGS += ${DEPEND_LIBS}
 
#
OBJ_PATH = ${BUILD_PATH}/${PROJECT_NAME}.o.tmp/

#
#OBJ_FILES = $(addprefix ${OBJ_PATH}/,$(patsubst %.c,%.o,$(wildcard *.c)))

#
OBJ_FILES = \
	${OBJ_PATH}/general.o \
	${OBJ_PATH}/thread.o \
	${OBJ_PATH}/signal.o \
	${OBJ_PATH}/crc32.o \
	${OBJ_PATH}/base64.o \
	${OBJ_PATH}/clock.o \
	${OBJ_PATH}/geometry.o \
	${OBJ_PATH}/allocator.o \
	${OBJ_PATH}/mman.o \
	${OBJ_PATH}/buffer.o \
	${OBJ_PATH}/pool.o \
	${OBJ_PATH}/tree.o \
	${OBJ_PATH}/map.o \
	${OBJ_PATH}/option.o \
	${OBJ_PATH}/getargs.o \
	${OBJ_PATH}/dirent.o \
	${OBJ_PATH}/notify.o \
	${OBJ_PATH}/iconv.o \
	${OBJ_PATH}/socket.o \
	${OBJ_PATH}/epoll.o \
	${OBJ_PATH}/scsi.o \
	${OBJ_PATH}/mtx.o \
	${OBJ_PATH}/mt.o \
	${OBJ_PATH}/blockio.o \
	${OBJ_PATH}/tar.o \
	${OBJ_PATH}/termios.o \
	${OBJ_PATH}/sqlite.o \
	${OBJ_PATH}/odbc.o \
	${OBJ_PATH}/openssl.o \
	${OBJ_PATH}/ffmpeg.o \
	${OBJ_PATH}/bmp.o \
	${OBJ_PATH}/freeimage.o \
	${OBJ_PATH}/uri.o \
	${OBJ_PATH}/html.o \
	${OBJ_PATH}/robots.o


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
	cp  -f $(CURDIR)/allocator.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/atomic.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/base64.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/blockio.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/bmp.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/buffer.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/clock.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/crc32.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/dirent.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/defs.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/epoll.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/ffmpeg.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/freeimage.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/general.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/geometry.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/getargs.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/html.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/iconv.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/odbc.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/map.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/mman.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/mt.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/mtx.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/notify.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/option.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/pool.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/robots.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/scsi.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/signal.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/socket.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/sqlite.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/openssl.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/tar.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/termios.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/thread.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/tree.h ${INSTALL_PATH_INC}/
	cp  -f $(CURDIR)/uri.h ${INSTALL_PATH_INC}/

#
uninstall:
#
	rm -f ${INSTALL_PATH_LIB}/${SO_REALNAME}
	rm -f ${INSTALL_PATH_LIB}/${SO_NAME}
	rm -f ${INSTALL_PATH_LIB}/${A_REALNAME}
	rm -f ${INSTALL_PATH_LIB}/${A_NAME}

#
	rm -f ${INSTALL_PATH_INC}/allocator.h
	rm -f ${INSTALL_PATH_INC}/atomic.h
	rm -f ${INSTALL_PATH_INC}/base64.h
	rm -f ${INSTALL_PATH_INC}/blockio.h
	rm -f ${INSTALL_PATH_INC}/bmp.h
	rm -f ${INSTALL_PATH_INC}/buffer.h
	rm -f ${INSTALL_PATH_INC}/clock.h
	rm -f ${INSTALL_PATH_INC}/crc32.h
	rm -f ${INSTALL_PATH_INC}/defs.h
	rm -f ${INSTALL_PATH_INC}/dirent.h
	rm -f ${INSTALL_PATH_INC}/epoll.h
	rm -f ${INSTALL_PATH_INC}/ffmpeg.h
	rm -f ${INSTALL_PATH_INC}/freeimage.h
	rm -f ${INSTALL_PATH_INC}/general.h
	rm -f ${INSTALL_PATH_INC}/geometry.h
	rm -f ${INSTALL_PATH_INC}/getargs.h
	rm -f ${INSTALL_PATH_INC}/html.h
	rm -f ${INSTALL_PATH_INC}/iconv.h
	rm -f ${INSTALL_PATH_INC}/odbc.h
	rm -f ${INSTALL_PATH_INC}/map.h
	rm -f ${INSTALL_PATH_INC}/mman.h
	rm -f ${INSTALL_PATH_INC}/mt.h
	rm -f ${INSTALL_PATH_INC}/mtx.h
	rm -f ${INSTALL_PATH_INC}/notify.h
	rm -f ${INSTALL_PATH_INC}/option.h
	rm -f ${INSTALL_PATH_INC}/pool.h
	rm -f ${INSTALL_PATH_INC}/robots.h
	rm -f ${INSTALL_PATH_INC}/scsi.h
	rm -f ${INSTALL_PATH_INC}/signal.h
	rm -f ${INSTALL_PATH_INC}/socket.h
	rm -f ${INSTALL_PATH_INC}/sqlite.h
	rm -f ${INSTALL_PATH_INC}/openssl.h
	rm -f ${INSTALL_PATH_INC}/tar.h
	rm -f ${INSTALL_PATH_INC}/termios.h
	rm -f ${INSTALL_PATH_INC}/thread.h
	rm -f ${INSTALL_PATH_INC}/tree.h
	rm -f ${INSTALL_PATH_INC}/uri.h
