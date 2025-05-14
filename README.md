
# 名称

ABCDK(A Better C language Development Kit)

## 版权

[英文](LICENSE)

## 简介

为了支持在Linux/Unix系统中快速开发软件项目而创建的解决方案，提供关于网络、数据库、链表、多叉树、硬盘、磁带、文件、目录、多媒体等开发接口。

## 主要模块

- 3rdparty 第三方组件
- share 共享文档
- src 源码目录
- tools 工具组件


## 查看编译帮助

```bash
$ ./configure.sh -h
$ make help
```
## 编译和安装


## 本地编译和安装

```bash
$ ./configure.sh [ ... ]
$ make
$ make install
```

## 交叉编译和安装

```bash
$ ./configure.sh -d COMPILER_PREFIX=/target_platform/bin/ -d COMPILER_C_NAME=gcc -d COMPILER_C_NAME=g++ [ ... ]
$ make
$ make install
```

