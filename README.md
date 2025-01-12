
# 名称

ABCDK(A Better C language Development Kit)

## 版权

[英文](LICENSE)

## 简介

ABCDK为了支持在Linux/Unix系统中快速开发软件项目而创建的解决方案，提供关于网络、数据库、链表、多叉树、硬盘、磁带、文件、目录、多媒体等开发接口。

## 主要模块

- auto 自动编译
- doc 文档说明
- lib 基础组件
- test 测试组件
- tool 工具组件

## 拉取子项目

```bash
$ git submodule update --init --remote  --force  --merge --recursive
```

## 查看编译帮助

```bash
$ ./configure.sh -h
$ make help
```
## 编译和安装


## 本地编译和打包

```bash
$ ./configure.sh [ ... ]
$ make
$ make package
```

## 交叉编译和打包

```bash
$ ./configure.sh -e COMPILER_PREFIX=/target_platform/bin/ -e COMPILER_NAME=gcc [ ... ]
$ make
$ make package
```

