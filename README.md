
# 名称

ABCDK(A Better C language Development Kit)

## 版权

[英文](LICENSE)

## 简介

为了支持在Linux/Unix系统中快速开发软件项目而创建的解决方案，提供关于网络、数据库、链表、多叉树、硬盘、磁带、文件、目录、多媒体等开发接口。

## 主要模块

- 3rdparty 第三方组件
- script 脚本组件
- share 共享文档
- src 源码目录


## 查看编译帮助

```bash
$ ../aconf/configure.sh -h
$ make help
```
## 编译和安装


## 本地编译和打包

```bash
$ ../aconf/configure.sh [ ... ]
$ make
$ make package
```

## 交叉编译和打包

```bash
$ ../aconf/configure.sh -d COMPILER_PREFIX=/target_platform/bin/ -d COMPILER_NAME=gcc [ ... ]
$ make
$ make package
```

