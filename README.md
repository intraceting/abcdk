
# ABCDK

A Better C language Development Kit. 

## 简介

为支持在gnu/linux平台中使用C/C++语言开发项目而创建，支持文件、网络、数据库、设备、图像等。

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

```bash
$ ./configure.sh [ ... ]
$ make
$ sudo make install
```

## 编译和打包

```bash
$ ./configure.sh [ ... ]
$ make
$ make package
```

