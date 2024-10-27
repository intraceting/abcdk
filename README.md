
# ABCDK

一个C语言的开发工具包。

## 简介

为支持在Linux平台中使用C/C++语言开发项目而创建。

## 主要模块

- auto 自动编译
- doc 文档说明显
- lib 基础组件
- script 一些脚本
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

