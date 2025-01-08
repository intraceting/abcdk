
# 名称

ABCDK(一个更好的C语言开发包)

## 版权

[英文](LICENSE)

## 简介

为支持在Linux平台中辅助工作创建，以C语言作为主要编程语言的软件开发工具包，提供本机编译和交叉编译两种能力。 

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

