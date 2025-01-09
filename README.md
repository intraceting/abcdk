
# 名称

ABCDK(一个较好的C语言开发的工具包)

## 版权

[英文](LICENSE)

## 简介

为支持在Linux平台中辅助工作创建，以C语言作为主要编程语言的软件开发工具包。 

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
$ ./configure.sh -e CC=/target_platform/bin/gcc -e AR=/target_platform/bin/ar [ ... ]
$ make
$ make package
```

