
# ABCDK

一个C语言的开发工具包。

## 简介

为支持在GNU/Linux平台中使用C/C++语言开发项目而创建，支持文件、网络、数据库、设备、图像等。

## 主要模块

- asio 异步IO
- audio 音频
- curl CURL二次封装
- database 数据库(unixodbc,sqlite,redis)
- ffmpeg FFMPEG二次封装
- http HTTP
- image 图像(freeimage)
- json JSON(json-c)
- log 日志
- mp4 MP4
- rtp RTP
- sdp SDP
- shell 外部命令二次封装
- ssl SSL套件(openssl)
- util 基础工具
- vidoe 视频

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

