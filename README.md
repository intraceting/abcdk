
## ABCDK

* A better c development kit. 

### License

* ABCDK is MIT-licensed.

### 查看编译帮助

```bash
configure.sh -?
```
### 编译和安装

```bash
configure.sh [ ... ]
make
sudo make install
```

### 编译和打包

```bash
configure.sh [ ... ]
make
sudo make package
```

### 目录说明

- abcdkcomm 网络通讯。
    - mux 多路复用器，如：支持单连多任务模式。
- abcdkutil 基础工具。
    - allocator 带引用计数器的内存块。
    - atomic 原子操作。
    - base64 BASE64编解码。
    - blockio 块对齐的IO接口，如：磁带、或tar格式文件。
    - bmp BMP文件存取。
    - buffer 缓存。
    - clock 计时器。
    - crc32 CRC32计算。
    - defs 引用的文件、常用的宏定义。
    - dirent 目录扫描和统计。
    - epoll EPOLL扩展。
    - ffmpeg FFMPEG扩展。
    - freeimage FREEIMAGE扩展。
    - general 内存、文件、字符串等。
    - geometry 关于几何的几个方法。
    - getargs 参数解析，如：命令行参数。
    - html HTML解析。
    - iconv ICONV扩展。
    - map 一个简单的key-value容器。
    - mman 文件的内存眏射。
    - mt 磁带，如：LTO。
    - mtx 磁带库，如：IBM、Quantum、DELL等。
    - notify 文件变更监视。
    - odbc unixODBC扩展。
    - openssl OPENSSL扩展。
    - option 配置管理。
    - pool 环形池。
    - robots ROBOTS解析。
    - scsi 常用的SCSI指令，如：磁带、磁带库。
    - signal 简单的信号处理。
    - socket SOCKET扩展。
    - sqlite SQLITE扩展。
    - tar TAR读写。
    - termios 终端输入输出。
    - thread 线程、互斥量、事件等。
    - tree 一个简单的树结构容器。
    - uri URL解析。
- tests 测试样例。
    - comm_test 网络通讯测试。
    - util_test 常用工具测试。
- tools 自定义工具。
    - crawler 静态的HTML解析，如：分析URL。
    - lsb_release 系统产品信息查看。
    - mt 磁带操作。
    - mtx 磁带库操作。
    - odbc_login ODBC连接测试。
    - robots 静态的ROBOTS规则过滤。