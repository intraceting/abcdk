
## 第三方组件包所需要的环境变量。如果未指定或者为空，则以本机为目标平台。
```bash
export _PKG_TARGET_MACHINE="组件包的目标机器。x86_64-linux-gnu,aarch64-linux-gnu,arm-linux-gnu等。"
export _PKG_TARGET_WORDBIT="组件包的目标字宽。32或64。"
export _THIRDPARTY_PKG_PREFIX="组件包的安装路径。"
export _THIRDPARTY_PKG_FIND_MODE="PKG组件包搜索模式(only,both,default)。"
```
