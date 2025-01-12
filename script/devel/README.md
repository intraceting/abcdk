
## 第三方组件包所需要的环境变量。如果未指定或者为空，则以本机为目标平台。
```bash
export _THIRDPARTY_PREFIX="组件包的安装路径。"
export _THIRDPARTY_MACHINE="组件包的目标机器。x86_64-linux-gnu,aarch64-linux-gnu,arm-linux-gnu等。"
export _THIRDPARTY_BITWIDE="组件包的目标位宽。32或64。"
export _THIRDPARTY_PKG_CONFIG_PREFIX="PKG组件包前缀。"
export _THIRDPARTY_PKG_CONFIG_LIBDIR="PKG组件包搜索路径(覆盖)。"
export _THIRDPARTY_PKG_CONFIG_PATH="PKG组件包搜索路径(添加)。"
```
