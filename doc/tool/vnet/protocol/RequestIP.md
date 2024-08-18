## 请求IP地址

请求：

|CMD     |IPV4TYPE |IPV4ADDR |IPV6TYPE |IPADDR   |
|:-------|:--------|:--------|:--------|:--------|
|2 bytes |1 type   |4 bytes  |1 type   |16 bytes |

* CMD：命令。1
* IPV4TYPE：IPV4类型。0 静态，1 动态。
* IPV4ADDR：IPV4地址。
* IPV6TYPE：IPV6类型。0 静态，1 动态。
* IPV6ADDR：IPV6地址。

应答：

|CMD     |ERRNO   |IPV4ADDR |IPV6ADDR |IPV4ADDR |IPV6ADDR |
|:-------|:-------|:--------|:--------|:--------|:--------|
|2 bytes |4 bytes |4 bytes  |16 bytes |4 bytes  |16 bytes |

* CMD：命令。1
* ERRNO：出错码。0 无，1 拒绝访问，11 需要重试。
* IPV4ADDR：IPV4地址(客户端)。
* IPV6ADDR：IPV6地址(客户端)。
* IPV4ADDR：IPV4地址(服务端)。
* IPV6ADDR：IPV6地址(服务端)。