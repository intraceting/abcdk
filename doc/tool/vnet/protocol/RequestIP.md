## 请求IP地址

请求：

|CMD     |IPV4TYPE |IPV4MASK |IPV4ADDR |IPV6TYPE |IPV6MASK |IPV6DDR  |
|:-------|:--------|:--------|:--------|:--------|:--------|:--------|
|2 bytes |1 type   |1 byte   |4 types  |1 byte   |1 type   |16 types |

* CMD：命令。1
* IPV4TYPE：IPV4类型。0 静态，1 动态。
* IPV4MASK：IPV4掩码。静态有效。
* IPV4ADDR：IPV4地址。静态有效。
* IPV6TYPE：IPV6类型。0 静态，1 动态。
* IPV6MASK：IPV6掩码。静态有效。
* IPV6ADDR：IPV6地址。静态有效。

应答：

|CMD     |ERRNO   |IPV4MASK |IPV4ADDR |IPV6MASK |IPV6ADDR |IPV4ADDR |IPV6ADDR |
|:-------|:-------|:--------|:--------|:--------|:--------|:--------|:--------|
|2 bytes |4 bytes |1 byte   |4 bytes  |1 type   |16 bytes |4 bytes  |16 bytes |

* CMD：命令。1
* ERRNO：出错码。0 无，1 拒绝访问，11 需要重试。
* IPV4MASK：IPV4掩码。
* IPV4ADDR：IPV4地址(本机)。
* IPV6MASK：IPV6掩码。
* IPV6ADDR：IPV6地址(本机)。
* IPV4ADDR：IPV4地址(网关)。
* IPV6ADDR：IPV6地址(网关)。