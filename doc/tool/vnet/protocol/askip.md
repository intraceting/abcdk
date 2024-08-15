## 请求IP地址

请求：

|CMD     |
|:-------|
|2 bytes |

* CMD：命令。1

应答：

|CMD     |ERRNO   |IPV4    |IPV6     |
|:-------|:-------|:-------|:--------|
|2 bytes |2 bytes |4 bytes |16 bytes |

* CMD：命令。1
* ERRNO：出错码。0 无，1 拒绝访问，11 重试。
* IPV4：IPv4地址。全0为无效值。
* IPV6：IPv6地址。全0为无效值。