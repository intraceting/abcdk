## 投递消息

请求：

|SRPC    |CMD     |LENGTH |PAYLOAD |
|:-------|:-------|:------|:-------|
|N bytes |2 bytes |2 type |N bytes |

* SRPC-HDR：SRPC协议。
* CMD：命令。2
* LENGTH：消息长度。长度为0时仅更新链路活动时间。
* PAYLOAD：有效载荷。

应答：

* 无