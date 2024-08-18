## 投递消息

请求：

|CMD     |LENGTH |PAYLOAD |
|:-------|:------|:-------|
|2 bytes |3 type |N bytes |

* CMD：命令。2
* LENGTH：消息长度。长度为0时仅更新链路活动时间。
* PAYLOAD：有效载荷。

应答：

* 无