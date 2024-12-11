## 消息

请求：

|Length  |CMD     |MID      |Data     |
|:-------|:-------|:--------|:--------|
|4 bytes |1 byte  |8 bytes  |N bytes  |

* Length：长度。注：不包括自己。
* CMD：指令。2 请求。
* MID：编号。
* DATA：数据。


应答：

|Length  |CMD     |MID      |Data     |
|:-------|:-------|:--------|:--------|
|4 bytes |1 byte  |8 types  |N bytes  |

* Length：长度。注：不包括自己。
* CMD：指令。1 应答。
* MID：编号。
* DATA：数据。