## 消息

请求：

|Length  |NONCE    |CMD     |MID      |Data     |
|:-------|:--------|:-------|:--------|:--------|
|4 bytes |32 bytes |1 byte  |8 bytes  |N bytes  |

* Length：长度。注：不包括自己。
* NONCE：随机码。
* CMD：指令。2 请求。
* MID：编号。
* DATA：数据。


应答：

|Length  |NONCE    |CMD     |MID      |Data     |
|:-------|:--------|:-------|:--------|:--------|
|4 bytes |32 bytes |1 byte  |8 bytes  |N bytes  |

* Length：长度。注：不包括自己。
* NONCE：随机码。
* CMD：指令。1 应答。
* MID：编号。
* DATA：数据。