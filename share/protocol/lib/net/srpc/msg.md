## 消息

请求：

|Length  |CMD     |MID      |NONCE    |Data     |
|:-------|:-------|:--------|:--------|:--------|
|4 bytes |1 byte  |8 bytes  |32 bytes |N bytes  |

* Length：长度。注：不包括自己。
* CMD：指令码。2 请求。
* MID：消息ID。
* NONCE：NONCE码。
* DATA：变长数据。


应答：

|Length  |CMD     |MID      |NONCE    |Data     |
|:-------|:-------|:--------|:--------|:--------|
|4 bytes |1 byte  |8 bytes  |32 bytes |N bytes  |

* Length：长度。注：不包括自己。
* CMD：指令码。1 应答。
* MID：消息ID。
* NONCE：NONCE码。
* DATA：变长数据。