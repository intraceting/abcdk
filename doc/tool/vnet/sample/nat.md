# TUN到ETH的nat配置方案。

## 利用防火墙NAT功能将tun0(网卡)的流量转发到网卡eth0(网卡)。

### 设置nat，添加一条POSTROUTING规则，将来自10.0.0.0/24网段的流量转发到eth0中。
```bash
sudo iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -o eth0 -j MASQUERADE
```

### 设置filter，添加两条FORWARD规则，允许从tun0到eth0的流量转发，并且允许eth0到tun0的流量转发。
```bash
sudo iptables -I FORWARD 1 -i tun0 -o eth0 -j ACCEPT
sudo iptables -I FORWARD 1 -i eth0 -o tun0 -j ACCEPT
```

### 查看nat组POSTROUTING规则。
```bash
sudo iptables -t nat -vnL POSTROUTING --line-numbers
```

### 查看filter组FORWARD规则。
```bash
sudo iptables -vL FORWARD --line-numbers
```