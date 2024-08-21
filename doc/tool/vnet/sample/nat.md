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

void setup_route_table() {
  run("sysctl -w net.ipv4.ip_forward=1");

#ifdef AS_CLIENT
  run("iptables -t nat -A POSTROUTING -o tun0 -j MASQUERADE");
  run("iptables -I FORWARD 1 -i tun0 -m state --state RELATED,ESTABLISHED -j ACCEPT");
  run("iptables -I FORWARD 1 -o tun0 -j ACCEPT");
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "ip route add %s via $(ip route show 0/0 | sed -e 's/.* via \([^ ]*\).*/\1/')", SERVER_HOST);
  run(cmd);
  run("ip route add 0/1 dev tun0");
  run("ip route add 128/1 dev tun0");
#else
  run("iptables -t nat -A POSTROUTING -s 10.8.0.0/16 ! -d 10.8.0.0/16 -m comment --comment 'vpndemo' -j MASQUERADE");
  run("iptables -A FORWARD -s 10.8.0.0/16 -m state --state RELATED,ESTABLISHED -j ACCEPT");
  run("iptables -A FORWARD -d 10.8.0.0/16 -j ACCEPT");
#endif
}

iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT


