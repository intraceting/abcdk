#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##

#
SHELL_PATH=$(cd `dirname $0`; pwd)
SHELL_NAME=$(basename $0)

#
exitIFerror()
#errno
#errstr
#exitcode
{
    if [ $# -ne 3 ];then
    {
        echo "需要三个参数，分别是：errno，errstr，exitcode。"
        exit 1
    }
    fi 
    
    if [ $1 -ne 0 ];then
    {
        echo $2
        exit $3
    }
    fi
}

#检查opensll是否已经安装。
which openssl >> /dev/null 2>&1
exitIFerror $? "openssl未安装。" 1

#检查opensll-perl是否已经安装。
which c_rehash >> /dev/null 2>&1
exitIFerror $? "openssl-perl未安装。" 1

#检查MYCAPATH是否存在。
if [ "${MYCAPATH}" == "" ];then
    MYCAPATH="$HOME/mycapath/"
fi

#创建需要的路径。
mkdir -p ${MYCAPATH}

#
PAWD="No"
DAYS=365
CRL_DAYS=30
CRL_DP="https://localhost/${NAME}/crl.pem"
OR_NAME="MYCERT"
OU_NAME=""
DNS_NAMES="localhost,localhost4,localhost6"
IP_ADDRS="127.0.0.1,::1"
SY_NAME=""
SF_HOME="$HOME/mycerts/myRootCA/"
LF_HOME="$HOME/mycerts/myLeafCert/"
CMD=7

#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]

    -h, --help 
     打印帮助信息。

    -p, --pawd < yes | no >
     创建私钥密码。默认：${PAWD}

    -d, --days < NUMBER >
     有效期限(天)。默认：${DAYS}
     
    --crl-days < NUMBER >
     吊销列表有效期限(天)。默认：${CRL_DAYS}
     
    --crl-dp < URI >
     吊销列表发布地址，签发子证书时有效。默认：${CRL_DP}

    -O, --or-name < STRING >
     组织(政府、企业)名称。默认：${OR_NAME}

    -U, --ou-name < STRING >
     常用(主机、域)名称。默认：${OU_NAME}

    -D, --dns-names < STRING[,STRING,...] >
     DNS(主机、域)名称列表，以英文“,”为分界符。默认：${DNS_NAMES}
     
    -I, --ip-addrs < STRING[,STRING,...] >
     IP地址列表，以英文“,”为分界符。默认：${IP_ADDRS}

    -L --leaf-home < PATH >
     叶证书路径。默认：${LF_HOME}

    -H, --home < PATH >
     证书路径。默认：${SF_HOME}

    -c, --cmd < NUMBER >
     操作码。默认：${CMD}

     1：创建CA根证书。
     2：签发子证书。
     3：吊销子证书。
     4：更新吊销列表。
     5：打印吊销列表。
     6：打印子证书表。
     7：打印证书信息。
     8：签发子证书(CA)。
EOF
}

SHORTOPTS="h,p:,d:,O:,U:,D:,I:,L:,H:,c:"
LONGOPTS="help,pawd:,days:,crl-days:,crl-dp:,or-name:,ou-name:,dns-names:,ip-addrs:,leaf-home:,home:,cmd:"
ARGS=$(getopt --options $SHORTOPTS  --longoptions $LONGOPTS -- "$@")
if [ $? != 0 ];then
{
    PrintUsage
    exit 22
}
fi

eval set -- "${ARGS}"
while true;
do
    case ${1} in
    -h|--help)
        PrintUsage
        exit 22
    ;;
    -p|--pawd)
        PAWD=${2}
        shift 2
    ;;
    -d|--days)
        DAYS=${2}
        shift 2
    ;;
    --crl-days)
        CRL_DAYS=${2}
        shift 2
    ;;
    --crl-dp)
        CRL_DP="${2}"
        shift 2
    ;;
    -O|--or-name)
        OR_NAME="${2}"
        shift 2
    ;;
    -U|--ou-name)
        OU_NAME="${2}"
        shift 2
    ;;
    -D|--dns-names)
        DNS_NAMES="${2}"
        shift 2
    ;;
    -I|--ip-addrs)
        IP_ADDRS="${2}"
        shift 2
    ;;
    -L|--leaf-home)
        LF_HOME=$(realpath "${2}")
        shift 2
    ;;
    -H|--home)
        SF_HOME=$(realpath "${2}")
        shift 2
    ;;
    -c|--cmd)
        CMD=${2}
        shift 2
    ;;
    --)
        shift 1
        break;
    ;;
    *)
        PrintUsage
        exit 22
    ;;
    esac
done


#配置文件路径和文件名。
CNF_NAME="openssl.conf"
CNF_FILE="conf.d/${CNF_NAME}"
#私钥文件路径和文件名。
KEY_NAME="key.pem"
KEY_FILE="${KEY_NAME}"
#请求文件路径和文件名。
CSR_NAME="csr.pem"
CSR_FILE="conf.d/cache/${CSR_NAME}"
#证书文件路径和文件名。
CRT_NAME="crt.pem"
CRT_FILE="${CRT_NAME}"
#吊销列表文件路径和文件名。
CRL_NAME="crl.pem"
CRL_FILE="${CRL_NAME}"
#数据库文件路径和文件名。
DB_NAME="db.txt"
DB_FILE="conf.d/cache/${DB_NAME}"
#序列号文件路径和文件名。
SER_NUM_NAME="serial.num"
SER_NUM_FILE="conf.d/cache/${SER_NUM_NAME}"
#吊销序列号文件路径和文件名。
CRLSER_NUM_NAME="crlserial.num"
CRLSER_NUM_FILE="conf.d/cache/${CRLSER_NUM_NAME}"
#随机数文件路径和文件名。
RAND_NUM_NAME="rand.num"
RAND_NUM_FILE="conf.d/cache/${RAND_NUM_NAME}"
#子证书备份路径。
SON_CERT_BACK_PATH="conf.d/certs.backup"

#
make_base_conf()
{

#创建工作路径。
mkdir -p ${WK_PATH}
if [ ! -d ${WK_PATH} ];then
    exitIFerror 1 "'${WK_PATH}' must be an existing directory." 22
fi

#在工作路径中创建需要的子路径。
mkdir -p ${WK_PATH}/conf.d/cache
exitIFerror $? "'${WK_PATH}' must be an existing directory." 22


#在工作路径中创建需要的子路径。
mkdir -p ${WK_PATH}/${SON_CERT_BACK_PATH}
exitIFerror $? "'${WK_PATH}/${SON_CERT_BACK_PATH}' must be an existing directory." 22

#
if [ ! -f ${WK_PATH}/${DB_FILE} ];then
	touch ${WK_PATH}/${DB_FILE}
    exitIFerror $? "创建${WK_PATH}/${DB_FILE}失败。" 1
fi
	
if [ ! -f ${WK_PATH}/${SER_NUM_FILE} ];then
	openssl rand -out ${WK_PATH}/${SER_NUM_FILE} -hex 16
    exitIFerror $? "创建${WK_PATH}/${SER_NUM_FILE}失败。" 1
fi
	
if [ ! -f ${WK_PATH}/${CRLSER_NUM_FILE} ];then
	openssl rand -out ${WK_PATH}/${CRLSER_NUM_FILE} -hex 16
    exitIFerror $? "创建${WK_PATH}/${CRLSER_NUM_FILE}失败。" 1
fi

if [ ! -f ${WK_PATH}/${RAND_NUM_FILE} ];then
	openssl rand -out ${WK_PATH}/${RAND_NUM_FILE} -hex 16 
    exitIFerror $? "创建${WK_PATH}/${RAND_NUM_FILE}失败。" 1
fi

# 检查私钥文件。
if [ ! -f ${WK_PATH}/${KEY_FILE} ];then
{
	echo "创建私钥文件......"
    if [ "${PAWD}" == "yes" ];then
	    openssl genrsa -des3 -out ${WK_PATH}/${KEY_FILE} -f4 4096 >> /dev/null 2>&1
    else 
        openssl genrsa -out ${WK_PATH}/${KEY_FILE} -f4  4096 >> /dev/null 2>&1
    fi
    exitIFerror $? "创建${WK_PATH}/${KEY_FILE}失败。" 1
    echo "创建${WK_PATH}/${KEY_FILE}完成。"
}
fi

#创建配置文件。
cat > ${WK_PATH}/${CNF_FILE} << EOF
# OpenSSL CA configuration file

# 签发下级证书时使用。
[ ca ]
default_ca = CA_default

# 签发下级证书时使用。
[ CA_default ]
dir = .
new_certs_dir = \$dir/${SON_CERT_BACK_PATH}
database = \$dir/${DB_FILE}
serial = \$dir/${SER_NUM_FILE}
crlnumber = \$dir/${CRLSER_NUM_FILE}
RANDFILE = \$dir/${RAND_NUM_FILE}
certificate = \$dir/${CRT_FILE}
private_key = \$dir/${KEY_FILE}
#
default_md = sha512
copy_extensions = copy
unique_subject = no
policy = signing_policy

# 签发策略，签发下级证书时使用。
[ signing_policy ]
countryName = optional
stateOrProvinceName = optional
localityName = optional
organizationName = optional
commonName = optional
emailAddress = optional

# 注：自签名证书使用此段段落。
[ v3_ca_self ]
# 基本约束。
basicConstraints = critical,CA:true
# CA证书用途。
keyUsage = critical,keyCertSign,cRLSign
# 扩展密钥用途(CA默认不启用)。
#extendedKeyUsage = critical,serverAuth,clientAuth

# 签发下级证书时，对下级CA证书约束。
[ v3_ca_sub ]
# 基本约束。
basicConstraints = critical,CA:true
# CA证书用途。
keyUsage = critical,keyCertSign,cRLSign
# 扩展密钥用途(CA默认不启用)。
#extendedKeyUsage = critical,serverAuth,clientAuth
# 吊销分发（自签名证书不能启用）。
crlDistributionPoints=URI:${CRL_DP}

# 签发下级证书时，对下级非CA证书约束。
[ v3_ca_sub_not ]
# 基本约束。
basicConstraints = critical,CA:false
# 非CA证书用途。
keyUsage = critical,digitalSignature,keyEncipherment
# 扩展密钥用途。
extendedKeyUsage = critical,serverAuth,clientAuth
# 吊销分发（自签名证书不能启用）。
crlDistributionPoints=URI:${CRL_DP}

# 向CA机构申请证书时使用。
# 注：自签名证书也使用此段段落。
[ req ]
prompt = no
default_md = sha512
distinguished_name = req_distinguished_name
req_extensions = req_extensions

# 向CA机构申请证书时使用。
# 注：自签名证书也使用此段段落。
[ req_distinguished_name ]
organizationName = ${OR_NAME}
commonName = ${OU_NAME}

# 向CA机构申请证书时使用。
# 注：自签名证书也使用此段段落。
[ req_extensions ]
# 主题替代名称(域名)。
subjectAltName = critical,@alt_names

EOF
exitIFerror $? "创建${WK_PATH}/${CNF_FILE}失败。" 1
}


#
append_alt_names_conf()
{
OLDIFS="$IFS"
IFS=","
DNS_ARR=(${DNS_NAMES})
IP_ARR=(${IP_ADDRS})
IFS="$OLDIFS"

cat >> ${WK_PATH}/${CNF_FILE} << EOF
# 向CA机构申请证书时使用。
# 注：自签名证书也使用此段段落。
[ alt_names ]
EOF

#遍历写入DNS列表
NUM=0
for VAR in ${DNS_ARR[@]}
do
NUM=`expr $NUM + 1`;
echo "DNS.$NUM = $VAR" >> ${WK_PATH}/${CNF_FILE}
done

#遍历写入IP列表
NUM=0
for VAR in ${IP_ARR[@]}
do
NUM=`expr $NUM + 1`;
echo "IP.$NUM = $VAR" >> ${WK_PATH}/${CNF_FILE}
done

}

if [ ${CMD} -eq 1 ];then
{
    #检查证书是否已经创建。
    if [ -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "证书已经创建。" 1
    fi

    #
    if [ "${OU_NAME}" == "" ];then
        exitIFerror 1 "必须指定常用(主机、域)名称。" 1
    fi

    #生成配置文件。
    WK_PATH=${SF_HOME}
    make_base_conf
    append_alt_names_conf

    #保存当前目录，并设置工作目录为当前目录。
    CURDIR=$(pwd)
    cd ${SF_HOME}

    echo "创建自签名CA证书......"
    SN=$(openssl rand -hex 16)
    openssl req -new -x509 -config ${SF_HOME}/${CNF_FILE}  -extensions v3_ca_self -set_serial "0x${SN}" -key ${SF_HOME}/${KEY_FILE} -out ${SF_HOME}/${CRT_FILE} -days ${DAYS} -batch  >> /dev/null 2>&1
    exitIFerror $? "创建${SF_HOME}/${CRT_FILE}失败。" 1
    echo "创建${SF_HOME}/${CRT_FILE}完成。"

    #恢复当前目录。
    cd ${CURDIR}

    #截取目录做为代号。
    SYMBOL_NAME=$(basename ${SF_HOME})

    #复制证书到MYCAPATH目录。
    cp -f ${SF_HOME}/${CRT_FILE} ${MYCAPATH}/${SYMBOL_NAME}.crt
    #更新MYCAPATH目录。
    c_rehash  ${MYCAPATH}
}
elif [ ${CMD} -eq 2 ] || [ ${CMD} -eq 8 ] ;then
{
    #检查CA证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书不存在或未创建。" 1
    fi

    #检查证书是否已经创建。
    if [ -f ${LF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${LF_HOME}叶证书已经创建。" 1
    fi
    
    #
    if [ "${OU_NAME}" == "" ];then
        exitIFerror 1 "必须指定常用(主机、域)名称。" 1
    fi

    #生成配置文件。
    WK_PATH=${LF_HOME}
    make_base_conf
    append_alt_names_conf

    #保存当前目录，并设置叶证书工作目录。
    CURDIR=$(pwd)
    cd ${LF_HOME}

    echo "创建证书请求文件......"
	openssl req -new -config ${LF_HOME}/${CNF_FILE} -key ${LF_HOME}/${KEY_FILE} -out ${LF_HOME}/${CSR_FILE} -batch  >> /dev/null 2>&1
    exitIFerror $? "创建${LF_HOME}/${CSR_FILE}失败。" 1
    echo "创建${LF_HOME}/${CSR_FILE}完成。"

    #设置CA工作目录。
    cd ${SF_HOME}

    #生产新的随机序列号。
    openssl rand -hex -out ${SF_HOME}/${SER_NUM_FILE}  16

    echo "签发子证书......"
    echo "根据提示输入父级CA密码(如果存在的话)。"
    #如果配置文件中未配父级证书和私钥，则需要用参数指定。如下：
    #-keyfile ${SF_HOME}/${KEY_NAME} -cert ${SF_HOME}/${CRT_NAME}
    if [ ${CMD} -eq 8 ];then
        openssl ca -config ${SF_HOME}/${CNF_FILE} -extensions v3_ca_sub -in ${LF_HOME}/${CSR_FILE} -out ${LF_HOME}/${CRT_FILE}  -days ${DAYS}  -batch >> /dev/null 2>&1
    else
        openssl ca -config ${SF_HOME}/${CNF_FILE} -extensions v3_ca_sub_not -in ${LF_HOME}/${CSR_FILE} -out ${LF_HOME}/${CRT_FILE}  -days ${DAYS}  -batch >> /dev/null 2>&1
    fi 
	exitIFerror $? "签发${LF_HOME}/${CRT_FILE}失败。" 1
    echo "签发${LF_HOME}/${CRT_FILE}完成。"

    #恢复当前目录。
    cd ${CURDIR}
    
    #复制子证书(CA)到MYCAPATH目录。
    if [ ${CMD} -eq 8 ];then
    {
        #截取目录做为代号。
        SYMBOL_NAME=$(basename ${LF_HOME})

        #复制证书到MYCAPATH目录。
        cp -f ${LF_HOME}/${CRT_FILE} ${MYCAPATH}/${SYMBOL_NAME}.crt
        #更新MYCAPATH目录。
        c_rehash  ${MYCAPATH}
    }
    fi
}
elif [ ${CMD} -eq 3 ];then
{
    #检查CA证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书不存在或未创建。" 1
    fi

    #检查证书是否已经创建。
    if [ ! -f ${LF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${LF_HOME}叶证书不存在或未创建。" 1
    fi

    #获取证书序列号。
    SERIAL=$(openssl x509 -in ${LF_HOME}/${CRT_FILE} -serial -noout 2>> /dev/null |cut -d = -f 2)

    #检查证书是否属于当前CA。
    if [ ! -f ${SF_HOME}/${SON_CERT_BACK_PATH}/${SERIAL}.pem ];then
        exitIFerror 1 "证书${LF_HOME}/${CRT_FILE}不属于当前CA。" 1
    fi

    #保存当前目录，并设置CA工作目录为当前目录。
    CURDIR=$(pwd)
    cd ${SF_HOME}

    echo "吊销证书......"
    openssl ca -config ${SF_HOME}/${CNF_FILE} -revoke ${LF_HOME}/${CRT_FILE} -batch >> /dev/null 2>&1
    exitIFerror $? "吊销证书${LF_HOME}/${CRT_FILE}失败或已经吊销。" 1
    echo "吊销证书${LF_HOME}/${CRL_FILE}完成。"
    
    #恢复当前目录。
    cd ${CURDIR}
}
elif [ ${CMD} -eq 4 ];then
{
    #检查证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书不存在或未创建。" 1
    fi

    #保存当前目录，并设置工作目录为当前目录。
    CURDIR=$(pwd)
    cd ${SF_HOME}

    #生产新的随机序列号。
    openssl rand -hex -out ${SF_HOME}/${CRLSER_NUM_FILE} 16 

    echo "更新吊销列表......"
    openssl ca -gencrl -config ${SF_HOME}/${CNF_FILE} -out ${SF_HOME}/${CRL_FILE} -crldays ${CRL_DAYS} -batch >> /dev/null 2>&1
    exitIFerror $? "更新吊销列表${SF_HOME}/${CRL_FILE}失败。" 1
    echo "更新吊销列表${SF_HOME}/${CRL_FILE}完成。"
        
    #恢复当前目录。
    cd ${CURDIR}

    #截取目录做为代号。
    SYMBOL_NAME=$(basename ${SF_HOME})

    #复制证书到MYCAPATH目录。
    cp -f ${SF_HOME}/${CRL_FILE} ${MYCAPATH}/${SYMBOL_NAME}.crl
    #更新MYCAPATH目录。
    c_rehash ${MYCAPATH}

}
elif [ ${CMD} -eq 5 ];then
{
    #检查证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书未创建或工作路径错误。" 1
    fi

    #检查吊销列表是否已经创建。
    if [ ! -f ${SF_HOME}/${CRL_FILE} ];then
        exitIFerror 1 "${SF_HOME}吊销列表未创建或工作路径错误。" 1
    fi

    echo "打印吊销列表......"
    openssl crl -in ${SF_HOME}/${CRL_FILE} -text -noout
    exitIFerror $? "打印吊销列表${SF_HOME}/${CRL_FILE}失败。" 1
    echo "打印吊销列表${SF_HOME}/${CRL_FILE}完成。"

}
elif [ ${CMD} -eq 6 ];then
{
    #检查证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书未创建或工作路径错误。" 1
    fi

    echo "打印子证书列表......"
    find ${SF_HOME}/${SON_CERT_BACK_PATH}/ -type f -name *.pem -exec openssl x509 -serial -subject -noout -in {} \;
    echo "打印子证书列表${SF_HOME}/${SON_CERT_BACK_PATH}完成。"
}
elif [ ${CMD} -eq 7 ];then
{
    #检查证书是否已经创建。
    if [ ! -f ${SF_HOME}/${CRT_FILE} ];then
        exitIFerror 1 "${SF_HOME}证书未创建或工作路径错误。" 1
    fi

    echo "打印证书信息......"
    openssl x509 -in ${SF_HOME}/${CRT_FILE} -text -noout
    exitIFerror $? "打印证书信息${SF_HOME}/${CRT_FILE}失败。" 1
    echo "打印证书信息${SF_HOME}/${CRT_FILE}完成。"
}
fi