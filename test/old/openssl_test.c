/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "abcdk/util/getargs.h"
#include "abcdk/util/openssl.h"
#include "abcdk/util/socket.h"


#ifdef HEADER_RSA_H

void test_rsa(abcdk_tree_t *opt)
{
    OpenSSL_add_all_algorithms();
    OpenSSL_add_all_ciphers();
    ERR_load_crypto_strings();
    ERR_load_RSA_strings();

    RSA *key = NULL;
    RSA *prikey = NULL;
    RSA *pubkey = NULL;
    int chk;
#if 1


        key = abcdk_openssl_rsa_create(2048,RSA_F4);

        chk = abcdk_openssl_rsa_to_file(
            abcdk_option_get(opt, "--rsa-key-prifile", 0, "/tmp/abc-pri.pem"),
            key,
            1,
            abcdk_option_get(opt, "--rsa-key-pwd", 0, "123456"));

        assert(chk > 0);

        chk = abcdk_openssl_rsa_to_file(
            abcdk_option_get(opt, "--rsa-key-pubfile", 0, "/tmp/abc-pub.pem"),
            key,
            0,
            abcdk_option_get(opt, "--rsa-key-pwd", 0, "123456"));

        assert(chk > 0);

#else

    prikey = abcdk_openssl_rsa_from_file(
        abcdk_option_get(opt, "--rsa-key-prifile", 0, ""),
        1,
        abcdk_option_get(opt, "--rsa-key-pwd", 0, ""));
    pubkey = abcdk_openssl_rsa_from_file(
        abcdk_option_get(opt, "--rsa-key-pubfile", 0, ""),
        0,
        //abcdk_option_get(opt, "--rsa-key-pwd", 0, ""));
        NULL);

    assert(prikey && pubkey);



    char *buf1 = (char*) abcdk_heap_alloc(1000);
    char *buf2 = (char*) abcdk_heap_alloc(2000);
    char *buf3 = (char*) abcdk_heap_alloc(2000);

    memset(buf1,'a',1000);

    ssize_t m = abcdk_openssl_rsa_ecb_encrypt(buf2,buf1,1000,pubkey,0,RSA_NO_PADDING);

    int e = ERR_get_error();
    char szErrMsg[1024] = {0};
    char *pTmp = NULL;
    pTmp = ERR_error_string(e,szErrMsg); 

    printf("%s\n",pTmp);

    assert(m>0);

    chk = abcdk_openssl_rsa_ecb_decrypt(buf3,buf2,m,prikey,1,RSA_NO_PADDING);

    assert(chk >0);



    assert(memcmp(buf1,buf3,1000)==0);


    abcdk_heap_free(buf1);
    abcdk_heap_free(buf2);
    abcdk_heap_free(buf3);

#endif 

    RSA_free(key);
    RSA_free(prikey);
    RSA_free(pubkey);
}

#endif //HEADER_RSA_H

#ifdef HEADER_AES_H

void test_aes(abcdk_tree_t *opt)
{
    AES_KEY ek,dk;
    AES_KEY ek2,dk2;
    uint8_t iv[AES_BLOCK_SIZE * 4] = {0};

    assert(abcdk_openssl_aes_set_key(&ek,"abcde",5,9,1)>0);
    assert(abcdk_openssl_aes_set_key(&dk,"abcde",5,9,0)>0);

    assert(abcdk_openssl_aes_set_key(&ek2,"qwert",5,9,1)>0);
    assert(abcdk_openssl_aes_set_key(&dk2,"qwert",5,9,0)>0);

    assert(abcdk_openssl_aes_set_iv(iv,"12345789344wewqerwreqwer",20,9)>0);

    char *buf1 = (char*) abcdk_heap_alloc(1000);
    char *buf2 = (char*) abcdk_heap_alloc(2000);
    char *buf3 = (char*) abcdk_heap_alloc(2000);

    memset(buf1,'a',1000);

    int len = AES_BLOCK_SIZE * 16;
    AES_bi_ige_encrypt(buf1,buf2,len,&ek,&ek2,iv,AES_ENCRYPT);

    AES_bi_ige_encrypt(buf2,buf3,len,&dk,&dk2,iv,AES_DECRYPT);


    assert(memcmp(buf1,buf3,len)==0);


   abcdk_heap_free(buf1);
    abcdk_heap_free(buf2);
    abcdk_heap_free(buf3);

}

#endif //HEADER_AES_H

#ifdef HEADER_SSL_H

void test_ssl(abcdk_tree_t *opt)
{
   SSL_library_init();
   OpenSSL_add_all_algorithms();
   SSL_load_error_strings();
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL  
    const SSL_METHOD *method = TLSv1_2_client_method();
#else
    const SSL_METHOD *method = TLS_client_method();
#endif 

    SSL_CTX * ctx = SSL_CTX_new(method);

    int chk = abcdk_openssl_ctx_load_cert(ctx, NULL,
                                     abcdk_option_get(opt, "--rsa-key-prifile", 0, "/tmp/abc-pri.pem"),
                                     abcdk_option_get(opt, "--rsa-key-pwd", 0, "123456"));

    assert(chk == 0);

    SSL* s = abcdk_openssl_ssl_alloc(ctx);


    // abcdk_sockaddr_t addr={0};
    // assert(abcdk_sockaddr_from_string(&addr,"www.taobao.com:443",1)==0);

    // int c = abcdk_socket(addr.family,0);
    
    // assert(abcdk_connect(c,&addr,10000)==0);

    // assert(abcdk_openssl_ssl_handshake(c,s,0,10000)==0);


    // abcdk_closep(&c);

    abcdk_openssl_ssl_free(&s);

    SSL_CTX_free(ctx);
}

#endif //HEADER_SSL_H

#ifdef HEADER_HMAC_H

void test_hmac(abcdk_tree_t *opt)
{
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX *hmac = (HMAC_CTX *)abcdk_heap_alloc(sizeof(HMAC_CTX));
#else
    HMAC_CTX *hmac = HMAC_CTX_new();
#endif
    /*123456*/
    assert(abcdk_openssl_hmac_init(hmac,"123456",6,ABCDK_OPENSSL_HMAC_SHA256)==0);

    HMAC_Update(hmac,"123456",6);

    uint8_t buf[100]={0};
    int len;

    HMAC_Final(hmac,buf,&len);

    for(int i = 0;i<len;i++)
        printf("%02x",buf[i]);
    printf("\n");

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_cleanup(hmac);
    abcdk_heap_free(hmac);
#else
    HMAC_CTX_free(hmac);
#endif 

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX *hmac2 = (HMAC_CTX *)abcdk_heap_alloc(sizeof(HMAC_CTX));
#else
    HMAC_CTX *hmac2 = HMAC_CTX_new();
#endif

    /*123457*/
    assert(abcdk_openssl_hmac_init(hmac2,"123456",6,ABCDK_OPENSSL_HMAC_SHA256)==0);

    HMAC_Update(hmac2,"123456",6);

    uint8_t buf2[100]={0};
    int len2;

    HMAC_Final(hmac2,buf2,&len2);

    for(int i = 0;i<len2;i++)
        printf("%02x",buf2[i]);
    printf("\n");

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_cleanup(hmac2);
    abcdk_heap_free(hmac2);
#else
    HMAC_CTX_free(hmac2);
#endif 

}

#endif //HEADER_HMAC_H

int main(int argc, char **argv)
{

    abcdk_tree_t *t = abcdk_tree_alloc(NULL);

    abcdk_getargs(t,argc,argv,"--");

#ifdef HEADER_RSA_H

    test_rsa(t);
    
#endif //HEADER_RSA_H

#ifdef HEADER_AES_H

    test_aes(t);

#endif //HEADER_AES_H

#ifdef HEADER_SSL_H

    test_ssl(t);

#endif //HEADER_SSL_H

#ifdef HEADER_HMAC_H

    test_hmac(t);

#endif //HEADER_HMAC_H

    abcdk_tree_free(&t);

    return 0;
}