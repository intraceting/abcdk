/*
 * onvif.c
 *
 *  Created on: 2023年09月27日
 * 
 * OVIVF环境创建规则：
 * 
 * 1：创建工作目录。
 * 2：复制$PREFIX/gsoap/import目录到工作目录。
 * 3：复制$PREFIX/gsoap/custom目录到工作目录。
 * 4：复制$PREFIX/gsoap/WS/typemap.dat目录到工作目录。
 * 
 * 1：在typemap.dat文件内，添加“trt2 = "http://www.onvif.org/ver20/media/wsdl"”，用于支持第二版。
 * 2：在typemap.dat文件内，启用支持“duration.h”的代码，具体方法参考文件内头部的注释。
 * 3：在typemap.dat文件内，启用支持“struct_timeval.h”的代码，具体方法参考文件内头部的注释。
 * 
 * $ wsdl2h -P -x -t ./typemap.dat  -o onvif.env.h -c -s  \
 *      https://www.onvif.org/ver10/network/wsdl/remotediscovery.wsdl \ 
 *      https://www.onvif.org/ver10/device/wsdl/devicemgmt.wsdl \
 *      https://www.onvif.org/ver10/media/wsdl/media.wsdl \
 *      https://www.onvif.org/ver20/media/wsdl/media.wsdl  \
 *      https://www.onvif.org/ver20/ptz/wsdl/ptz.wsdl
 * 
 * 3：在onvif.env.h文件内，添加“#import "wsse.h"”，用于支持SSL。
 * 4：在import/was5.h文件内，修改SOAP_ENV__Fault的名字，用于解决多次重定义的问题。
 * 
 * $ soapcpp2 -2 -C -L -c -x -I ./import/ -I ./custom/ onvif.env.h
 * 
 * 5：在soapH.h文件内，soap_getelement、soap_putelement等几个函数启用C接口风格，解决dom.c文件内部分功能无法使用的问题。
 * 6：编译选项增加-DWITH_OPENSSL -DWITH_DOM，用于支持SSL。
 * 7：全局变量namespaces复制到适当位置。
 * 8：依赖的源码文件见下面列表。
 *      custom/duration.c|h
 *      custom/struct_timeval.c|h
 *      plugin/mecevp.c|h
 *      plugin/smdevp.c|h
 *      plugin/smdevp.c|h
 *      plugin/threads.c|h
 *      plugin/wsaapi.c|h
 *      plugin/wsseapi.c|h
 *      dom.c 
 *      stdsoap2.c|h
 * 
 */
#include "onvif.h"

#include "gsoap/stdsoap2.h"
#include "gsoap/plugin/wsaapi.h"
#include "gsoap/plugin/wsseapi.h"

/**
 * 从nsmap复制而来。
 *
 * 原始文件由soapcpp2生成，但被定义在头文件中，多次包含会发生重复定义错误。
 */

/* This defines the global XML namespaces[] table to #include and compile
   The first four entries are mandatory and should not be removed */
SOAP_NMAC struct Namespace namespaces[] = {
        { "SOAP-ENV", "http://www.w3.org/2003/05/soap-envelope", "http://schemas.xmlsoap.org/soap/envelope/", NULL },
        { "SOAP-ENC", "http://www.w3.org/2003/05/soap-encoding", "http://schemas.xmlsoap.org/soap/encoding/", NULL },
        { "xsi", "http://www.w3.org/2001/XMLSchema-instance", "http://www.w3.org/*/XMLSchema-instance", NULL },
        { "xsd", "http://www.w3.org/2001/XMLSchema", "http://www.w3.org/*/XMLSchema", NULL },
        { "wsa", "http://schemas.xmlsoap.org/ws/2004/08/addressing", "http://www.w3.org/2005/08/addressing", NULL },
        { "wsdd", "http://schemas.xmlsoap.org/ws/2005/04/discovery", NULL, NULL },
        { "chan", "http://schemas.microsoft.com/ws/2005/02/duplex", NULL, NULL },
        { "wsa5", "http://www.w3.org/2005/08/addressing", "http://schemas.xmlsoap.org/ws/2004/08/addressing", NULL },
        { "c14n", "http://www.w3.org/2001/10/xml-exc-c14n#", NULL, NULL },
        { "ds", "http://www.w3.org/2000/09/xmldsig#", NULL, NULL },
        { "saml1", "urn:oasis:names:tc:SAML:1.0:assertion", NULL, NULL },
        { "saml2", "urn:oasis:names:tc:SAML:2.0:assertion", NULL, NULL },
        { "wsu", "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd", NULL, NULL },
        { "xenc", "http://www.w3.org/2001/04/xmlenc#", NULL, NULL },
        { "wsc", "http://docs.oasis-open.org/ws-sx/ws-secureconversation/200512", "http://schemas.xmlsoap.org/ws/2005/02/sc", NULL },
        { "wsse", "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd", "http://docs.oasis-open.org/wss/oasis-wss-wssecurity-secext-1.1.xsd", NULL },
        { "xmime", "http://tempuri.org/xmime.xsd", NULL, NULL },
        { "xop", "http://www.w3.org/2004/08/xop/include", NULL, NULL },
        { "wsrfbf", "http://docs.oasis-open.org/wsrf/bf-2", NULL, NULL },
        { "wsnt", "http://docs.oasis-open.org/wsn/b-2", NULL, NULL },
        { "wstop", "http://docs.oasis-open.org/wsn/t-1", NULL, NULL },
        { "tt", "http://www.onvif.org/ver10/schema", NULL, NULL },
        { "tdn", "http://www.onvif.org/ver10/network/wsdl", NULL, NULL },
        { "tds", "http://www.onvif.org/ver10/device/wsdl", NULL, NULL },
        { "tptz", "http://www.onvif.org/ver20/ptz/wsdl", NULL, NULL },
        { "trt", "http://www.onvif.org/ver10/media/wsdl", NULL, NULL },
        { "trt2", "http://www.onvif.org/ver20/media/wsdl", NULL, NULL },
        { NULL, NULL, NULL, NULL} /* end of namespaces[] */
    };

void onvif_errmsg2log(onvif_t *ctx,int level)
{
    assert(ctx != NULL);

    aicontrib_log_printf(level, "[soap] error: %d, %s, %s\n", ctx->error, *soap_faultcode(ctx), *soap_faultstring(ctx));
}

void *onvif_calloc(onvif_t *ctx, size_t size)
{
    void *p = NULL;

    assert(ctx != NULL && size > 0);

    p = soap_malloc(ctx, size);
    if (!p)
        return NULL;

    memset(p, 0, size);

    return p;
}

void *onvif_memdup(onvif_t *ctx, const void *data, size_t size)
{
    void *p = NULL;

    assert(ctx != NULL && data != NULL && size > 0);

    p = onvif_calloc(ctx,size+1);
    if(!p)
        return NULL;

    memcpy(p,data,size);

    return p;
}

void onvif_destroy(onvif_t **ctx)
{
    onvif_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p=*ctx;
    *ctx = NULL;

    soap_destroy(ctx_p); 
    soap_end(ctx_p);
    soap_done(ctx_p); 
    soap_free(ctx_p);
}

onvif_t * onvif_create(int connect_timeout)
{
    onvif_t *ctx = NULL;

    ctx = soap_new();
    if(!ctx)
        return NULL;

    /*设置命名空间。*/
    soap_set_namespaces(ctx, namespaces); 

    /*设置为UTF-8编码，否则叠加中文OSD会乱码。*/
    soap_set_mode(ctx, SOAP_C_UTFSTRING); 
    
    ctx->recv_timeout = 15;          
    ctx->send_timeout = 15;
    ctx->connect_timeout = connect_timeout;
    ctx->socket_flags = MSG_NOSIGNAL;     

    return ctx;
}

void onvif_set_env_header(onvif_t *ctx,const char *wsa_to,const char *wsa_action,const char *wsa_mid)
{
    assert(NULL != ctx && wsa_to != NULL && wsa_action != NULL);
    assert( *wsa_to != '\0' && *wsa_action != '\0');

    if(!ctx->header)
    {
        /*创建新的。*/
        ctx->header = (onvif_env_header_t *)onvif_calloc(ctx, sizeof(onvif_env_header_t));
        /*默认配置。*/
        soap_default_SOAP_ENV__Header(ctx, ctx->header);
    }

    ctx->header->wsa__MessageID = (char*)(wsa_mid?wsa_mid:soap_wsa_rand_uuid(ctx));
    ctx->header->wsa__To = (char *)onvif_memdup(ctx,wsa_to, strlen(wsa_to));
    ctx->header->wsa__Action = (char *)onvif_memdup(ctx, wsa_action, strlen(wsa_action));

    return;
}

int onvif_send_probe(onvif_t *ctx, const char *endpoint, const char *type, const char *scope)
{
    onvif_wsdd_probetype_t req;
    int chk;

    assert(NULL != ctx && NULL != endpoint && NULL != type);

    memset(&req, 0, sizeof(onvif_wsdd_probetype_t));
    soap_default_wsdd__ProbeType(ctx, &req);

    req.Types = (char *)onvif_memdup(ctx, type, strlen(type));
    req.Scopes = (onvif_wsdd_scopestype_t *)onvif_calloc(ctx, sizeof(onvif_wsdd_scopestype_t));
    req.Scopes->__item = (char *)(scope ? onvif_memdup(ctx, scope, strlen(scope)) : onvif_calloc(ctx, 1));

    chk = soap_send___wsdd__Probe(ctx, endpoint, NULL, &req);
    if(chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx,LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_recv_probe_matches(onvif_t *ctx, onvif_wsdd_probematches_t *matches)
{
    int chk = 0;

    assert(NULL != ctx && NULL != matches);

    memset(matches, 0, sizeof(onvif_wsdd_probematches_t));
    chk = soap_recv___wsdd__ProbeMatches(ctx, matches);
    if(chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx,LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_setauthinfo(struct soap *ctx, const char *username, const char *password)
{
    int chk = 0;

    assert(NULL != ctx && NULL != username && NULL != password);

    chk = soap_wsse_add_UsernameTokenDigest(ctx, NULL, username, password);
    if(chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx,LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getdeviceinfo(onvif_t *ctx, const char *xaddr, onvif_tds_getdeviceinfo_rsp_t *rsp)
{
    onvif_tds_getdeviceinfo_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && NULL != rsp);
    assert(*xaddr != '\0');

    memset(&req, 0, sizeof(onvif_tds_getdeviceinfo_req_t));
    memset(rsp, 0, sizeof(onvif_tds_getdeviceinfo_rsp_t));

    chk = soap_call___tds__GetDeviceInformation(ctx, xaddr, NULL, &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getcapabilitie(onvif_t *ctx,const char *xaddr, onvif_tds_getcapabilities_rsp_t *rsp)
{
    onvif_tds_getcapabilities_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && NULL != rsp);
    assert(*xaddr != '\0');

    memset(&req, 0, sizeof(onvif_tds_getcapabilities_req_t));
    memset(rsp, 0, sizeof(onvif_tds_getcapabilities_rsp_t));

    chk = soap_call___tds__GetCapabilities(ctx, xaddr, NULL, &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getservice(onvif_t *ctx,const char *xaddr, onvif_tds_getservices_rsp_t *rsp)
{
    onvif_tds_getservices_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && NULL != rsp);
    assert(*xaddr != '\0');

    memset(&req, 0, sizeof(onvif_tds_getservices_req_t));
    memset(rsp, 0, sizeof(onvif_tds_getservices_rsp_t));

    req.IncludeCapability = xsd__boolean__false_;

    chk = soap_call___tds__GetServices(ctx, xaddr, NULL, &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getprofile(onvif_t *ctx,const char *xaddr, onvif_trt_getprofiles_rsq_t *rsp)
{
    onvif_trt_getprofiles_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && NULL != rsp);
    assert(*xaddr != '\0');

    memset(&req, 0, sizeof(onvif_trt_getprofiles_req_t));
    memset(rsp, 0, sizeof(onvif_trt_getprofiles_rsq_t));

    chk = soap_call___trt__GetProfiles(ctx, xaddr, "", &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getstreamuri(onvif_t *ctx, const char *xaddr, const char *token, onvif_trt_getstreamuri_rsq_t *rsp)
{
    onvif_trt_getstreamuri_req_t req;
    struct tt__StreamSetup ss;
    struct tt__Transport ts;
    int chk;

    assert(NULL != ctx && NULL != xaddr && token != NULL && NULL != rsp);
    assert(*xaddr != '\0');

    ss.Stream = tt__StreamType__RTP_Unicast;
    ss.Transport = &ts;
    ss.Transport->Protocol = tt__TransportProtocol__RTSP;
    ss.Transport->Tunnel = NULL;
    req.StreamSetup = &ss;
    req.ProfileToken = (char *)onvif_memdup(ctx, token, strlen(token));

    chk = soap_call___trt__GetStreamUri(ctx, xaddr, NULL, &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getprofile_v2(onvif_t *ctx,const char *xaddr, onvif_trt2_getprofiles_rsq_t *rsp)
{
    onvif_trt2_getprofiles_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && NULL != rsp);
    assert(*xaddr != '\0');

    memset(&req, 0, sizeof(onvif_trt2_getprofiles_req_t));
    memset(rsp, 0, sizeof(onvif_trt2_getprofiles_rsq_t));

    req.__sizeType = 1;

    chk = soap_call___trt2__GetProfiles(ctx, xaddr, "", &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}

int onvif_getstreamuri_v2(onvif_t *ctx,const char *xaddr, const char *protocol,const char *token, onvif_trt2_getstreamuri_rsq_t *rsp)
{
    onvif_trt2_getstreamuri_req_t req;
    int chk;

    assert(NULL != ctx && NULL != xaddr && protocol != NULL && token != NULL && NULL != rsp);
    assert(*xaddr != '\0' && *protocol != '\0' && *token != '\0');

    memset(&req, 0, sizeof(onvif_trt2_getstreamuri_req_t));
    memset(rsp, 0, sizeof(onvif_trt2_getstreamuri_rsq_t));

    req.Protocol = (char*)onvif_memdup(ctx, protocol, strlen(protocol));
    req.ProfileToken = (char*)onvif_memdup(ctx, token, strlen(token));

    chk = soap_call___trt2__GetStreamUri(ctx, xaddr, "", &req, rsp);
    if (chk != SOAP_OK)
    {
        onvif_errmsg2log(ctx, LOG_ERR);
        return -1;
    }

    return 0;
}