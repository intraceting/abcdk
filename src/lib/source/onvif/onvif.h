/*
 * onvif.h
 *
 *  Created on: 2023年09月27日
 *      
 */
#ifndef ONVIF_ONVIF_H
#define ONVIF_ONVIF_H

#ifdef HAVE_ABCDK
#include "abcdk.h"
#endif //HAVE_ABCDK

#include "util/general.h"

#include "soapH.h"

__BEGIN_DECLS

/*定义常用结构体别名。*/
typedef struct soap onvif_t;
typedef struct SOAP_ENV__Header onvif_env_header_t;

typedef struct wsdd__ScopesType onvif_wsdd_scopestype_t;
typedef struct wsdd__ProbeType onvif_wsdd_probetype_t;
typedef struct __wsdd__ProbeMatches onvif_wsdd_probematches_t; 
typedef struct wsdd__ProbeMatchType onvif_wsdd_probematchtype_t;

typedef struct _tds__GetDeviceInformation onvif_tds_getdeviceinfo_req_t;
typedef struct _tds__GetDeviceInformationResponse onvif_tds_getdeviceinfo_rsp_t;
typedef struct _tds__GetCapabilities onvif_tds_getcapabilities_req_t;
typedef struct _tds__GetCapabilitiesResponse onvif_tds_getcapabilities_rsp_t;
typedef struct _tds__GetServices onvif_tds_getservices_req_t;
typedef struct _tds__GetServicesResponse onvif_tds_getservices_rsp_t;

typedef struct _trt__GetProfiles onvif_trt_getprofiles_req_t;
typedef struct _trt__GetProfilesResponse onvif_trt_getprofiles_rsq_t;
typedef struct _trt__GetStreamUri onvif_trt_getstreamuri_req_t;
typedef struct _trt__GetStreamUriResponse onvif_trt_getstreamuri_rsq_t;

typedef struct _trt2__GetProfiles onvif_trt2_getprofiles_req_t;
typedef struct _trt2__GetProfilesResponse onvif_trt2_getprofiles_rsq_t;
typedef struct _trt2__GetStreamUri onvif_trt2_getstreamuri_req_t;
typedef struct _trt2__GetStreamUriResponse onvif_trt2_getstreamuri_rsq_t;


void onvif_errmsg2log(struct soap *ctx,int level);

void *onvif_calloc(struct soap *ctx, size_t size);
void *onvif_memdup(struct soap *ctx,const void *data, size_t size);

void onvif_destroy(onvif_t **ctx);
onvif_t *onvif_create(int connect_timeout);

void onvif_set_env_header(onvif_t *ctx,const char *wsa_to,const char *wsa_action,const char *wsa_mid);

int onvif_send_probe(onvif_t *ctx, const char *endpoint, const char *type, const char *scope);
int onvif_recv_probe_matches(onvif_t *ctx, onvif_wsdd_probematches_t *matches);

int onvif_setauthinfo(onvif_t *ctx, const char *username, const char *password);

int onvif_getdeviceinfo(onvif_t *ctx,const char *xaddr, onvif_tds_getdeviceinfo_rsp_t *rsp);
int onvif_getcapabilitie(onvif_t *ctx,const char *xaddr, onvif_tds_getcapabilities_rsp_t *rsp);
int onvif_getservice(onvif_t *ctx,const char *xaddr, onvif_tds_getservices_rsp_t *rsp);

int onvif_getprofile(onvif_t *ctx,const char *xaddr, onvif_trt_getprofiles_rsq_t *rsp);
int onvif_getstreamuri(onvif_t *ctx,const char *xaddr, const char *token, onvif_trt_getstreamuri_rsq_t *rsp);

int onvif_getprofile_v2(onvif_t *ctx,const char *xaddr, onvif_trt2_getprofiles_rsq_t *rsp);
int onvif_getstreamuri_v2(onvif_t *ctx,const char *xaddr, const char *protocol, const char *token, onvif_trt2_getstreamuri_rsq_t *rsp);

__END_DECLS

#endif //ONVIF_ONVIF_H
