/*
 * onvif.hpp
 *
 *  Created on: 2023年10月23日
 * 
*/
#ifndef ONVIF_ONVIF_HPP
#define ONVIF_ONVIF_HPP

#include <string>
#include <vector>
#include <map>

#include "onvif.h"

namespace aicontrib
{
    namespace onvif
    {
        template <typename T>
        inline T *Calloc(struct soap *ctx)
        {
            return (T *)onvif_calloc(ctx, sizeof(T));
        }

        inline int Probe(std::vector<std::string> &xaddrs,int timeout = 3,const char *endpoint = "soap.udp://239.255.255.250:3702")
        {
            onvif_t *ctx = NULL;
            onvif_wsdd_probematches_t matches;
            int chk;

            xaddrs.clear();
            
            ctx = onvif_create(1);
            if(!ctx)
                return -1;

            /*默认3秒: 3 seconds.*/
            ctx->recv_timeout = (timeout > 0 ? timeout : 3);

            onvif_set_env_header(ctx, "urn:schemas-xmlsoap-org:ws:2005:04:discovery", "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe", NULL);
            onvif_send_probe(ctx, endpoint, "tdn:NetworkVideoTransmitter", NULL);

            while (1)
            {
                chk = onvif_recv_probe_matches(ctx, &matches);

                if (chk != SOAP_OK && ctx->error)
                {
                    onvif_errmsg2log(ctx, LOG_ERR);
                    break;
                }

                if (chk != SOAP_OK)
                    continue;

                if (ctx->error)
                {
                    onvif_errmsg2log(ctx, LOG_ERR);
                    continue;
                }

                if (!matches.wsdd__ProbeMatches)
                    continue;

                for (int i = 0; i < matches.wsdd__ProbeMatches->__sizeProbeMatch; i++)
                {
                    struct wsdd__ProbeMatchType *p = &matches.wsdd__ProbeMatches->ProbeMatch[i];
                    xaddrs.push_back(p->XAddrs);
                }
            }

            onvif_destroy(&ctx);

            return xaddrs.size();
        }

        inline int Probe(std::vector<std::vector<std::string>> &xaddrs,int timeout = 3,const char *endpoint = "soap.udp://239.255.255.250:3702")
        {
#ifdef ABCDK_H
            std::vector<std::string> tmp;
            int chk;

            chk = Probe(tmp,timeout,endpoint);
            if(chk <= 0)
                return 0;

            xaddrs.resize(tmp.size());
            for (int i = 0; i < tmp.size(); i++)
            {
                const char *p = tmp[i].c_str();

                while (1)
                {
                    abcdk_object_t *p1 = abcdk_strtok3(&p, " ", 1);
                    if (!p1)
                        break;

                    xaddrs[i].push_back(p1->pstrs[0]);

                    abcdk_object_unref(&p1);
                }
            }

            return chk;
#else
            return -1;
#endif //ABCDK_H
        }

        inline int GetDeviceInfo(std::map<std::string, std::string> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_tds_getdeviceinfo_rsp_t rsp = {0};
            int chk;

            assert(xaddr != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getdeviceinfo(ctx, xaddr, &rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            info["Manufacturer"] = rsp.Manufacturer;
            info["Model"] = rsp.Model;
            info["FirmwareVersion"] = rsp.FirmwareVersion;
            info["SerialNumber"] = rsp.SerialNumber;
            info["HardwareId"] = rsp.HardwareId;

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetCapabilitie(std::map<std::string, std::string> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_tds_getcapabilities_rsp_t rsp = {0};
            int chk;

            assert(xaddr != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getcapabilitie(ctx, xaddr, &rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            if (rsp.Capabilities->Analytics)
                info["Analytics.XAddr"] = rsp.Capabilities->Analytics->XAddr;
            if (rsp.Capabilities->Device)
                info["Device.XAddr"] = rsp.Capabilities->Device->XAddr;
            if (rsp.Capabilities->Events)
                info["Events.XAddr"] = rsp.Capabilities->Events->XAddr;
            if (rsp.Capabilities->Imaging)
                info["Imaging.XAddr"] = rsp.Capabilities->Imaging->XAddr;
            if (rsp.Capabilities->Media)
                info["Media.XAddr"] = rsp.Capabilities->Media->XAddr;
            if (rsp.Capabilities->PTZ)
                info["PTZ.XAddr"] = rsp.Capabilities->PTZ->XAddr;

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetService(std::vector<std::string> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_tds_getservices_rsp_t rsp = {0};
            int chk;

            assert(xaddr != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getservice(ctx, xaddr, &rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            for(int i = 0;i<rsp.__sizeService;i++)
            {
                info.push_back(rsp.Service[i].XAddr);
            }

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetProfile(std::vector<std::map<std::string, std::string>> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_trt_getprofiles_rsq_t rsp = {0};
            int chk;

            assert(xaddr != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getprofile(ctx, xaddr,&rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            info.resize(rsp.__sizeProfiles);
            for (int i = 0; i < rsp.__sizeProfiles; i++)
            {
                 info[i]["Name"] = rsp.Profiles[i].Name;
                 info[i]["Token"] = rsp.Profiles[i].token;
            }

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetStreamUri(std::string &info, const char *xaddr,const char *token, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_trt_getstreamuri_rsq_t rsp = {0};
            int chk;

            assert(xaddr != NULL && token != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getstreamuri(ctx, xaddr, token,&rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            info = rsp.MediaUri->Uri;

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetStreamUri(std::vector<std::map<std::string, std::string>> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            std::vector<std::string> service_info;
            std::vector<std::map<std::string, std::string>> profile_info;
            int chk;

            info.clear();

            chk = GetService(service_info, xaddr, username, password);
            if (chk != 0)
                return -1;

            for (auto &t : service_info)
            {
                chk = GetProfile(profile_info, t.c_str(), username, password);
                if (chk != 0)
                    continue;

                for (auto &t2 : profile_info)
                {
                   std::map<std::string, std::string> tmp;

                    chk = GetStreamUri(tmp[t2["Name"]], t.c_str(), t2["Token"].c_str(), username, password);
                    if (chk != 0)
                        continue;

                    info.push_back(tmp);
                }
            }

            return info.size();
        }
        
        inline int GetProfile_V2(std::vector<std::map<std::string, std::string>> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_trt2_getprofiles_rsq_t rsp = {0};
            int chk;

            assert(xaddr != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getprofile_v2(ctx, xaddr,&rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            info.resize(rsp.__sizeProfiles);
            for (int i = 0; i < rsp.__sizeProfiles; i++)
            {
                info[i]["Name"] = rsp.Profiles[i].Name;
                info[i]["Token"] = rsp.Profiles[i].token;
            }

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetStreamUri_V2(std::string &info, const char *xaddr,const char *protocol,const char *token, const char *username = NULL, const char *password = NULL)
        {
            onvif_t *ctx = NULL;
            onvif_trt2_getstreamuri_rsq_t rsp = {0};
            int chk;

            assert(xaddr != NULL && protocol != NULL && token != NULL);

            info.clear();

            ctx = onvif_create(3);
            if (!ctx)
                return -1;

            if (username && password)
            {
                chk = onvif_setauthinfo(ctx, username, password);
                if (chk != 0)
                {
                    onvif_destroy(&ctx);
                    return -2;
                }
            }

            chk = onvif_getstreamuri_v2(ctx, xaddr, protocol,token,&rsp);
            if (chk != 0)
            {
                onvif_destroy(&ctx);
                return -3;
            }

            info = rsp.Uri;

            onvif_destroy(&ctx);

            return 0;
        }

        inline int GetStreamUri_V2(std::vector<std::map<std::string, std::string>> &info, const char *xaddr, const char *username = NULL, const char *password = NULL)
        {
            std::vector<std::string> service_info;
            std::vector<std::map<std::string, std::string>> profile_info;
            int chk;

            info.clear();

            chk = GetService(service_info, xaddr, username, password);
            if (chk != 0)
                return -1;

            for (auto &t : service_info)
            {
                chk = GetProfile_V2(profile_info, t.c_str(), username, password);
                if (chk != 0)
                    continue;

                for (auto &t2 : profile_info)
                {
                    std::map<std::string, std::string> tmp;

                    chk = GetStreamUri_V2(tmp[t2["Name"]], t.c_str(), "rtsp", t2["Token"].c_str(), username, password);
                    if (chk != 0)
                        continue;

                    info.push_back(tmp);
                }
            }

            return info.size();
        }

    } // namespace onvif
} // namespace aicontrib

#endif //ONVIF_ONVIF_HPP
