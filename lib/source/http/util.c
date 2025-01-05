/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/http/util.h"

/** HTTP状态码。*/
static struct _abcdk_http_status_dict
{
    uint32_t code;
    const char *desc;
} abcdk_http_status_dict[] = {
    {100, "100 Continue"},
    {101, "101 Switching Protocol"},
    {103, "103 Early Hints"},
    {200, "200 OK"},
    {201, "201 Created"},
    {203, "203 Non-Authoritative Information"},
    {204, "204 No Content"},
    {205, "205 Reset Content"},
    {206, "206 Partial Content"},
    {300, "300 Multiple Choices"},
    {301, "301 Moved Permanently"},
    {302, "302 Found"},
    {303, "303 See Other"},
    {304, "304 Not Modified"},
    {307, "307 Temporary Redirect"},
    {308, "308 Permanent Redirect"},
    {400, "400 Bad Request"},
    {401, "401 Unauthorized"},
    {402, "402 Payment Required"},
    {403, "403 Forbidden"},
    {404, "404 Not Found"},
    {405, "405 Method Not Allowed"},
    {406, "406 Not Acceptable"},
    {407, "407 Proxy Authentication Required"},
    {408, "408 Request Timeout"},
    {409, "409 Conflict"},
    {411, "411 Length Required"},
    {412, "412 Precondition Failed"},
    {413, "413 Payload Too Large"},
    {414, "414 URI Too Long"},
    {415, "415 Unsupported Media Type"},
    {416, "416 Range Not Satisfiable"},
    {417, "417 Expectation Failed"},
    {418, "418 I'm a teapot"},
    {422, "422 Unprocessable Entity"},
    {425, "425 Too Early"},
    {426, "426 Upgrade Required"},
    {428, "428 Precondition Required"},
    {429, "429 Too Many Requests"},
    {431, "431 Request Header Fields Too Large"},
    {451, "451 Unavailable For Legal Reasons"},
    {500, "500 Internal Server Error"},
    {501, "501 Not Implemented"},
    {502, "502 Bad Gateway"},
    {503, "503 Service Unavailable"},
    {504, "504 Gateway Timeout"},
    {505, "505 HTTP Version Not Supported"},
    {506, "506 Variant Also Negotiates"},
    {507, "507 Insufficient Storage"},
    {508, "508 Loop Detected"},
    {511, "511 Network Authentication Required"}
};

const char *abcdk_http_status_desc(uint32_t code)
{
    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_http_status_dict); i++)
    {
        if (abcdk_http_status_dict[i].code == code)
            return abcdk_http_status_dict[i].desc;
    }

    return NULL;
}

/** HTTP内容类型。*/
static struct _abcdk_http_content_type_dict
{
    const char *ext;
    const char *desc;
} abcdk_http_content_type_dict[] = {
    {".*", "application/octet-stream"},
    {".tif", "image/tiff"},
    {".001", "application/x-001"},
    {".301", "application/x-301"},
    {".323", "text/h323"},
    {".906", "application/x-906"},
    {".907", "drawing/907"},
    {".a11", "application/x-a11"},
    {".acp", "audio/x-mei-aac"},
    {".ai", "application/postscript"},
    {".aif", "audio/aiff"},
    {".aifc", "audio/aiff"},
    {".aiff", "audio/aiff"},
    {".anv", "application/x-anv"},
    {".asa", "text/asa"},
    {".asf", "video/x-ms-asf"},
    {".asp", "text/asp"},
    {".asx", "video/x-ms-asf"},
    {".au", "audio/basic"},
    {".avi", "video/avi"},
    {".awf", "application/vnd.adobe.workflow"},
    {".biz", "text/xml"},
    {".bmp", "application/x-bmp"},
    {".bot", "application/x-bot"},
    {".c4t", "application/x-c4t"},
    {".c90", "application/x-c90"},
    {".cal", "application/x-cals"},
    {".cat", "application/vnd.ms-pki.seccat"},
    {".cdf", "application/x-netcdf"},
    {".cdr", "application/x-cdr"},
    {".cel", "application/x-cel"},
    {".cer", "application/x-x509-ca-cert"},
    {".cg4", "application/x-g4"},
    {".cgm", "application/x-cgm"},
    {".cit", "application/x-cit"},
    {".class", "java/*"},
    {".cml", "text/xml"},
    {".cmp", "application/x-cmp"},
    {".cmx", "application/x-cmx"},
    {".cot", "application/x-cot"},
    {".crl", "application/pkix-crl"},
    {".crt", "application/x-x509-ca-cert"},
    {".csi", "application/x-csi"},
    {".css", "text/css"},
    {".cut", "application/x-cut"},
    {".dbf", "application/x-dbf"},
    {".dbm", "application/x-dbm"},
    {".dbx", "application/x-dbx"},
    {".dcd", "text/xml"},
    {".dcx", "application/x-dcx"},
    {".der", "application/x-x509-ca-cert"},
    {".dgn", "application/x-dgn"},
    {".dib", "application/x-dib"},
    {".dll", "application/x-msdownload"},
    {".doc", "application/msword"},
    {".dot", "application/msword"},
    {".drw", "application/x-drw"},
    {".dtd", "text/xml"},
    {".dwf", "Model/vnd.dwf"},
    {".dwf", "application/x-dwf"},
    {".dwg", "application/x-dwg"},
    {".dxb", "application/x-dxb"},
    {".dxf", "application/x-dxf"},
    {".edn", "application/vnd.adobe.edn"},
    {".emf", "application/x-emf"},
    {".eml", "message/rfc822"},
    {".ent", "text/xml"},
    {".epi", "application/x-epi"},
    {".eps", "application/x-ps"},
    {".eps", "application/postscript"},
    {".etd", "application/x-ebx"},
    {".exe", "application/x-msdownload"},
    {".fax", "image/fax"},
    {".fdf", "application/vnd.fdf"},
    {".fif", "application/fractals"},
    {".fo", "text/xml"},
    {".frm", "application/x-frm"},
    {".g4", "application/x-g4"},
    {".gbr", "application/x-gbr"},
    {".", "application/x-"},
    {".gif", "image/gif"},
    {".gl2", "application/x-gl2"},
    {".gp4", "application/x-gp4"},
    {".hgl", "application/x-hgl"},
    {".hmr", "application/x-hmr"},
    {".hpg", "application/x-hpgl"},
    {".hpl", "application/x-hpl"},
    {".hqx", "application/mac-binhex40"},
    {".hrf", "application/x-hrf"},
    {".hta", "application/hta"},
    {".htc", "text/x-component"},
    {".htm", "text/html"},
    {".html", "text/html"},
    {".htt", "text/webviewhtml"},
    {".htx", "text/html"},
    {".icb", "application/x-icb"},
    {".ico", "image/x-icon"},
    {".ico", "application/x-ico"},
    {".iff", "application/x-iff"},
    {".ig4", "application/x-g4"},
    {".igs", "application/x-igs"},
    {".iii", "application/x-iphone"},
    {".img", "application/x-img"},
    {".ins", "application/x-internet-signup"},
    {".isp", "application/x-internet-signup"},
    {".IVF", "video/x-ivf"},
    {".java", "java/*"},
    {".jfif", "image/jpeg"},
    {".jpe", "image/jpeg"},
    {".jpe", "application/x-jpe"},
    {".jpeg", "image/jpeg"},
    {".jpg", "image/jpeg"},
    {".jpg", "application/x-jpg"},
    {".js", "application/x-javascript"},
    {".jsp", "text/html"},
    {".la1", "audio/x-liquid-file"},
    {".lar", "application/x-laplayer-reg"},
    {".latex", "application/x-latex"},
    {".lavs", "audio/x-liquid-secure"},
    {".lbm", "application/x-lbm"},
    {".lmsff", "audio/x-la-lms"},
    {".ls", "application/x-javascript"},
    {".ltr", "application/x-ltr"},
    {".m1v", "video/x-mpeg"},
    {".m2v", "video/x-mpeg"},
    {".m3u", "audio/mpegurl"},
    {".m4e", "video/mpeg4"},
    {".mac", "application/x-mac"},
    {".man", "application/x-troff-man"},
    {".math", "text/xml"},
    {".mdb", "application/msaccess"},
    {".mdb", "application/x-mdb"},
    {".mfp", "application/x-shockwave-flash"},
    {".mht", "message/rfc822"},
    {".mhtml", "message/rfc822"},
    {".mi", "application/x-mi"},
    {".mid", "audio/mid"},
    {".midi", "audio/mid"},
    {".mil", "application/x-mil"},
    {".mml", "text/xml"},
    {".mnd", "audio/x-musicnet-download"},
    {".mns", "audio/x-musicnet-stream"},
    {".mocha", "application/x-javascript"},
    {".movie", "video/x-sgi-movie"},
    {".mp1", "audio/mp1"},
    {".mp2", "audio/mp2"},
    {".mp2v", "video/mpeg"},
    {".mp3", "audio/mp3"},
    {".mp4", "video/mpeg4"},
    {".mpa", "video/x-mpg"},
    {".mpd", "application/vnd.ms-project"},
    {".mpe", "video/x-mpeg"},
    {".mpeg", "video/mpg"},
    {".mpg", "video/mpg"},
    {".mpga", "audio/rn-mpeg"},
    {".mpp", "application/vnd.ms-project"},
    {".mps", "video/x-mpeg"},
    {".mpt", "application/vnd.ms-project"},
    {".mpv", "video/mpg"},
    {".mpv2", "video/mpeg"},
    {".mpw", "application/vnd.ms-project"},
    {".mpx", "application/vnd.ms-project"},
    {".mtx", "text/xml"},
    {".mxp", "application/x-mmxp"},
    {".net", "image/pnetvue"},
    {".nrf", "application/x-nrf"},
    {".nws", "message/rfc822"},
    {".odc", "text/x-ms-odc"},
    {".out", "application/x-out"},
    {".p10", "application/pkcs10"},
    {".p12", "application/x-pkcs12"},
    {".p7b", "application/x-pkcs7-certificates"},
    {".p7c", "application/pkcs7-mime"},
    {".p7m", "application/pkcs7-mime"},
    {".p7r", "application/x-pkcs7-certreqresp"},
    {".p7s", "application/pkcs7-signature"},
    {".pc5", "application/x-pc5"},
    {".pci", "application/x-pci"},
    {".pcl", "application/x-pcl"},
    {".pcx", "application/x-pcx"},
    {".pdf", "application/pdf"},
    {".pdf", "application/pdf"},
    {".pdx", "application/vnd.adobe.pdx"},
    {".pfx", "application/x-pkcs12"},
    {".pgl", "application/x-pgl"},
    {".pic", "application/x-pic"},
    {".pko", "application/vnd.ms-pki.pko"},
    {".pl", "application/x-perl"},
    {".plg", "text/html"},
    {".pls", "audio/scpls"},
    {".plt", "application/x-plt"},
    {".png", "image/png"},
    {".png", "application/x-png"},
    {".pot", "application/vnd.ms-powerpoint"},
    {".ppa", "application/vnd.ms-powerpoint"},
    {".ppm", "application/x-ppm"},
    {".pps", "application/vnd.ms-powerpoint"},
    {".ppt", "application/vnd.ms-powerpoint"},
    {".ppt", "application/x-ppt"},
    {".pr", "application/x-pr"},
    {".prf", "application/pics-rules"},
    {".prn", "application/x-prn"},
    {".prt", "application/x-prt"},
    {".ps", "application/x-ps"},
    {".ps", "application/postscript"},
    {".ptn", "application/x-ptn"},
    {".pwz", "application/vnd.ms-powerpoint"},
    {".r3t", "text/vnd.rn-realtext3d"},
    {".ra", "audio/vnd.rn-realaudio"},
    {".ram", "audio/x-pn-realaudio"},
    {".ras", "application/x-ras"},
    {".rat", "application/rat-file"},
    {".rdf", "text/xml"},
    {".rec", "application/vnd.rn-recording"},
    {".red", "application/x-red"},
    {".rgb", "application/x-rgb"},
    {".rjs", "application/vnd.rn-realsystem-rjs"},
    {".rjt", "application/vnd.rn-realsystem-rjt"},
    {".rlc", "application/x-rlc"},
    {".rle", "application/x-rle"},
    {".rm", "application/vnd.rn-realmedia"},
    {".rmf", "application/vnd.adobe.rmf"},
    {".rmi", "audio/mid"},
    {".rmj", "application/vnd.rn-realsystem-rmj"},
    {".rmm", "audio/x-pn-realaudio"},
    {".rmp", "application/vnd.rn-rn_music_package"},
    {".rms", "application/vnd.rn-realmedia-secure"},
    {".rmvb", "application/vnd.rn-realmedia-vbr"},
    {".rmx", "application/vnd.rn-realsystem-rmx"},
    {".rnx", "application/vnd.rn-realplayer"},
    {".rp", "image/vnd.rn-realpix"},
    {".rpm", "audio/x-pn-realaudio-plugin"},
    {".rsml", "application/vnd.rn-rsml"},
    {".rt", "text/vnd.rn-realtext"},
    {".rtf", "application/msword"},
    {".rtf", "application/x-rtf"},
    {".rv", "video/vnd.rn-realvideo"},
    {".sam", "application/x-sam"},
    {".sat", "application/x-sat"},
    {".sdp", "application/sdp"},
    {".sdw", "application/x-sdw"},
    {".sit", "application/x-stuffit"},
    {".slb", "application/x-slb"},
    {".sld", "application/x-sld"},
    {".slk", "drawing/x-slk"},
    {".smi", "application/smil"},
    {".smil", "application/smil"},
    {".smk", "application/x-smk"},
    {".snd", "audio/basic"},
    {".sol", "text/plain"},
    {".sor", "text/plain"},
    {".spc", "application/x-pkcs7-certificates"},
    {".spl", "application/futuresplash"},
    {".spp", "text/xml"},
    {".ssm", "application/streamingmedia"},
    {".sst", "application/vnd.ms-pki.certstore"},
    {".stl", "application/vnd.ms-pki.stl"},
    {".stm", "text/html"},
    {".sty", "application/x-sty"},
    {".svg", "text/xml"},
    {".swf", "application/x-shockwave-flash"},
    {".tdf", "application/x-tdf"},
    {".tg4", "application/x-tg4"},
    {".tga", "application/x-tga"},
    {".tif", "image/tiff"},
    {".tif", "application/x-tif"},
    {".tiff", "image/tiff"},
    {".tld", "text/xml"},
    {".top", "drawing/x-top"},
    {".torrent", "application/x-bittorrent"},
    {".tsd", "text/xml"},
    {".txt", "text/plain"},
    {".uin", "application/x-icq"},
    {".uls", "text/iuls"},
    {".vcf", "text/x-vcard"},
    {".vda", "application/x-vda"},
    {".vdx", "application/vnd.visio"},
    {".vml", "text/xml"},
    {".vpg", "application/x-vpeg005"},
    {".vsd", "application/vnd.visio"},
    {".vsd", "application/x-vsd"},
    {".vss", "application/vnd.visio"},
    {".vst", "application/vnd.visio"},
    {".vst", "application/x-vst"},
    {".vsw", "application/vnd.visio"},
    {".vsx", "application/vnd.visio"},
    {".vtx", "application/vnd.visio"},
    {".vxml", "text/xml"},
    {".wav", "audio/wav"},
    {".wax", "audio/x-ms-wax"},
    {".wb1", "application/x-wb1"},
    {".wb2", "application/x-wb2"},
    {".wb3", "application/x-wb3"},
    {".wbmp", "image/vnd.wap.wbmp"},
    {".wiz", "application/msword"},
    {".wk3", "application/x-wk3"},
    {".wk4", "application/x-wk4"},
    {".wkq", "application/x-wkq"},
    {".wks", "application/x-wks"},
    {".wm", "video/x-ms-wm"},
    {".wma", "audio/x-ms-wma"},
    {".wmd", "application/x-ms-wmd"},
    {".wmf", "application/x-wmf"},
    {".wml", "text/vnd.wap.wml"},
    {".wmv", "video/x-ms-wmv"},
    {".wmx", "video/x-ms-wmx"},
    {".wmz", "application/x-ms-wmz"},
    {".wp6", "application/x-wp6"},
    {".wpd", "application/x-wpd"},
    {".wpg", "application/x-wpg"},
    {".wpl", "application/vnd.ms-wpl"},
    {".wq1", "application/x-wq1"},
    {".wr1", "application/x-wr1"},
    {".wri", "application/x-wri"},
    {".wrk", "application/x-wrk"},
    {".ws", "application/x-ws"},
    {".ws2", "application/x-ws"},
    {".wsc", "text/scriptlet"},
    {".wsdl", "text/xml"},
    {".wvx", "video/x-ms-wvx"},
    {".xdp", "application/vnd.adobe.xdp"},
    {".xdr", "text/xml"},
    {".xfd", "application/vnd.adobe.xfd"},
    {".xfdf", "application/vnd.adobe.xfdf"},
    {".xhtml", "text/html"},
    {".xls", "application/vnd.ms-excel"},
    {".xls", "application/x-xls"},
    {".xlw", "application/x-xlw"},
    {".xml", "text/xml"},
    {".xpl", "audio/scpls"},
    {".xq", "text/xml"},
    {".xql", "text/xml"},
    {".xquery", "text/xml"},
    {".xsd", "text/xml"},
    {".xsl", "text/xml"},
    {".xslt", "text/xml"},
    {".xwd", "application/x-xwd"},
    {".x_b", "application/x-x_b"},
    {".sis", "application/vnd.symbian.install"},
    {".sisx", "application/vnd.symbian.install"},
    {".x_t", "application/x-x_t"},
    {".ipa", "application/vnd.iphone"},
    {".apk", "application/vnd.android.package-archive"},
    {".xap", "application/x-silverlight-app"}};

const char *abcdk_http_content_type_desc(const char *ext)
{
    const char *p;

    assert(ext != NULL);

    p = strrchr(ext, '.');
    if (!p)
        return NULL;

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_http_content_type_dict); i++)
    {
        if (abcdk_strcmp(abcdk_http_content_type_dict[i].ext,p,0)==0)
            return abcdk_http_content_type_dict[i].desc;
    }

    return abcdk_http_content_type_desc(".*");
}

void abcdk_http_auth_digest(abcdk_md5_t *ctx, const char *user, const char *pawd,
                            const char *method, const char *url, const char *realm, const char *nonce)
{
    char digest_ha1[33] = {0}, digest_ha2[33] = {0};

    assert(ctx != NULL && user != NULL && pawd != NULL && method != NULL && url != NULL && realm != NULL && nonce != NULL);

    /*user:realm:passwod*/
    abcdk_md5_reset(ctx);
    abcdk_md5_update(ctx, user, strlen(user));
    abcdk_md5_update(ctx, ":", 1);
    abcdk_md5_update(ctx, realm, strlen(realm));
    abcdk_md5_update(ctx, ":", 1);
    abcdk_md5_update(ctx, pawd, strlen(pawd));
    abcdk_md5_final2hex(ctx,digest_ha1,0);

    /*method:url*/
    abcdk_md5_reset(ctx);
    abcdk_md5_update(ctx, method, strlen(method));
    abcdk_md5_update(ctx, ":", 1);
    abcdk_md5_update(ctx, url, strlen(url));
    abcdk_md5_final2hex(ctx,digest_ha2,0);

    /*ha1:nonce:ha2*/
    abcdk_md5_reset(ctx);
    abcdk_md5_update(ctx, digest_ha1, 32);
    abcdk_md5_update(ctx, ":", 1);
    abcdk_md5_update(ctx, nonce, strlen(nonce));
    abcdk_md5_update(ctx, ":", 1);
    abcdk_md5_update(ctx, digest_ha2, 32);
}

void abcdk_http_parse_request_header0(const char *req, abcdk_object_t **method, abcdk_object_t **location, abcdk_object_t **version)
{
    const char *p = NULL, *p_next = NULL;

    assert(req != NULL);

    abcdk_object_unref(method);
    abcdk_object_unref(location);
    abcdk_object_unref(version);

    p_next = req;

    if (method)
    {
        *method = abcdk_strtok3(&p_next, " ", 1);
        if (!*method)
            return;
    }
    else
    {
        abcdk_strtok2(&p_next, " ", 1);
    }

    if (location)
    {
        *location = abcdk_strtok3(&p_next, " ", 1);
        if (!*location)
            return;
    }
    else
    {
        abcdk_strtok2(&p_next, " ", 1);
    }

    if (version)
    {
        *version = abcdk_strtok3(&p_next, "\r\n", 1);
        if (!*version)
            return;
    }
    else
    {
        abcdk_strtok2(&p_next, " ", 1);
    }
}

void abcdk_http_parse_form(abcdk_option_t *opt,const char *form)
{
    const char *p = NULL, *p_next = NULL;
    const char *p2 = NULL, *p2_next = NULL;
    abcdk_object_t *key = NULL, *val = NULL;

    assert(opt != NULL && form != NULL);

    p_next = form;

    for (;;)
    {
        p = abcdk_strtok2(&p_next, "&", 1);
        if (!p)
            break;

        abcdk_object_unref(&key);
        abcdk_object_unref(&val);

        p2_next = p;
        p2 = abcdk_strtok(&p2_next, "=");
        if (!p2)
            break;

        key = abcdk_url_decode2(p2, p2_next - p2, 0);
        if (!key)
            break;

        if (*p2_next == '=')
            p2_next += 1;

        p2 = abcdk_strtok(&p2_next, "&");
        if (!p2)
            break;

        val = abcdk_url_decode2(p2, p2_next - p2, 0);
        if (!val)
            break;

        abcdk_option_set(opt,key->pstrs[0], val->pstrs[0]);
    }

    abcdk_object_unref(&key);
    abcdk_object_unref(&val);
}

abcdk_object_t *abcdk_http_chunked_copyfrom(const void *data, size_t size)
{
    abcdk_object_t *obj = NULL;
    ssize_t pos = 0;
    int chk;

    obj = abcdk_object_alloc2(16 + 2 + size + 2);
    if (!obj)
        return NULL;

    chk = sprintf(obj->pstrs[0], "%zx\r\n", size);
    if (chk <= 0)
        goto final_error;

    pos += chk;

    if(data != NULL && size >0)
    {
        memcpy(obj->pstrs[0] + pos, data, size);
        pos += size;
    }

    memcpy(obj->pstrs[0] + pos, "\r\n", 2);
    pos += 2;

    /*修正格式化后的数据长度。*/
    obj->sizes[0] = pos;

    return obj;

final_error:

    abcdk_object_unref(&obj);
    return NULL;
}

abcdk_object_t *abcdk_http_chunked_vformat(int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    char hdr[19] = {0};
    ssize_t pos = 0;
    int chk;

    assert(max > 0 && fmt != NULL);

    obj = abcdk_object_alloc2(16 + 2 + max + 2);
    if (!obj)
        return NULL;

    /*头部长度。*/
    pos += 18;

    /*先格式化数据，计算出数据长度。*/
    chk = vsprintf(obj->pstrs[0] + pos, fmt, ap);
    if (chk <= 0)
        goto final_error;

    /*格式化长度，填充到块头部。*/
    sprintf(hdr, "%-16x\r\n", chk);
    memcpy(obj->pstrs[0], hdr, 18);

    /*累加长度。*/
    pos += chk;

    /*添加尾部。*/
    memcpy(obj->pstrs[0] + pos, "\r\n", 2);
    pos += 2;

    /*修正格式化后的数据长度。*/
    obj->sizes[0] = pos;

    return obj;

final_error:

    abcdk_object_unref(&obj);
    return NULL;
}

abcdk_object_t *abcdk_http_chunked_format(int max, const char *fmt, ...)
{
    abcdk_object_t *obj;

    assert(max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    obj = abcdk_http_chunked_vformat(max, fmt, ap);
    va_end(ap);

    return obj;
}

void abcdk_http_parse_auth(abcdk_option_t **opt,const char *auth)
{
    char method[100] = {0};
    const char *p, *p_next;
    abcdk_object_t *basic_de = NULL, *basic_de2 = NULL;
    abcdk_object_t *digest_de = NULL, *digest_de2 = NULL;
    abcdk_option_t *opt_p;

    assert(opt != NULL && auth != NULL);

    /*可以需要创建新的。*/
    if (*opt == NULL)
        opt_p = *opt = abcdk_option_alloc("");
    else
        opt_p = *opt;

    if (!opt_p)
        return;

    p_next = auth;

    p = abcdk_strtok(&p_next, " ");
    if(!p)
        goto END;

    //Basic,Digest,...
    strncpy(method, p, p_next - p);

    //1
    abcdk_option_set(opt_p,"auth-method",method);

    if (abcdk_strcmp(method, "Basic", 0) == 0)
    {
        p = abcdk_strtok2(&p_next, "\r", 1);
        if(!p)
            goto END;

        basic_de = abcdk_basecode_decode2(p,p_next-p,64);
        if(!basic_de)
            goto END;

        basic_de2 = abcdk_strtok2vector(basic_de->pstrs[0],":");
        if(!basic_de2)
            goto END;

        //2
        abcdk_option_set(opt_p,"username",basic_de2->pstrs[0]);
        abcdk_option_set(opt_p,"password",basic_de2->pstrs[1]);
    }
    else if (abcdk_strcmp(method, "Digest", 0) == 0)
    {
        p = abcdk_strtok2(&p_next, "\r", 1);
        if(!p)
            goto END;

        digest_de = abcdk_strtok2vector(p,", ");
        if(!digest_de)
            goto END;

        for (int i = 0; i< digest_de->numbers; i++)
        {
            digest_de2 = abcdk_strtok2pair(digest_de->pstrs[i],"=");
            if(!digest_de2)
                continue;

            abcdk_strtrim2(digest_de2->pstrs[0], isspace, "\"\'", 2);
            abcdk_strtrim2(digest_de2->pstrs[1], isspace, "\"\'", 2);

            //2
            abcdk_option_set(opt_p,digest_de2->pstrs[0],digest_de2->pstrs[1]);

            abcdk_object_unref(&digest_de2);
        }
    }

END:

    abcdk_object_unref(&basic_de);
    abcdk_object_unref(&basic_de2);
    abcdk_object_unref(&digest_de);
    abcdk_object_unref(&digest_de2);

}

int abcdk_http_check_auth(abcdk_option_t *opt, abcdk_http_auth_load_pawd_cb load_pawd_cb, void *opaque)
{
    abcdk_md5_t *md5_ctx;
    char pawd_buf[160] = {0};
    const char *auth_method = NULL, *http_method = NULL;
    const char *user = NULL, *pawd = NULL;
    const char *method = NULL, *uri = NULL, *realm = NULL, *nonce = NULL, *response = NULL;
    char digest_rsp[33] = {0};
    int chk;

    assert(opt != NULL && load_pawd_cb != NULL);

    http_method = abcdk_option_get(opt, "http-method", 0, "");
    auth_method = abcdk_option_get(opt, "auth-method", 0, "");
    user = abcdk_option_get(opt, "username", 0, "");
    if (!http_method || !auth_method || !user)
        return -22;

    chk = load_pawd_cb(opaque, user, pawd_buf);
    if (chk != 0)
        return -22;

    if (abcdk_strcmp(auth_method, "Basic", 0) == 0)
    {
        pawd = abcdk_option_get(opt, "password", 0, "");
        if (!pawd)
            return -22;

        if (abcdk_strcmp(pawd, pawd_buf, 1) == 0)
            return 0;
    }
    else if (abcdk_strcmp(auth_method, "Digest", 0) == 0)
    {
        uri = abcdk_option_get(opt, "uri", 0, "");
        realm = abcdk_option_get(opt, "realm", 0, "");
        nonce = abcdk_option_get(opt, "nonce", 0, "");
        response = abcdk_option_get(opt, "response", 0, "");
        if (!uri || !realm || !nonce || !response)
            return -22;

        md5_ctx = abcdk_md5_create();
        if (!md5_ctx)
            return -1;

        abcdk_http_auth_digest(md5_ctx, user, pawd_buf, http_method, uri, realm, nonce);
        abcdk_md5_final2hex(md5_ctx, digest_rsp, 0);

        chk = abcdk_strcmp(digest_rsp, response, 0);
        abcdk_md5_destroy(&md5_ctx);

        if (chk == 0)
            return 0;
    }

    return -1;
}
