/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "util/http.h"


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
        if (abcdk_http_status_dict->code == code)
            return abcdk_http_status_dict->desc;
    }

    return NULL;
}