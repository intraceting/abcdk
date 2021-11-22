/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/json.h"

#ifdef _json_h_

void abcdk_json_unref(json_object **obj)
{
    int chk;
    
    if(!obj || !*obj)
        return;

    chk = json_object_put(*obj);
    assert(chk == 1);

    /*Set to NULL(0).*/
    *obj = NULL;
}

json_object *abcdk_json_refer(json_object *obj)
{
    assert(obj != NULL);

    return json_object_get(obj);
}

json_object *abcdk_json_parse(const char *str)
{
    assert(str != NULL);

    return json_tokener_parse(str);
}

const char *abcdk_json_string(json_object *obj)
{
    assert(obj != NULL);

    return json_object_to_json_string(obj);
}

#endif //_json_h_
