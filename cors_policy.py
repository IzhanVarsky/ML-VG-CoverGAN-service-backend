import functools

import cherrypy

white_hosts = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # "http://109.188.135.85:5001",
    "http://api.statanly.com:5001",
    "http://81.3.154.178:5001",
]


def make_cors():
    # print("Request headers:")
    # print(cherrypy.request.headers)

    if "Referer" in cherrypy.request.headers:
        host_origin = cherrypy.request.headers["Referer"]
        print("referer host_origin:", host_origin)
    elif "ORIGIN" in cherrypy.request.headers:
        host_origin = cherrypy.request.headers["ORIGIN"]
        print("host_origin:", host_origin)
    else:
        return
    host_origin = host_origin.rstrip("/")
    if host_origin in white_hosts:
        cherrypy.response.headers['Access-Control-Allow-Origin'] = host_origin
    cherrypy.response.headers["Access-Control-Allow-Headers"] = "*"
    cherrypy.response.headers["Access-Control-Allow-Methods"] = "*"


def set_cors_policy(func):
    @functools.wraps(func)
    def __wrapper(*args, **kwargs):
        make_cors()
        if cherrypy.request.method == "OPTIONS":
            return

        return func(*args, **kwargs)

    return __wrapper
