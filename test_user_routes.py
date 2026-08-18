import ssl
import time
import urllib.request
import urllib.error

ctx = ssl._create_unverified_context()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        return None


opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ctx),
    NoRedirect(),
)


urls = [
    "https://127.0.0.1:5000/users/edit/2",
    "https://127.0.0.1:5000/users/2/access",
]


for url in urls:
    print()
    print("TEST:", url)

    started = time.perf_counter()

    try:
        req = urllib.request.Request(
            url,
            method="GET",
        )

        response = opener.open(
            req,
            timeout=10,
        )

        elapsed = time.perf_counter() - started

        print("STATUS:", response.status)
        print("TIME:", round(elapsed, 3), "sec")
        print("LOCATION:", response.headers.get("Location"))

    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started

        print("STATUS:", exc.code)
        print("TIME:", round(elapsed, 3), "sec")
        print("LOCATION:", exc.headers.get("Location"))

    except Exception as exc:
        elapsed = time.perf_counter() - started

        print("ERROR:", type(exc).__name__, str(exc))
        print("TIME:", round(elapsed, 3), "sec")
