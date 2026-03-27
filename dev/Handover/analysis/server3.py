from http.server import HTTPServer, SimpleHTTPRequestHandler, test

import json
import os
import sys
import urllib.parse


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        # Provide a stable endpoint for the analysis viewer to discover XML files.
        # This avoids relying on directory listing HTML, which is suppressed when
        # index.html exists in the directory.
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/__list__":
            xml_files = sorted([name for name in os.listdir(os.getcwd()) if name.lower().endswith(".xml")])
            payload = json.dumps(xml_files).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        return super().do_GET()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test(
        CORSRequestHandler,
        HTTPServer,
        port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000,
        bind="127.0.0.1",
    )
