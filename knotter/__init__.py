import http.server
import socketserver
import webbrowser
import os
import knotter.server
import multiprocessing

def static_server():
    port = 8000

    handler = http.server.SimpleHTTPRequestHandler

    httpd = socketserver.TCPServer(('', port), handler)

    print('Serving at port', port)
    here = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(here, 'static'))
    httpd.serve_forever()

def open_tab():
    webbrowser.open_new_tab('http://localhost:{}'.format(8000))

def run_server():
    process = [multiprocessing.Process(target=static_server),
            multiprocessing.Process(target=knotter.server.run),
            multiprocessing.Process(target=open_tab)]

    for p in process:
        p.start()

    for p in process:
        p.join()
