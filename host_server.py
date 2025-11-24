# Save as https_server.py in your Desktop or HTML folder
import http.server, ssl
httpd = http.server.HTTPServer(('0.0.0.0', 8443), http.server.SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket, certfile='cert.pem', keyfile='key.pem', server_side=True)
print('Serving HTTPS on port 8443')
httpd.serve_forever()

# https://129.97.71.84:8443/quest_controller.html