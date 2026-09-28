"""Network security guard snippet injected into sandbox executions.

Neutralizes cross-platform network socket calls to prevent external communication.
"""

BOOTSTRAP_NETWORK_GUARD = '''# Sandbox Security Guard - Network Isolation
import socket
import sys

_original_socket_connect = socket.socket.connect

def _blocked_connect(self, *args, **kwargs):
    raise PermissionError("External network access is blocked by sandbox security policy.")

# Patch low-level socket connections
socket.socket.connect = _blocked_connect
if hasattr(socket, "create_connection"):
    socket.create_connection = _blocked_connect

# Also neutralize common standard libraries if imported
try:
    import urllib.request
    def _blocked_urlopen(*args, **kwargs):
        raise PermissionError("External network access is blocked by sandbox security policy.")
    urllib.request.urlopen = _blocked_urlopen
except ImportError:
    pass

try:
    import http.client
    def _blocked_http_connect(self, *args, **kwargs):
        raise PermissionError("External network access is blocked by sandbox security policy.")
    http.client.HTTPConnection.connect = _blocked_http_connect
    http.client.HTTPSConnection.connect = _blocked_http_connect
except ImportError:
    pass
'''
