"""Standalone network security guard hook for subprocess isolation."""

# Define NetworkAccessBlockedError matching domain exception hierarchy
class NetworkAccessBlockedError(PermissionError):
    """Raised when sandbox code attempts external network access."""
    def __init__(self, message="External network access is blocked by sandbox security policy."):
        super().__init__(message)
        self.message = message

def _blocked_network_call(*args, **kwargs):
    raise NetworkAccessBlockedError("External network access is blocked by sandbox security policy.")

# Patch low-level socket module
try:
    import socket
    socket.socket.connect = _blocked_network_call
    socket.socket.bind = _blocked_network_call
    socket.create_connection = _blocked_network_call
    socket.getaddrinfo = _blocked_network_call
    socket.gethostbyname = _blocked_network_call
    socket.gethostbyname_ex = _blocked_network_call
except Exception:
    pass

# Patch standard HTTP client modules
try:
    import urllib.request
    urllib.request.urlopen = _blocked_network_call
except Exception:
    pass

try:
    import http.client
    http.client.HTTPConnection.connect = _blocked_network_call
    http.client.HTTPSConnection.connect = _blocked_network_call
except Exception:
    pass

# Neutralize interactive display methods (e.g. Plotly / Matplotlib fig.show())
try:
    import plotly.graph_objects as _go
    _go.Figure.show = lambda self, *args, **kwargs: None
except Exception:
    pass

try:
    import matplotlib.pyplot as _plt
    _plt.show = lambda *args, **kwargs: None
except Exception:
    pass
