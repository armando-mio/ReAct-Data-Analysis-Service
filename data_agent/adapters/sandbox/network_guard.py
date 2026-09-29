"""Network security guard snippet for sandbox isolation.

Neutralizes socket and network access within executed sandbox scripts.
"""

import socket


class NetworkAccessBlockedError(RuntimeError, PermissionError):
    """Raised when sandbox execution attempts external network access."""

    def __init__(self, message: str = "External network access is prohibited in this sandbox.") -> None:
        super().__init__(message)
        self.message = message


def block_network() -> None:
    """Neutralize socket creations and connections inside the sandbox environment."""

    def _blocked(*args, **kwargs):
        raise NetworkAccessBlockedError("External network access is prohibited in this sandbox.")

    class BlockedSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            raise NetworkAccessBlockedError("External network access is prohibited in this sandbox.")

        def connect(self, *args, **kwargs):
            raise NetworkAccessBlockedError("External network access is prohibited in this sandbox.")

    socket.socket = BlockedSocket
    socket.create_connection = _blocked
