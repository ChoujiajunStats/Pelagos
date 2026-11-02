"""Length-delimited metadata and float-grid frames over an SSH tunnel."""
import json
import socket
import struct

MAX_HEADER = 65536
MAX_PAYLOAD = 36 * 1024 * 1024  # bounded native Wavemap + bounded float ROI


def send_packet(connection, metadata, payload=b''):
    header = json.dumps(metadata, allow_nan=False, separators=(',', ':')).encode()
    if len(header) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError('Planner packet exceeds configured bound')
    connection.sendall(struct.pack('!II', len(header), len(payload)) + header + payload)


def receive_packet(connection, stopped=lambda: False):
    def read(size):
        result = bytearray()
        while len(result) < size:
            if stopped():
                raise EOFError('Planner connection stopped')
            try:
                chunk = connection.recv(size - len(result))
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError('Planner connection closed')
            result.extend(chunk)
        return bytes(result)

    header_size, payload_size = struct.unpack('!II', read(8))
    if header_size > MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError('Planner packet exceeds configured bound')
    metadata = json.loads(read(header_size))
    if not isinstance(metadata, dict) or not isinstance(metadata.get('type'), str):
        raise ValueError('Invalid planner packet metadata')
    return metadata, read(payload_size)
