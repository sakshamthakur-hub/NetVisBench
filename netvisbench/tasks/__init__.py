"""Task families. Each module exposes generate(out_dir, seed, per_difficulty)."""
from . import capture, cwnd, ipv4_header, subnet, tcp_seq, topology

FAMILIES = {
    "topology": topology,
    "subnet": subnet,
    "tcp_seq": tcp_seq,
    "cwnd": cwnd,
    "ipv4_header": ipv4_header,
    "capture": capture,
}
