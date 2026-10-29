"""Standalone CLI; localhost-only unless caller explicitly opts into exposure."""
import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description='Ball-MARL playable local web game')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'])
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error('port must be 1..65535')
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        print('WARNING: Web game has no authentication. Do not expose to untrusted networks.')
    uvicorn.run('game_server.app:app', host=args.host, port=args.port, log_level=args.log_level,
                workers=1, ws_max_size=4096)
