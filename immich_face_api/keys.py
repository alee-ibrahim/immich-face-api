"""Create or hash API keys.

  python -m immich_face_api.keys new            # prints a new random key on stdout (give it to the client)
  python -m immich_face_api.keys hash < keyfile  # prints the SHA-256 hex digest for FACE_API_API_KEYS

Keys are read from stdin, never from the command line (argv ends up in shell history and process lists).
"""

import sys

from .security import hash_key, new_key


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "new":
        print(new_key())
        return 0
    if cmd == "hash":
        key = sys.stdin.readline().strip()
        if len(key) < 32:
            print("refusing: key shorter than 32 characters", file=sys.stderr)
            return 1
        print(hash_key(key))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
