#!/usr/bin/env sh
# curl example. The key is read from a file into a header file, so it never appears in argv or shell history.
#   FACE_API_URL=https://faces.example.org ./identify.sh client.key photo.jpg
set -eu
key_file=$1
photo=$2
hdr=$(mktemp)
trap 'rm -f "$hdr"' EXIT
chmod 600 "$hdr"
printf 'Authorization: Bearer %s\n' "$(cat "$key_file")" > "$hdr"
curl -sS --fail-with-body -H @"$hdr" -F "image=@${photo}" "${FACE_API_URL%/}/v1/identify?candidates=3"
