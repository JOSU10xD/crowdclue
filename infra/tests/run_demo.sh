#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:3000"
ADMIN_SECRET="letmein"

echo "1) Register user 'jason'"
curl -s -X POST -H "Content-Type: application/json" -d '{"name":"jason","password":"secret"}' $BASE/api/register || true
echo; echo

echo "2) Login jason (get JWT)"
TOKEN=$(curl -s -X POST -H "Content-Type: application/json" -d '{"name":"jason","password":"secret"}' $BASE/api/login | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
echo "Token: $TOKEN" >/dev/stderr
echo

echo "3) Admin upload an image"
RESP=$(curl -s -X POST -H "x-admin-secret: $ADMIN_SECRET" -F "image=@/tmp/photo.jpg" -F "title=demo" $BASE/admin/images)
echo "Upload response: $RESP"
IMAGE_ID=$(echo "$RESP" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
echo "Image id: $IMAGE_ID"
echo

echo "4) User posts a clue to that image"
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"text":"possible clue"}' $BASE/api/images/$IMAGE_ID/clues
echo; echo

echo "5) List image detail"
curl -s $BASE/api/images/$IMAGE_ID
echo; echo

echo "6) Admin accepts the contributor (by contributor name)"
curl -s -X POST -H "Content-Type: application/json" -H "x-admin-secret: $ADMIN_SECRET" -d '{"contributor":"jason"}' $BASE/admin/images/$IMAGE_ID/clues/accept_by_contributor
echo; echo

echo "7) Scoreboard"
curl -s $BASE/api/scoreboard
echo; echo
