#!/bin/bash
URL="http://localhost:8080/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ "$RESPONSE" == "200" ]; then
  echo "✅ Health check PASSED — app is healthy"
  exit 0
else
  echo "❌ Health check FAILED (got $RESPONSE) — triggering rollback"
  exit 1
fi