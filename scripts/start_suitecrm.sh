#!/usr/bin/env bash
# Start SuiteCRM Docker container and wait for it to be healthy.
# Run this before any trajectory collection.
#
# Usage: bash scripts/start_suitecrm.sh
set -e

COMPOSE_FILE="$(dirname "$0")/../suitecrm-compose.yaml"
SUITECRM_SETUP_COMPOSE="$(dirname "$0")/../../ST-WebAgentBench/suitecrm_setup/docker-compose.yaml"

# Prefer the benchmark's own compose if present (it has the demo data init-db)
if [ -f "$SUITECRM_SETUP_COMPOSE" ]; then
    COMPOSE="$SUITECRM_SETUP_COMPOSE"
    COMPOSE_DIR="$(dirname "$SUITECRM_SETUP_COMPOSE")"
    echo "Using ST-WebAgentBench suitecrm_setup compose: $COMPOSE"
else
    COMPOSE="$COMPOSE_FILE"
    COMPOSE_DIR="$(dirname "$COMPOSE_FILE")"
    echo "Using project compose: $COMPOSE"
fi

echo ""
echo "Starting SuiteCRM containers..."
docker compose -f "$COMPOSE" up -d

echo ""
echo "Waiting for SuiteCRM to be reachable at http://localhost:8080 ..."
for i in $(seq 1 30); do
    STATUS=$(curl -s --max-time 3 -o /dev/null -w "%{http_code}" http://localhost:8080 2>/dev/null || echo "000")
    if [ "$STATUS" -ge 200 ] 2>/dev/null && [ "$STATUS" -lt 500 ] 2>/dev/null; then
        echo "SuiteCRM is up (HTTP $STATUS)"
        break
    fi
    echo "  Attempt $i/30: HTTP $STATUS — waiting 5s..."
    sleep 5
done

echo ""
echo "Container status:"
docker compose -f "$COMPOSE" ps

echo ""
echo "If this is the first run, load the demo data:"
echo "  SETUP_COMPOSE=$SUITECRM_SETUP_COMPOSE"
echo "  docker exec -i suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 < \$(dirname \$SETUP_COMPOSE)/init-db/demo_data.sql"
