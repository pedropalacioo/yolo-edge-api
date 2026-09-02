#!/bin/bash
set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-$HOME/yolo-edge-api}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-6}"
HEALTH_WAIT="${HEALTH_WAIT:-10}"

cd "$DEPLOY_PATH"
PREVIOUS=$(docker inspect yolo-api --format '{{.Config.Image}}' 2>/dev/null || true)
echo "[INFO] Imagem atual: ${PREVIOUS:-nenhuma}"
docker compose pull yolo-api
docker compose up -d

for ((attempt = 1; attempt <= HEALTH_RETRIES; attempt++)); do
    sleep "$HEALTH_WAIT"
    if curl -sf "$HEALTH_URL" >/dev/null; then
        echo "[OK] Deploy concluído e health check aprovado."
        exit 0
    fi
    echo "[AVISO] Health check $attempt/$HEALTH_RETRIES falhou."
done

echo "[ERRO] Nova versão não ficou saudável."
if [[ -n "$PREVIOUS" ]]; then
    echo "[ROLLBACK] Restaurando $PREVIOUS"
    IMAGE="$PREVIOUS" docker compose up -d --force-recreate yolo-api
    curl -sf --retry "$HEALTH_RETRIES" --retry-delay "$HEALTH_WAIT" "$HEALTH_URL" >/dev/null
    echo "[ROLLBACK] Serviço restaurado."
fi
exit 1
