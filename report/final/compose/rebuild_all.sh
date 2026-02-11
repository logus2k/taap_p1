#!/bin/bash
set -e

echo "[$(date +%H:%M:%S)] "GAN vs. Human" Game - FULL CLEAN REBUILD"

./stop.sh

# Remove containers
docker ps -aq --filter "name=gan_game" | xargs -r docker rm -f

# Remove images
docker image inspect logus2k/gan_game.server:latest >/dev/null 2>&1 && docker rmi -f logus2k/gan_game.server:latest || true
docker image inspect logus2k/gan_game:latest        >/dev/null 2>&1 && docker rmi -f logus2k/gan_game:latest        || true

docker image prune -f >/dev/null

# Rebuild images (hard, no cache)
docker build --no-cache -t logus2k/gan_game.server:latest -f gan_game.server.Dockerfile ..
docker build --no-cache -t logus2k/gan_game:latest        -f gan_game.Dockerfile        ..

echo "[$(date +%H:%M:%S)] Rebuild complete. Run ./start.sh"
