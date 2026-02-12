#!/bin/bash
set -e

echo "[$(date +%H:%M:%S)] "GAN vs. Human" Game - FULL REMOVE"

./stop.sh

# Remove containers (if any)
docker ps -aq --filter "name=gan_game" | xargs -r docker rm -f

# Remove images (all tags)
docker rmi -f $(docker images -q logus2k/gan_game.server) 2>/dev/null || true
docker rmi -f $(docker images -q logus2k/gan_game) 2>/dev/null || true

docker image prune -f >/dev/null

echo "[$(date +%H:%M:%S)] Remove complete."
