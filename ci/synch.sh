#!/usr/bin/env bash
rsync -vr -e 'ssh -i /Users/denemmy/projects/mirror_ai/amazon_keys/private_key.pem' \
    --exclude-from .gitignore /Users/denemmy/projects/mirror_ai/web-server/web-tools/ \
    denemmy@train4.mirror-ai.net:/data/train2/projects/web-tools