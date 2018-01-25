#!/usr/bin/env bash

fswatch -o /Users/denemmy/projects/mirror_ai/web-server/web-tools/ |\
    xargs -n1 /Users/denemmy/projects/mirror_ai/web-server/web-tools/ci/synch.sh