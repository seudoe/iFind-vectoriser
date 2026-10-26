#!/usr/bin/env bash
# Push HEAD to the HF Space with a GIT_SHA stamp so GET /version reports the deployed commit.
# usage: ./deploy_hf.sh   (needs the `hf-seudoe` git remote)
set -euo pipefail
SHA=$(git rev-parse HEAD)
URL=$(git remote get-url hf-seudoe)
T=$(mktemp -d)
git archive HEAD | tar -x -C "$T"
echo "$SHA" > "$T/GIT_SHA"
cd "$T"
git init -q -b main
git -c user.name=ci -c user.email=ci@users.noreply.github.com add -A
git -c user.name=ci -c user.email=ci@users.noreply.github.com commit -q -m "deploy $SHA"
git push -q --force "$URL" HEAD:main
echo "deployed ${SHA:0:7} to HF Space"
