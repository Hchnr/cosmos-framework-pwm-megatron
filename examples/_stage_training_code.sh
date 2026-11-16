#!/usr/bin/env bash
# Deploy the current working tree to a node-local working directory.
# NFS cwd lookup can fail during native-library initialization on some nodes.
set -euo pipefail
SOURCE_REPO="${1:?Pass the shared repository root}"
LOCAL_PARENT="${COSMOS_LOCAL_CODE_ROOT:-/opt/cosmos-framework-pwm}"
mkdir -p "$LOCAL_PARENT"
LOCAL_REPO="$(mktemp -d "$LOCAL_PARENT/run.XXXXXXXX")"

# Include pending source changes; exclude ignored data, models, outputs and venvs.
# A fresh directory prevents files removed from the source from surviving a deploy.
git -C "$SOURCE_REPO" ls-files -z --cached --others --exclude-standard |
    tar -C "$SOURCE_REPO" --null --no-recursion -T - -cf - |
    tar -C "$LOCAL_REPO" -xf -
git -C "$SOURCE_REPO" rev-parse HEAD > "$LOCAL_REPO/git_commit.txt"
git -C "$SOURCE_REPO" branch --show-current > "$LOCAL_REPO/git_branch.txt"
printf '%s\n' "$LOCAL_REPO"
