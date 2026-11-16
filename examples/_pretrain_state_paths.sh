#!/usr/bin/env bash
# Resolve published absolute dataset symlinks on fresh and existing mount layouts.
set -euo pipefail

LEGACY_ROOT="${1:-/share/mixed}"
MIXED_ROOT="${2:-/share-mixed}"

if [[ ! -d "$MIXED_ROOT" ]]; then
    echo "ERROR: mixed-storage mount not found: $MIXED_ROOT" >&2
    exit 1
fi
if [[ ! -e "$LEGACY_ROOT" && ! -L "$LEGACY_ROOT" ]]; then
    ln -s "$MIXED_ROOT" "$LEGACY_ROOT"
fi
for mount in project projset; do
    target="$MIXED_ROOT/$mount"
    alias="$LEGACY_ROOT/$mount"
    if [[ ! -d "$target/eai_pwm" ]]; then
        echo "ERROR: required mixed-storage directory not found: $target/eai_pwm" >&2
        exit 1
    fi
    if [[ ! -e "$alias" && ! -L "$alias" ]]; then
        ln -s "$target" "$alias"
    fi
    if [[ ! -d "$alias" ]]; then
        echo "ERROR: invalid legacy storage directory: $alias" >&2
        exit 1
    fi
    # Some images already contain an empty projset directory. Preserve it and
    # provide the published dataset prefix beneath it instead of replacing it.
    if [[ ! -e "$alias/eai_pwm" && ! -L "$alias/eai_pwm" ]]; then
        ln -s "$target/eai_pwm" "$alias/eai_pwm"
    fi
    if [[ "$(readlink -f "$alias/eai_pwm")" != "$(readlink -f "$target/eai_pwm")" ]]; then
        echo "ERROR: $alias/eai_pwm does not resolve to $target/eai_pwm" >&2
        exit 1
    fi
done
