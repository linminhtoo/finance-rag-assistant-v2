#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# https://opentelemetry.io/docs/collector/install/binary/linux/#manual-linux-installation
# curl --proto '=https' --tlsv1.2 -fOL https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.141.0/otelcol_0.141.0_linux_amd64.tar.gz
# tar -xvf otelcol_0.141.0_linux_amd64.tar.gz

script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
project_root="$(
  cd -- "$script_dir/.." >/dev/null 2>&1
  pwd
)"

otelcol_bin="${OTELCOL_BIN:-$script_dir/otelcol}"
otelcol_config="${OTELCOL_CONFIG:-$script_dir/otelcol-config.yaml}"

if [[ ! -x "$otelcol_bin" ]]; then
  echo "Error: otelcol binary not found/executable at: $otelcol_bin" >&2
  echo "Install otelcol and place it at scripts/otelcol, or set OTELCOL_BIN=/path/to/otelcol" >&2
  exit 1
fi
if [[ ! -f "$otelcol_config" ]]; then
  echo "Error: otelcol config not found at: $otelcol_config" >&2
  echo "Expected: scripts/otelcol-config.yaml (or set OTELCOL_CONFIG=/path/to/config.yaml)" >&2
  exit 1
fi

now=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$project_root/logs"
"$otelcol_bin" --config "$otelcol_config" \
2>&1 | tee "$project_root/logs/otelcol_app.$now.log"
  # 2>&1 | tee "$project_root/logs/otelcol_ingest_html_to_markdown.$now.log"

# check:
# curl -s -X POST http://127.0.0.1:4318/v1/traces \
#   -H 'Content-Type: application/json' \
#   -d '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"curl-test"}}]},"scopeSpans":[{"scope":{"name":"demo"},"spans":[{"traceId":"0af7651916cd43dd8448eb211c80319c","spanId":"b7ad6b7169203331","name":"test-span","kind":1,"startTimeUnixNano":"1690000000000000000","endTimeUnixNano":"1690000001000000000"}]}]}]}]}'

# you should see:
# 2025-12-13T18:09:21.152+0800    info    ResourceTraces #0 service.name=curl-test
# ScopeTraces #0 demo
# test-span 0af7651916cd43dd8448eb211c80319c b7ad6b7169203331
#         {"resource": {"service.instance.id": "15e2f4bc-bf2c-45f6-b816-2db1e06d9d99", "service.name": "otelcol", "service.version": "0.141.0"}, "otelcol.component.id": "debug", "otelcol.component.kind": "exporter", "otelcol.signal": "traces"}
