#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# https://opentelemetry.io/docs/collector/install/binary/linux/#manual-linux-installation
# curl --proto '=https' --tlsv1.2 -fOL https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v0.141.0/otelcol_0.141.0_linux_amd64.tar.gz
# tar -xvf otelcol_0.141.0_linux_amd64.tar.gz

now=$(date +"%Y%m%d_%H%M%S")
./otelcol --config ./otelcol-config.yaml \
	    2>&1 | tee ../logs/otelcol_ingest_html_to_markdown.$now.log

# now=$(date +"%Y%m%d_%H%M%S")
# ./otelcol --config ./otelcol-config.yaml \
#     2>&1 | tee otelcol.$now.log

# check:
# curl -s -X POST http://127.0.0.1:4318/v1/traces \
#   -H 'Content-Type: application/json' \
#   -d '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"curl-test"}}]},"scopeSpans":[{"scope":{"name":"demo"},"spans":[{"traceId":"0af7651916cd43dd8448eb211c80319c","spanId":"b7ad6b7169203331","name":"test-span","kind":1,"startTimeUnixNano":"1690000000000000000","endTimeUnixNano":"1690000001000000000"}]}]}]}]}'

# you should see:
# 2025-12-13T18:09:21.152+0800    info    ResourceTraces #0 service.name=curl-test
# ScopeTraces #0 demo
# test-span 0af7651916cd43dd8448eb211c80319c b7ad6b7169203331
#         {"resource": {"service.instance.id": "15e2f4bc-bf2c-45f6-b816-2db1e06d9d99", "service.name": "otelcol", "service.version": "0.141.0"}, "otelcol.component.id": "debug", "otelcol.component.kind": "exporter", "otelcol.signal": "traces"}
