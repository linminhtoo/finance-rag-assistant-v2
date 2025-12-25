#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
# set default VLLM_API_KEY to "test" if not set
: "${VLLM_API_KEY:=test}"

curl http://localhost:8989/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -H "Authorization: Bearer ${VLLM_API_KEY}" \
	  -d '{
	    "model": "allenai/olmOCR-2-7B-1025",
	    "messages": [
	      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain the concept of AI in simple terms."}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}'
