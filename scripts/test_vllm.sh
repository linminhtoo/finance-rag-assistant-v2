#!/bin/bash

curl http://localhost:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "allenai/olmOCR-2-7B-1025",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain the concept of AI in simple terms."}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}'
