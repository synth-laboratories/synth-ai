#!/bin/bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
PORT=8010
echo "Starting banking77 local API on port $PORT"
nohup python -c "
import uvicorn
from demos.gepa_banking77.localapi_banking77 import app
uvicorn.run(app, host='0.0.0.0', port=$PORT)
" > banking77_localapi.log 2>&1 &
echo "Local API started. PID: $!"
echo "Log file: banking77_localapi.log"
echo "Health check: http://localhost:$PORT/health"