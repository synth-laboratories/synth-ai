"""Transport bound shared by direct Index and Index-only MCP clients."""

# The backend's bounded retrieval can be followed by separately bounded
# monitor delivery. A 30-second Research transport default is too short for
# a valid monitored Index response. Explicit caller timeouts still win.
INDEX_TRANSPORT_TIMEOUT_SECONDS = 120.0
