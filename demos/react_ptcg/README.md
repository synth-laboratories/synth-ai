### React PTCG (Human vs AI) Demo

Synth’s PTCG **React UI** currently lives in `overzealous/tcg_ui`, and it talks to the Rust backend `overzealous/tcg_server` over HTTP + WebSocket.

This README is a “demo wrapper” so it’s discoverable from `synth-ai/demos/`.

---

### Architecture (high level)

- **Backend**: `overzealous/tcg_server`
  - HTTP:
    - `POST /game/new`
    - `POST /game/new_ai` (creates **P1 human vs P2 AI**)
  - WebSocket:
    - `GET /game/:game_id/player/:player_id` (streams `GameState`, `Prompt`, `Event`, `GameOver`, …)
- **Frontend**: `overzealous/tcg_ui` (Vite/React)
  - Reads `VITE_TCG_API_BASE` to know where the server is
  - Can connect to a specific game with `VITE_TCG_GAME_ID` + `VITE_TCG_PLAYER`

---

### Start the server (`tcg_server`)

From `overzealous/`:

```bash
cargo run -p tcg_server
```

Notes:
- The server uses the local SQLite DBs under `overzealous/data/` by default (cards + server DB).

---

### Start the UI (`tcg_ui`)

From `overzealous/tcg_ui/`:

```bash
npm install
```

Then run the dev server (pointing at your Rust backend):

```bash
export VITE_TCG_API_BASE="http://localhost:3000"
npm run dev
```

---

### Create a Human vs AI game

The UI has flows for creating games / queueing matches, but the minimal API contract is:
- `POST /game/new_ai` to create a game id
- Connect via WebSocket:
  - `ws://.../game/<game_id>/player/P1` for the human player

If you want, I can add a tiny helper script in this demo folder that:
- calls `POST /game/new_ai`
- prints the `VITE_TCG_GAME_ID` + `VITE_TCG_PLAYER` env vars to paste into your shell

