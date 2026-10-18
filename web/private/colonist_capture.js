// Feeds a colonist.io game to the advisor live, and records colonist's game messages.
//
// Use: start the advisor (python web/server.py --advisor), open colonist.io and the room (before the game starts),
// press F12 -> Console, paste this file, Enter. Play: the advisor (http://localhost:8001/advisor) opens the
// game by itself with the board, who is first and the setup placements filled in.
// The browser may ask to let colonist.io reach devices on your local network: allow it (that is the advisor).
// catanSave() downloads everything recorded as colonist_capture.json; catanCapture.length counts the messages.
(() => {
  if (window.catanCapture) return console.log("capture already running:", window.catanCapture.length, "messages");
  const ADVISOR = "http://localhost:8001/api/adv/colonist/push";
  const log = window.catanCapture = [];
  const seen = new WeakSet();
  const b64 = buf => { let s = ""; const u = new Uint8Array(buf); for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s); };
  // game messages are msgpack maps that start {id: "130", ...}
  const GAME = [0x82, 0xa2, 0x69, 0x64, 0xa3, 0x31, 0x33, 0x30];
  const isGame = buf => { const u = new Uint8Array(buf, 0, Math.min(8, buf.byteLength)); return GAME.every((x, i) => u[i] === x); };
  let queue = Promise.resolve(), blobs = Promise.resolve(), warned = false, last = null;
  const push = buf => {
    if (!isGame(buf)) return;
    const body = JSON.stringify({ b64: b64(buf) });
    queue = queue.then(() => fetch(ADVISOR, { method: "POST", headers: { "Content-Type": "text/plain" }, body })
      .then(r => r.json()).then(j => {
        if (j.error && j.error !== last) console.warn("advisor:", j.error);
        else if (j.name && j.name !== last) console.log("advisor: following game", j.name);
        last = j.error || j.name;
      })
      .catch(() => { if (!warned) { warned = true; console.warn("advisor not reachable at", ADVISOR, "(still recording)"); } }));
  };
  const record = (dir, ws, data) => {
    const e = { t: Date.now(), dir, url: ws.url };
    const keep = buf => { e.b64 = b64(buf); if (dir === "in") push(buf); };
    if (typeof data === "string") { e.text = data; log.push(e); }
    else if (data instanceof ArrayBuffer) { keep(data); log.push(e); }
    else if (ArrayBuffer.isView(data)) { keep(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)); log.push(e); }
    else if (data instanceof Blob) { log.push(e); blobs = blobs.then(() => data.arrayBuffer()).then(keep); }  // in arrival order
  };
  const watch = ws => {
    if (seen.has(ws)) return;
    seen.add(ws);
    ws.addEventListener("message", ev => record("in", ws, ev.data));
    console.log("catan capture: watching", ws.url);
  };
  // sockets opened from now on
  const Native = window.WebSocket;
  window.WebSocket = function (...args) { const ws = new Native(...args); watch(ws); return ws; };
  window.WebSocket.prototype = Native.prototype;
  Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  // sockets already open: caught the first time the page sends on them (heartbeats, every click)
  const send = Native.prototype.send;
  Native.prototype.send = function (data) { watch(this); record("out", this, data); return send.call(this, data); };
  window.catanSave = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(log)], { type: "application/json" }));
    a.download = "colonist_capture.json";
    a.click();
    console.log("saved", log.length, "messages");
  };
  console.log("catan capture running; sending the game to", ADVISOR, ". catanSave() downloads the recording");
})();
