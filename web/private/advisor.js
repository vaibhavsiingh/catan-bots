// Advisor mode: mirror a real game. The bot plays red (US); you click the opponent's (blue) moves.
"use strict";

const $ = id => document.getElementById(id);
const LABELS = { US: "Bot (you play it)", THEM: "Opponent" };
const TILE_CODES = { WOOD: "wo", BRICK: "br", SHEEP: "sh", WHEAT: "wh", ORE: "or" };
const CODE_RES = { wo: "WOOD", br: "BRICK", sh: "SHEEP", wh: "WHEAT", or: "ORE" };
const TILE_LIMIT = { WOOD: 4, BRICK: 3, SHEEP: 4, WHEAT: 4, ORE: 3, DESERT: 1 };
const NUM_LIMIT = { 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 1 };
const PORT_LIMIT = { "3": 4, WOOD: 1, BRICK: 1, SHEEP: 1, WHEAT: 1, ORE: 1 };
let geo = null, st = null, name = null, busy = false, showAll = false;

function side(...kids) { $("side").replaceChildren(...kids); }
function panel(title, ...kids) { return el("div", { class: "card-panel" }, title ? el("h2", { text: title }) : null, ...kids); }

// ---------------------------------------------------------------- home

async function home() {
  name = null; location.hash = ""; $("gamename").textContent = "";
  const t = await api("/api/adv/template", {});
  geo = t.geometry;
  drawBoard($("board"), geo, null, { tileLabel: t => "#" + t.i, portLabel: p => "P" + p.i });
  const list = (await api("/api/adv/list")).games;
  const live = await api("/api/adv/colonist/live");
  const file = el("input", { type: "file", accept: ".json,application/json", on: { change: async () => {
    try {
      const capture = JSON.parse(await file.files[0].text());
      open((await api("/api/adv/colonist", { capture })).name);
    } catch (e) { alert(e.message); }
  } } });
  side(panel("New game", el("button", { class: "primary big", on: { click: editor } }, "＋ Enter a new board")),
       panel("From colonist.io",
         el("p", { class: "note", text: "Live: paste web/private/colonist_capture.js into the colonist tab's console (F12) before the game starts; the game opens here by itself, with the board, who is first and the setup placements." }),
         live.name ? el("button", { class: "primary", on: { click: () => open(live.name) } }, "Open live game " + live.name) : null,
         live.error ? el("p", { class: "note", text: "colonist feed: " + live.error }) : null,
         el("p", { class: "note", text: "Or import a saved recording (colonist_capture.json):" }), file),
       panel("Saved games", list.length ? el("div", { class: "actions" }, list.map(g =>
         el("button", { on: { click: () => open(g) } }, g))) : el("p", { class: "note", text: "None yet." })));
  liveSeen = live.name;
}

// Follow the colonist feed: open a new live game from the home screen, and pick up moves imported into the open game.
let liveSeen = null;
setInterval(async () => {
  if (busy || document.getElementById("modal")) return;
  try {
    if (!name) {
      if (document.getElementById("gname")) return;  // the board editor is open
      const live = await api("/api/adv/colonist/live");
      if (live.name && live.name !== liveSeen) open(live.name);
    } else if (st) {
      const j = await api(`/api/adv/${name}`);
      if ((j.state.entries !== st.entries || j.state.colonist !== st.colonist) && !busy) { st = j.state; render(); }
    }
  } catch (e) { /* server restarting */ }
}, 2000);

// ---------------------------------------------------------------- board editor

function editor() {
  const tiles = Array(19).fill(null);   // {res, num}
  const ports = Array(9).fill(null);    // "3" | resource
  let focus = { kind: "tile", i: 0 }, pendingRes = null, first = null;

  const used = () => {
    const r = {}, n = {}, p = {};
    for (const t of tiles) if (t) { r[t.res] = (r[t.res] || 0) + 1; if (t.num) n[t.num] = (n[t.num] || 0) + 1; }
    for (const x of ports) if (x) p[x] = (p[x] || 0) + 1;
    return { r, n, p };
  };
  const advance = () => {
    pendingRes = null;
    const nt = tiles.findIndex(t => !t), np = ports.findIndex(x => !x);
    focus = nt >= 0 ? { kind: "tile", i: nt } : np >= 0 ? { kind: "port", i: np } : { kind: "done" };
    draw();
  };
  const text = () => ({
    tiles: tiles.map(t => !t ? "?" : t.res === "DESERT" ? "de" : TILE_CODES[t.res] + t.num).join(" "),
    ports: ports.map(x => !x ? "?" : x === "3" ? "3" : TILE_CODES[x]).join(" "),
  });

  function draw() {
    const opts = {
      tileLabel: t => { const x = tiles[t.i - 1]; return x ? (x.res === "DESERT" ? "desert" : `${TILE_CODES[x.res]}${x.num}`) : (pendingRes && focus.kind === "tile" && focus.i === t.i - 1 ? TILE_CODES[pendingRes] || "de" : "?"); },
      portLabel: p => { const x = ports[p.i - 1]; return x ? (x === "3" ? "3:1" : "2:1 " + RES_ICON[x]) : "P" + p.i; },
      tiles: {}, ports: {},
    };
    geo.tiles.forEach(t => {
      opts.tiles[t.i] = { cls: focus.kind === "tile" && focus.i === t.i - 1 ? "current" : "", onclick: () => { focus = { kind: "tile", i: t.i - 1 }; pendingRes = null; draw(); } };
      if (tiles[t.i - 1]) { const fill = tiles[t.i - 1].res; t.res = fill === "DESERT" ? null : fill; } else t.res = "EMPTY";
    });
    geo.ports.forEach(p => opts.ports[p.i] = { cls: focus.kind === "port" && focus.i === p.i - 1 ? "current" : "", onclick: () => { focus = { kind: "port", i: p.i - 1 }; draw(); } });
    drawBoard($("board"), { ...geo, tiles: geo.tiles.map(t => ({ ...t, res: t.res === "EMPTY" ? "EMPTY" : t.res })) }, null, opts);
    // EMPTY tiles: drawBoard uses class r-EMPTY via res; desert has res null -> r-DESERT
    const u = used();
    const pick = el("div", {});
    if (focus.kind === "tile" && !pendingRes) {
      pick.append(el("p", { class: "note", text: `Tile #${focus.i + 1}: which resource? (red outline on the board)` }),
        el("div", { class: "actions" }, [...RES, "DESERT"].map(r => el("button", {
          disabled: (u.r[r] || 0) >= TILE_LIMIT[r] && !(tiles[focus.i] && tiles[focus.i].res === r) ? "" : null,
          on: { click: () => { if (r === "DESERT") { tiles[focus.i] = { res: "DESERT", num: null }; advance(); } else { pendingRes = r; draw(); } } } },
          r === "DESERT" ? "🏜️ Desert" : `${RES_ICON[r]} ${RES_NAME[r]}`))));
    } else if (focus.kind === "tile") {
      pick.append(el("p", { class: "note", text: `Tile #${focus.i + 1} ${RES_NAME[pendingRes]}: which number?` }),
        el("div", { class: "grid-dice" }, Object.keys(NUM_LIMIT).map(n => el("button", {
          disabled: (u.n[n] || 0) >= NUM_LIMIT[n] ? "" : null,
          on: { click: () => { tiles[focus.i] = { res: pendingRes, num: +n }; advance(); } } }, n))),
        el("button", { class: "ghost", on: { click: () => { pendingRes = null; draw(); } } }, "← resource"));
    } else if (focus.kind === "port") {
      pick.append(el("p", { class: "note", text: `Harbour P${focus.i + 1} (red ring): which kind?` }),
        el("div", { class: "actions" }, ["3", ...RES].map(r => el("button", {
          disabled: (u.p[r] || 0) >= PORT_LIMIT[r] && ports[focus.i] !== r ? "" : null,
          on: { click: () => { ports[focus.i] = r; advance(); } } }, r === "3" ? "3:1 (any)" : `2:1 ${RES_ICON[r]} ${RES_NAME[r]}`))));
    } else {
      pick.append(el("p", { text: "Board complete. Check it against your game, then start." }));
    }
    const ready = tiles.every(Boolean) && ports.every(Boolean);
    const nameIn = el("input", { type: "text", placeholder: "game name (optional)", id: "gname" });
    const firstRow = el("div", { class: "actions" },
      el("button", { class: first === "us" ? "sel" : "", on: { click: () => { first = "us"; draw(); } } }, "We place first"),
      el("button", { class: first === "them" ? "sel" : "", on: { click: () => { first = "them"; draw(); } } }, "They place first"));
    const startBtn = el("button", { class: "primary big", disabled: ready && first ? null : "", on: { click: async () => {
      try {
        const t = text();
        const j = await api("/api/adv/new", { ...t, first, name: $("gname").value.trim() || undefined });
        open(j.name);
      } catch (e) { alert(e.message); }
    } } }, "▶ Start game");
    const paste = el("input", { type: "text", placeholder: "or paste: tiles (19) | ports (9), e.g. wo12 or11 … | wo 3 wh …" });
    const pasteBtn = el("button", { on: { click: () => {
      const [tt, pp] = paste.value.split("|").map(s => (s || "").trim().split(/\s+/).filter(Boolean));
      if (tt && tt.length === 19) tt.forEach((x, i) => {
        const letters = x.replace(/\d/g, "").toLowerCase(), num = parseInt(x.replace(/\D/g, ""), 10);
        tiles[i] = ["de", "ro", "desert"].includes(letters) ? { res: "DESERT", num: null } : CODE_RES[letters.slice(0, 2)] ? { res: CODE_RES[letters.slice(0, 2)], num } : null;
      });
      if (pp && pp.length === 9) pp.forEach((x, i) => ports[i] = x === "3" ? "3" : CODE_RES[x.slice(0, 2).toLowerCase()] || null);
      advance();
    } } }, "Fill");
    side(panel("1. Board", pick, el("p", { class: "note", text: "Click a tile or harbour on the board to change it. Tiles go #1–#19 in reading order; harbours P1–P9 clockwise from the top." }),
               el("div", { class: "row" }, paste, pasteBtn)),
         panel("2. Who places first?", firstRow, el("p", { class: "note", text: "On colonist: if corners light up for you at the start, you're first." })),
         panel("3. Start", nameIn, el("div", { style: "height:8px" }), startBtn));
  }
  draw();
}

// ---------------------------------------------------------------- game

async function open(n) {
  name = n; location.hash = n; $("gamename").textContent = n;
  const j = await api(`/api/adv/${n}`);
  geo = j.geometry; st = j.state;
  render();
}

async function post(op, body) {
  if (busy) return;
  busy = true;
  try {
    const j = await api(`/api/adv/${name}/${op}`, body);
    st = j.state;
    showAll = false;
  } catch (e) { alert(e.message); const j = await api(`/api/adv/${name}`); st = j.state; }
  busy = false;
  render();
}

// Ask for what the engine can't know (dice, the card bought, the card stolen), then send the action.
function send(a) {
  const go = extra => post("act", { i: a.i, entries: st.entries, ...extra });
  if (a.needs === "dice") return diceDialog(d => go({ dice: d }), a);
  if (a.needs === "card") return choose("Which card did you get?", ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"].map(c => ({
    label: DEV_NAME[c], onclick: () => go({ card: c }) })));
  if (a.needs === "steal") return choose(a.victim === "THEM" ? "What did you steal?" : "What did they steal from you?",
    a.steal.map(r => ({ label: `${RES_ICON[r]} ${RES_NAME[r]}`, onclick: () => go({ steal: r }) })));
  go({});
}

function diceDialog(cb, a) {
  closeModal();
  const back = el("div", { class: "modal-back", id: "modal", on: { click: e => { if (e.target.id === "modal") closeModal(); } } });
  const grid = el("div", { class: "grid-dice" });
  for (let n = 2; n <= 12; n++) grid.append(el("button", { class: n === 7 ? "primary" : "", on: { click: () => { closeModal(); cb(n); } } }, String(n)));
  back.append(el("div", { class: "modal" }, el("h3", { text: (a && a.type === "ROLL" && st.current === "THEM" ? "They rolled" : "You rolled") + ": total?" }), grid,
    el("p", { class: "note", text: "Keys: 2–9, 0 = 10, - = 11, = = 12" }),
    el("button", { class: "ghost", on: { click: closeModal } }, "Cancel")));
  document.body.append(back);
  const keys = { "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "0": 10, "-": 11, "=": 12 };
  const onKey = e => { if (!document.getElementById("modal")) return document.removeEventListener("keydown", onKey);
    if (keys[e.key]) { document.removeEventListener("keydown", onKey); closeModal(); cb(keys[e.key]); } };
  document.addEventListener("keydown", onKey);
}

function pickTarget(acts) {
  if (acts.length === 1) return send(acts[0]);
  choose("Steal from whom?", acts.map(a => ({ label: a.victim ? `Steal from ${LABELS[a.victim]}` : "Nobody", onclick: () => send(a) })));
}

function render() {
  const us = st.current === "US";
  const advice = us && st.advice !== null ? st.actions[st.advice] : null;
  let targets = {};
  if (!st.winner && (!us || showAll)) targets = boardTargets(st.actions, pickTarget);
  else if (advice) targets = boardTargets([advice], pickTarget, "advice");
  drawBoard($("board"), geo, st, targets);

  const top = panel(null, el("div", { class: "row" }, el("span", { class: "dice", text: diceText(st) }),
    el("span", { class: "note", text: `Turn ${st.turn} · step ${st.entries}` })));
  if (st.colonist) top.append(el("div", { class: "banner them", text: "colonist: " + st.colonist + ". Enter the rest here by hand, or fix the step and it resumes." }));
  const ctl = panel(us ? "Bot's move" : "Opponent's move");
  if (st.winner) {
    ctl.append(el("div", { class: "banner", text: `${LABELS[st.winner]} reached ${10} VP by the engine's count.` }));
  } else if (us && advice) {
    ctl.append(el("div", { class: "banner", text: "🤖 " + advice.desc }));
    const label = advice.needs === "dice" ? "🎲 Rolled — enter dice" : advice.needs === "card" ? "🃏 Bought — which card?" :
      advice.needs === "steal" ? "✋ Done — what did you steal?" : "✓ Done (Enter)";
    ctl.append(el("div", { style: "height:8px" }), el("button", { class: "primary big", id: "donebtn", on: { click: () => send(advice) } }, label),
      el("button", { class: "ghost", on: { click: () => { showAll = !showAll; render(); } } }, showAll ? "Hide other moves" : "I did something else…"));
    if (showAll) ctl.append(el("p", { class: "note", text: "Click what you actually did on the board, or:" }), actionButtons(st.actions, send));
  } else {
    ctl.append(el("div", { class: "banner them", text: themPrompt() }));
    const reveal = st.them_hidden && st.prompt === "PLAY_TURN" ? [el("button", { on: { click: () => choose("They're playing which card?",
      ["KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"].map(c => ({ label: DEV_NAME[c], onclick: () => post("card", { card: c }) })),
      "Then click the play itself (it appears among their options).") } }, "🃏 They play a dev card…")] : [];
    ctl.append(el("div", { style: "height:8px" }), actionButtons(st.actions, send, { before: reveal }));
  }
  const tools = panel("Fix-ups", el("div", { class: "actions" },
    el("button", { on: { click: () => post("undo", {}) } }, "↶ Undo (u)"),
    el("button", { on: { click: tradeDialog } }, "🤝 Player trade"),
    el("button", { on: { click: adjDialog } }, "✏️ Correct a hand")));
  side(top, ctl, panel("Players", playersPanel(st, LABELS)), tools, panel("Log", logPanel(st)));
}

function themPrompt() {
  const p = st.prompt;
  if (p === "BUILD_INITIAL_SETTLEMENT") return "Click where they placed their settlement.";
  if (p === "BUILD_INITIAL_ROAD") return "Click the road they placed.";
  if (p === "MOVE_ROBBER") return "Click the tile they moved the robber to.";
  if (p === "DISCARD") return "They discard: click each card they discarded.";
  if (st.actions.some(a => a.type === "ROLL")) return "Their turn: enter their roll (or a dev card they play first).";
  return "Click what they built, or a button. End turn when it's your turn on colonist.";
}

function counterDialog(title, sides, onOk) {
  // sides: [{key, label}] -> counts per resource
  closeModal();
  const counts = {};
  const back = el("div", { class: "modal-back", id: "modal" });
  const box = el("div", { class: "modal" }, el("h3", { text: title }));
  for (const s of sides) {
    counts[s.key] = Object.fromEntries(RES.map(r => [r, 0]));
    const row = el("div", { class: "actions" });
    const redraw = () => row.replaceChildren(...RES.map(r => el("span", { class: "row" },
      el("button", { on: { click: () => { counts[s.key][r] -= s.signed || counts[s.key][r] > 0 ? 1 : 0; redraw(); } } }, "−"),
      el("span", { text: `${RES_ICON[r]} ${counts[s.key][r]}` }),
      el("button", { on: { click: () => { counts[s.key][r] += 1; redraw(); } } }, "+"))));
    redraw();
    box.append(el("p", { class: "note", text: s.label }), row);
  }
  box.append(el("div", { style: "height:10px" }), el("button", { class: "primary", on: { click: () => { closeModal(); onOk(counts); } } }, "OK"),
    el("button", { class: "ghost", on: { click: closeModal } }, "Cancel"));
  back.append(box);
  document.body.append(back);
}

function tradeDialog() {
  counterDialog("Trade between players", [{ key: "us", label: "We gave:" }, { key: "them", label: "They gave:" }],
    c => post("trade", { us: c.us, them: c.them }));
}

function adjDialog() {
  counterDialog("Correct a hand (the bank makes up the difference)", [{ key: "us", label: "Our hand: add (+) or remove (−)", signed: 1 },
    { key: "them", label: "Their hand: add (+) or remove (−)", signed: 1 }], async c => {
    for (const who of ["us", "them"]) for (const r of RES) if (c[who][r]) await post("adj", { who, res: r, n: c[who][r] });
  });
}

document.addEventListener("keydown", e => {
  if (document.getElementById("modal") || !name || e.target.tagName === "INPUT") return;
  if (e.key === "Enter" && $("donebtn")) $("donebtn").click();
  if (e.key === "u") post("undo", {});
});
$("home").addEventListener("click", home);
(location.hash.slice(1) ? open(location.hash.slice(1)).catch(home) : home());
