// Board drawing and shared UI pieces for both modes (play and advisor).
"use strict";

const RES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"];
const RES_NAME = { WOOD: "Lumber", BRICK: "Brick", SHEEP: "Wool", WHEAT: "Grain", ORE: "Ore" };
const RES_SHORT = { WOOD: "wo", BRICK: "br", SHEEP: "sh", WHEAT: "wh", ORE: "or" };
const RES_ICON = { WOOD: "🌲", BRICK: "🧱", SHEEP: "🐑", WHEAT: "🌾", ORE: "⛰️" };
const DEV_NAME = { KNIGHT: "Knight", ROAD_BUILDING: "Road Building", YEAR_OF_PLENTY: "Year of Plenty",
                   MONOPOLY: "Monopoly", VICTORY_POINT: "Victory Point" };
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...kids) {
  const e = tag.startsWith("svg:") ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (k === "text") e.textContent = v;
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.append(kid);
  return e;
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function edgeKey(e) { return e[0] < e[1] ? `${e[0]}-${e[1]}` : `${e[1]}-${e[0]}`; }

function pips(n) { return n ? 6 - Math.abs(7 - n) : 0; }

// Draw the board into `svg`.
//   geo: server geometry; st: server state (null for an empty template)
//   opts.nodes / opts.edges / opts.tiles / opts.ports: {key: {cls, onclick, title}} clickable targets
//   opts.colors: {US: css colour, THEM: css colour}; opts.tileLabel(t) -> text; opts.portLabel(p) -> text
function drawBoard(svg, geo, st, opts = {}) {
  svg.replaceChildren();
  const S = geo.size;
  const xs = Object.values(geo.nodes).map(p => p[0]), ys = Object.values(geo.nodes).map(p => p[1]);
  const pad = 62;
  svg.setAttribute("viewBox", `${Math.min(...xs) - pad} ${Math.min(...ys) - pad} ${Math.max(...xs) - Math.min(...xs) + 2 * pad} ${Math.max(...ys) - Math.min(...ys) + 2 * pad}`);
  const corner = [[0, -1], [0.866, -0.5], [0.866, 0.5], [0, 1], [-0.866, 0.5], [-0.866, -0.5]];
  const colors = opts.colors || { US: "var(--us)", THEM: "var(--them)" };

  // sea
  const sea = el("svg:g");
  for (const p of geo.ports) {
    const lab = opts.portLabel ? opts.portLabel(p) : (p.res ? `2:1 ${RES_ICON[p.res]}` : "3:1");
    const tgt = opts.ports && opts.ports[p.i];
    for (const n of p.nodes) {
      const q = geo.nodes[n];
      sea.append(el("svg:line", { x1: p.x, y1: p.y, x2: q[0], y2: q[1], class: "portline" }));
    }
    const g = el("svg:g", { class: "port" + (tgt ? " target " + (tgt.cls || "") : ""),
                            on: tgt ? { click: tgt.onclick } : {} });
    g.append(el("svg:circle", { cx: p.x, cy: p.y, r: 17, class: "portdisc" }));
    g.append(el("svg:text", { x: p.x, y: p.y + 4, class: "porttext", text: lab }));
    if (tgt && tgt.title) g.append(el("svg:title", { text: tgt.title }));
    sea.append(g);
  }
  svg.append(sea);

  // tiles
  for (const t of geo.tiles) {
    const pts = corner.map(([dx, dy]) => `${t.x + S * dx},${t.y + S * dy}`).join(" ");
    const tgt = opts.tiles && opts.tiles[t.i];
    const g = el("svg:g", { class: "tile" + (tgt ? " target " + (tgt.cls || "") : ""),
                            on: tgt ? { click: tgt.onclick } : {} });
    g.append(el("svg:polygon", { points: pts, class: "hex r-" + (t.res || "DESERT") }));
    if (opts.tileLabel) {
      g.append(el("svg:text", { x: t.x, y: t.y + 5, class: "tilelabel", text: opts.tileLabel(t) }));
    } else if (t.num) {
      g.append(el("svg:circle", { cx: t.x, cy: t.y, r: 13, class: "token" }));
      g.append(el("svg:text", { x: t.x, y: t.y + 3, class: "tokennum" + (t.num === 6 || t.num === 8 ? " hot" : ""), text: t.num }));
      g.append(el("svg:text", { x: t.x, y: t.y + 10, class: "tokenpips" + (t.num === 6 || t.num === 8 ? " hot" : ""), text: "•".repeat(pips(t.num)) }));
    }
    g.append(el("svg:text", { x: t.x, y: t.y - 21, class: "tileidx", text: "#" + t.i }));
    if (st && st.robber === t.i) {
      g.append(el("svg:g", { class: "robber" },
        el("svg:circle", { cx: t.x - 22, cy: t.y + 14, r: 6 }),
        el("svg:rect", { x: t.x - 27, y: t.y + 17, width: 10, height: 10, rx: 3 })));
    }
    if (tgt && tgt.title) g.append(el("svg:title", { text: tgt.title }));
    svg.append(g);
  }

  // edges: roads, then clickable targets
  const roads = new Map((st ? st.roads : []).map(r => [edgeKey(r.edge), r.color]));
  for (const e of geo.edges) {
    const k = edgeKey(e), a = geo.nodes[e[0]], b = geo.nodes[e[1]];
    const owner = roads.get(k);
    if (owner) svg.append(el("svg:line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "road", stroke: colors[owner] }));
    const tgt = opts.edges && opts.edges[k];
    if (tgt) {
      const g = el("svg:g", { class: "edgetarget " + (tgt.cls || ""), on: { click: tgt.onclick } });
      g.append(el("svg:line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "edgehit" }));
      g.append(el("svg:line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "edgeshow" }));
      if (tgt.title) g.append(el("svg:title", { text: tgt.title }));
      svg.append(g);
    }
  }

  // nodes: buildings, targets, numbers
  const builds = new Map((st ? st.buildings : []).map(b => [b.node, b]));
  for (const [n, p] of Object.entries(geo.nodes)) {
    const b = builds.get(+n), tgt = opts.nodes && opts.nodes[n];
    const g = el("svg:g", { class: "node" + (tgt ? " target " + (tgt.cls || "") : ""), on: tgt ? { click: tgt.onclick } : {} });
    if (b) {
      if (b.kind === "CITY") {
        g.append(el("svg:path", { d: `M${p[0] - 11},${p[1] + 8} v-10 l6,-6 l6,6 v-2 h10 v12 z`, fill: colors[b.color], class: "building" }));
      } else {
        g.append(el("svg:path", { d: `M${p[0] - 7},${p[1] + 7} v-8 l7,-7 l7,7 v8 z`, fill: colors[b.color], class: "building" }));
      }
    }
    if (tgt) g.append(el("svg:circle", { cx: p[0], cy: p[1], r: 9, class: "nodehit" }));
    if (opts.showNodeIds !== false && !b) g.append(el("svg:text", { x: p[0], y: p[1] + 3, class: "nodeid", text: n }));
    if (tgt && tgt.title) g.append(el("svg:title", { text: tgt.title }));
    svg.append(g);
  }
}

// Players panel: name, VP, cards, devs, badges.
function playersPanel(st, labels) {
  const box = el("div", { class: "players" });
  for (const [who, p] of Object.entries(st.players)) {
    const row = el("div", { class: "player " + who + (st.current === who && !st.winner ? " active" : "") });
    row.append(el("div", { class: "pname" }, el("span", { class: "swatch " + who }), labels[who] || who,
      el("span", { class: "vp", text: `${p.actual_vp !== undefined && p.actual_vp !== p.vp ? p.actual_vp : p.vp} VP` })));
    const bits = [`${p.cards} cards`, `${p.devs} dev`, `${p.knights} knights`, `road ${p.longest}`];
    if (p.road) bits.push("🛣️ Longest Road");
    if (p.army) bits.push("⚔️ Largest Army");
    row.append(el("div", { class: "pstats", text: bits.join(" · ") }));
    if (p.hand) {
      row.append(el("div", { class: "hand" }, RES.map(r => el("span", { class: "card r-" + r, title: RES_NAME[r] },
        el("b", { text: p.hand[r] }), " " + RES_ICON[r]))));
    }
    if (p.devlist && Object.keys(p.devlist).length) {
      row.append(el("div", { class: "pstats", text: "Dev cards: " + Object.entries(p.devlist).map(([d, n]) => `${DEV_NAME[d]}${n > 1 ? " ×" + n : ""}`).join(", ") }));
    }
    box.append(row);
  }
  return box;
}

function logPanel(st) {
  const box = el("div", { class: "log" });
  for (const line of st.log.slice().reverse()) box.append(el("div", { text: line }));
  return box;
}

function diceText(st) {
  if (!st.last_roll) return "";
  const face = ["", "⚀", "⚁", "⚂", "⚃", "⚄", "⚅"];
  return `${face[st.last_roll[0]]}${face[st.last_roll[1]]} ${st.last_roll[0] + st.last_roll[1]}`;
}

// Targets for the board from a list of actions: settlements/cities on nodes, roads on edges, robber on tiles.
// pick(actions) is called with the actions behind the clicked target (several for a robber tile with victims).
function boardTargets(actions, pick, cls = "") {
  const nodes = {}, edges = {}, tiles = {};
  const byTile = {};
  for (const a of actions) {
    if (a.node !== undefined) nodes[a.node] = { cls, title: a.desc, onclick: () => pick([a]) };
    else if (a.edge) edges[edgeKey(a.edge)] = { cls, title: a.desc, onclick: () => pick([a]) };
    else if (a.tile) (byTile[a.tile] = byTile[a.tile] || []).push(a);
  }
  for (const [t, acts] of Object.entries(byTile)) tiles[t] = { cls, title: acts.map(a => a.desc).join("\n"), onclick: () => pick(acts) };
  return { nodes, edges, tiles };
}

// A modal chooser: options = [{label, onclick, cls}]
function choose(title, options, note) {
  closeModal();
  const back = el("div", { class: "modal-back", id: "modal", on: { click: e => { if (e.target.id === "modal") closeModal(); } } });
  const box = el("div", { class: "modal" }, el("h3", { text: title }), note ? el("p", { class: "note", text: note }) : null);
  const grid = el("div", { class: "choices" });
  for (const o of options) grid.append(el("button", { class: o.cls || "", on: { click: () => { closeModal(); o.onclick(); } } }, o.label));
  box.append(grid, el("button", { class: "ghost", on: { click: closeModal } }, "Cancel"));
  back.append(box);
  document.body.append(back);
}
function closeModal() { const m = document.getElementById("modal"); if (m) m.remove(); }

// Group non-board actions into labelled buttons. onAct(action) handles a click.
function actionButtons(actions, onAct, extra = {}) {
  const box = el("div", { class: "actions" });
  const by = t => actions.filter(a => a.type === t);
  const one = (t, label, cls) => { for (const a of by(t)) box.append(el("button", { class: cls || "", on: { click: () => onAct(a) } }, label)); };
  one("ROLL", "🎲 Roll", "primary");
  one("PLAY_KNIGHT_CARD", "⚔️ Play Knight");
  one("PLAY_ROAD_BUILDING", "🛣️ Play Road Building");
  const yop = by("PLAY_YEAR_OF_PLENTY");
  if (yop.length) box.append(el("button", { on: { click: () => choose("Year of Plenty: take two", yop.map(a => ({
    label: a.res2.map(r => RES_ICON[r] + " " + RES_NAME[r]).join(" + "), onclick: () => onAct(a) }))) } }, "🎁 Play Year of Plenty"));
  const mono = by("PLAY_MONOPOLY");
  if (mono.length) box.append(el("button", { on: { click: () => choose("Monopoly: take every", mono.map(a => ({
    label: RES_ICON[a.res] + " " + RES_NAME[a.res], onclick: () => onAct(a) }))) } }, "💰 Play Monopoly"));
  one("BUY_DEVELOPMENT_CARD", "🃏 Buy development card");
  const trades = by("MARITIME_TRADE");
  if (trades.length) box.append(el("button", { on: { click: () => choose("Trade with the bank", trades.map(a => ({
    label: `${a.n} ${RES_ICON[a.give]} → 1 ${RES_ICON[a.get]}`, onclick: () => onAct(a) }))) } }, "🏦 Bank trade"));
  const disc = by("DISCARD_RESOURCE");
  if (disc.length) {
    box.append(el("div", { class: "note", text: "Discard one card:" }));
    for (const a of disc) box.append(el("button", { on: { click: () => onAct(a) } }, `${RES_ICON[a.res]} ${RES_NAME[a.res]}`));
  }
  if (extra.before) box.prepend(...extra.before);
  one("END_TURN", "⏭️ End turn", "end");
  return box;
}
