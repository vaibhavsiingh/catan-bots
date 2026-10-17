// Play mode: you (red) against the bot (blue).
"use strict";

let gid = null, geo = null, st = null, busy = false;
const LABELS = { US: "You", THEM: "Bot" };
const $ = id => document.getElementById(id);

function promptText() {
  if (st.winner) return st.winner === "US" ? "🎉 You win!" : "🤖 The bot wins.";
  if (st.current !== "US") return "🤖 Bot is thinking…";
  const types = new Set(st.actions.map(a => a.type));
  if (st.prompt === "BUILD_INITIAL_SETTLEMENT") return "Place a settlement: click a highlighted corner.";
  if (st.prompt === "BUILD_INITIAL_ROAD") return "Place a road next to it: click a highlighted edge.";
  if (st.prompt === "MOVE_ROBBER") return "Move the robber: click a highlighted tile.";
  if (st.prompt === "DISCARD") return "Too many cards after a 7: discard one at a time.";
  if (types.has("ROLL")) return "Your turn: roll the dice (or play a development card first).";
  return "Build (click the board), trade, buy, or end your turn.";
}

function render() {
  const mine = st.current === "US" && !st.winner;
  const pick = acts => {
    if (acts.length === 1) return act(acts[0]);
    choose("Steal from whom?", acts.map(a => ({ label: a.victim ? `Steal from ${LABELS[a.victim]}` : "Nobody", onclick: () => act(a) })));
  };
  const targets = mine ? boardTargets(st.actions, pick) : {};
  drawBoard($("board"), geo, st, { ...targets, showNodeIds: false });
  $("status").replaceChildren(
    el("div", { class: "row" }, el("span", { class: "dice", text: diceText(st) }),
      el("span", { class: "note", text: `Turn ${st.turn} · first to 10 VP` })),
    el("div", { class: "banner" + (mine ? "" : " them"), text: promptText() }));
  const ctl = $("controls");
  ctl.replaceChildren(el("h2", { text: "Actions" }));
  if (mine) ctl.append(actionButtons(st.actions, act));
  if (st.winner) ctl.append(el("button", { class: "primary big", on: { click: startDialog } }, "Play again"));
  $("players").replaceChildren(playersPanel(st, LABELS));
  $("log").replaceChildren(logPanel(st));
}

async function act(a) {
  if (busy) return;
  busy = true;
  try {
    st = (await api(`/api/play/${gid}/act`, { i: a.i, step: st.step })).state;
    render();
    await botLoop();
  } catch (e) { alert(e.message); await load(); }
  busy = false;
}

async function botLoop() {
  while (st.current !== "US" && !st.winner) {
    await new Promise(r => setTimeout(r, 450));
    st = (await api(`/api/play/${gid}/bot`, {})).state;
    render();
  }
}

async function load() {
  const j = await api(`/api/play/${gid}`);
  geo = j.geometry; st = j.state;
  render();
}

async function start(first) {
  const j = await api("/api/play/new", { first });
  gid = j.id; geo = j.geometry; st = j.state;
  location.hash = gid;
  render();
  busy = true;
  try { await botLoop(); } finally { busy = false; }
}

function startDialog() {
  choose("New game: who places first?", [
    { label: "🎲 Random", cls: "primary", onclick: () => start("random") },
    { label: "Me", onclick: () => start("you") },
    { label: "The bot", onclick: () => start("bot") },
  ], "Standard rules, 1v1, first to 10 victory points. No trading between players.");
}

$("newgame").addEventListener("click", startDialog);
(async () => {
  gid = location.hash.slice(1) || null;
  if (gid) {
    try { await load(); busy = true; await botLoop(); busy = false; return; } catch (e) { gid = null; }
  }
  startDialog();
})();
