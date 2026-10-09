const state = {
  data: null,
  view: "overview",
  overviewGen: 1,
  transferOpponent: "max_damage",
  selectedTransfer: {train:1, eval:1},
  telemetry: {train:1, eval:1, opponent:"max_damage"},
  moveMode: "selected",
  inspector: {gen:1, pair:"pair01", train:1, opponent:"max_damage"}
};

const GEN_LABEL = {1:"Gen I",4:"Gen IV",9:"Gen IX"};
const GEN_COLOR = {1:"#f45b69",4:"#3b82f6",9:"#9c6ade"};
const METRIC_COLOR = ["#f45b69","#3b82f6","#5edc84","#ffd95b","#9c6ade","#ff984f","#39c6e8","#f472b6"];

const pct = x => `${(Number(x||0)*100).toFixed(1)}%`;
const score = x => Number(x||0).toFixed(3);
const pretty = s => String(s).replaceAll("_"," ").replace(/\b\w/g,c=>c.toUpperCase());
const clamp = (x,a=0,b=1) => Math.max(a,Math.min(b,Number(x)||0));

function one(list, pred){ return list.find(pred); }
function many(list,pred){ return list.filter(pred); }

function fillSelect(el, values, formatter=v=>v){
  el.innerHTML = values.map(v=>`<option value="${v}">${formatter(v)}</option>`).join("");
}

function metricRow(name, value, max=1, color="#3b82f6", note=""){
  const frac = max ? clamp(value/max,0,1) : 0;
  return `<div class="metric-row">
    <div><div class="metric-name">${name}</div>${note?`<div class="metric-note">${note}</div>`:""}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${frac*100}%;background:${color}"></div></div>
    <div class="metric-value">${max===1?pct(value):Number(value||0).toFixed(1)}</div>
  </div>`;
}

function currentBehavior(train, evalGen, opponent){
  return one(state.data.behavior, r =>
    +r.training_generation===+train &&
    +r.evaluation_generation===+evalGen &&
    r.opponent===opponent
  );
}

function currentDynamics(train, evalGen, opponent){
  return one(state.data.battleDynamics, r =>
    +r.training_generation===+train &&
    +r.evaluation_generation===+evalGen &&
    r.opponent===opponent
  );
}

function nav(){
  document.querySelectorAll(".nav-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      state.view=btn.dataset.view;
      document.querySelectorAll(".nav-btn").forEach(b=>b.classList.toggle("active",b===btn));
      document.querySelectorAll(".view").forEach(v=>v.classList.remove("active"));
      document.querySelector(`#view-${state.view}`).classList.add("active");
      render();
    });
  });
}

function renderOverview(){
  const gen=state.overviewGen;
  const opp="max_damage";
  const b=currentBehavior(gen,gen,opp);
  const d=currentDynamics(gen,gen,opp);
  const scores=state.data.transferMatrix.filter(r=>+r.training_generation===gen && +r.evaluation_generation===gen);

  document.querySelectorAll(".gen-chip").forEach(btn=>{
    btn.classList.toggle("active",+btn.dataset.gen===gen);
    btn.onclick=()=>{state.overviewGen=+btn.dataset.gen;renderOverview();}
  });

  document.getElementById("hudPolicy").textContent=GEN_LABEL[gen].toUpperCase();
  document.getElementById("hudOpponent").textContent="MAX DAMAGE";

  const md=one(scores,r=>r.opponent==="max_damage");
  const ta=one(scores,r=>r.opponent==="type_aware");
  document.getElementById("overviewQuickStats").innerHTML=`
    <div class="quick-stat"><strong>${score(md?.mean_mirror_score)}</strong><span>VS MAX DAMAGE</span></div>
    <div class="quick-stat"><strong>${score(ta?.mean_mirror_score)}</strong><span>VS TYPE AWARE</span></div>
    <div class="quick-stat"><strong>${d?Number(d.mean_mean_turns).toFixed(1):"—"}</strong><span>AVG TURNS</span></div>
    <div class="quick-stat"><strong>${b?Number(b.mean_mean_selected_base_power).toFixed(1):"—"}</strong><span>MOVE POWER</span></div>
  `;

  const ordered=["max_damage","type_aware"];
  document.getElementById("scoreboard").innerHTML=ordered.map((o,i)=>{
    const r=one(scores,x=>x.opponent===o);
    const v=r?.mean_mirror_score||0;
    return `<div class="score-row">
      <div class="score-label">${o==="max_damage"?"MAX":"TYPE"}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${v*100}%;background:${METRIC_COLOR[i]}"></div></div>
      <div class="bar-value">${score(v)}</div>
    </div>`;
  }).join("");

  const fp=[
    ["Damage",b?.mean_selected_damaging_rate,1,"#f45b69"],
    ["Status",b?.mean_selected_status_rate,1,"#9c6ade"],
    ["Buff",b?.mean_selected_any_buff_rate,1,"#5edc84"],
    ["Debuff",b?.mean_selected_any_debuff_rate,1,"#ff984f"],
    ["STAB",b?.mean_stab_move_rate,1,"#ffd95b"],
    ["Super-effective",b?.mean_super_effective_rate,1,"#39c6e8"],
  ];
  document.getElementById("fingerprintBars").innerHTML=fp.map(x=>metricRow(...x)).join("");

  const types=state.data.moveTypes
    .filter(r=>+r.training_generation===gen && +r.evaluation_generation===gen && r.opponent===opp)
    .sort((a,b)=>b.mean_selected_share-a.mean_selected_share).slice(0,6);
  document.getElementById("overviewTypeBars").innerHTML=types.map((r,i)=>
    metricRow(r.move_type,r.mean_selected_share,Math.max(...types.map(x=>x.mean_selected_share)),METRIC_COLOR[i%METRIC_COLOR.length],`selected share`)
  ).join("");

  const damage=b?.mean_selected_damaging_rate||0;
  const status=b?.mean_selected_status_rate||0;
  const narrative = damage > .9
    ? `${GEN_LABEL[gen]} policy is strongly damage-oriented: ${pct(damage)} of selected moves are damaging.`
    : `${GEN_LABEL[gen]} policy mixes direct damage (${pct(damage)}) with status play (${pct(status)}).`;
  document.getElementById("arenaNarrative").textContent=narrative;
}

function renderTransfer(){
  const sel=document.getElementById("transferOpponent");
  sel.value=state.transferOpponent;
  sel.onchange=()=>{state.transferOpponent=sel.value;renderTransfer();};

  const rows=state.data.transferMatrix.filter(r=>r.opponent===state.transferOpponent);
  const gens=[1,4,9];
  let html=`<div></div>${gens.map(g=>`<div class="matrix-label">${GEN_LABEL[g]}</div>`).join("")}`;
  for(const tr of gens){
    html+=`<div class="matrix-label">${GEN_LABEL[tr]}</div>`;
    for(const ev of gens){
      const r=one(rows,x=>+x.training_generation===tr && +x.evaluation_generation===ev);
      const v=r?.mean_mirror_score||0;
      const hue=210 - v*160;
      const selected=state.selectedTransfer.train===tr && state.selectedTransfer.eval===ev;
      html+=`<button class="matrix-cell ${selected?"selected":""}" data-tr="${tr}" data-ev="${ev}"
        style="background:hsl(${hue} 84% 72%)">
        <span><strong>${score(v)}</strong><small>${pct(v)}</small></span>
      </button>`;
    }
  }
  const matrix=document.getElementById("transferMatrix");
  matrix.innerHTML=html;
  matrix.querySelectorAll(".matrix-cell").forEach(btn=>{
    btn.onclick=()=>{
      state.selectedTransfer={train:+btn.dataset.tr,eval:+btn.dataset.ev};
      renderTransfer();
    };
  });

  const r=one(rows,x=>+x.training_generation===state.selectedTransfer.train && +x.evaluation_generation===state.selectedTransfer.eval);
  const gap=one(state.data.transferGap,x=>x.opponent===state.transferOpponent &&
    +x.training_generation===state.selectedTransfer.train && +x.evaluation_generation===state.selectedTransfer.eval);
  document.getElementById("transferCellDetail").innerHTML=`
    <div class="eyebrow">${GEN_LABEL[state.selectedTransfer.train]} → ${GEN_LABEL[state.selectedTransfer.eval]}</div>
    <div class="big-score">${score(r?.mean_mirror_score)}</div>
    <p>${state.transferOpponent==="max_damage"?"Against MaxDamage":"Against TypeAware"} across ${r?.n_seeds||0} seeds.</p>
    <div class="detail-grid">
      <div class="detail-chip"><span>Transfer gap</span><strong>${gap?(gap.mean_transfer_gap>=0?"+":"")+pct(gap.mean_transfer_gap):"—"}</strong></div>
      <div class="detail-chip"><span>SD</span><strong>${r?Number(r.sample_sd_mirror_score).toFixed(3):"—"}</strong></div>
      <div class="detail-chip"><span>Min</span><strong>${r?score(r.min_mirror_score):"—"}</strong></div>
      <div class="detail-chip"><span>Max</span><strong>${r?score(r.max_mirror_score):"—"}</strong></div>
    </div>
  `;

  const off=state.data.transferGap
    .filter(x=>x.opponent===state.transferOpponent && !x.is_in_domain)
    .sort((a,b)=>b.mean_transfer_gap-a.mean_transfer_gap);
  const maxAbs=Math.max(...off.map(x=>Math.abs(x.mean_transfer_gap)),.01);
  document.getElementById("transferGapBars").innerHTML=off.map((x,i)=>
    metricRow(`${GEN_LABEL[x.training_generation]} → ${GEN_LABEL[x.evaluation_generation]}`,
      Math.abs(x.mean_transfer_gap),maxAbs,x.mean_transfer_gap>=0?"#5edc84":"#f45b69",
      `${x.mean_transfer_gap>=0?"+":""}${pct(x.mean_transfer_gap)} vs native`)
  ).join("");

  const zs=state.data.zeroShot.filter(x=>x.opponent===state.transferOpponent);
  document.getElementById("zeroShotCards").innerHTML=zs.map(x=>`
    <div class="zero-card">
      <span>${GEN_LABEL[x.training_generation]}</span>
      <strong>${score(x.mean_off_diagonal_score)}</strong>
      <small>mean off-diagonal score</small>
    </div>`).join("");
}

function renderTelemetry(){
  const t=state.telemetry;
  ["telemetryTrain","telemetryEval"].forEach(id=>{
    const el=document.getElementById(id);
    el.value=id==="telemetryTrain"?t.train:t.eval;
  });
  document.getElementById("telemetryOpponent").value=t.opponent;

  const b=currentBehavior(t.train,t.eval,t.opponent);
  const d=currentDynamics(t.train,t.eval,t.opponent);

  const metrics=[
    ["Damage chosen",b?.mean_selected_damaging_rate,1,"#f45b69",`when available: ${pct(b?.mean_selection_given_available_damaging)}`],
    ["Status chosen",b?.mean_selected_status_rate,1,"#9c6ade",`when available: ${pct(b?.mean_selection_given_available_status)}`],
    ["Buff chosen",b?.mean_selected_any_buff_rate,1,"#5edc84",`when available: ${pct(b?.mean_selection_given_available_any_buff)}`],
    ["Debuff chosen",b?.mean_selected_any_debuff_rate,1,"#ff984f",`when available: ${pct(b?.mean_selection_given_available_any_debuff)}`],
    ["STAB",b?.mean_stab_move_rate,1,"#ffd95b","selected move rate"],
    ["Super-effective",b?.mean_super_effective_rate,1,"#39c6e8","selected move rate"],
    ["Resisted",b?.mean_resisted_rate,1,"#f472b6","selected move rate"],
    ["Mean base power",b?.mean_mean_selected_base_power,120,"#3b82f6","scaled to 120"],
  ];
  document.getElementById("telemetryMetrics").innerHTML=metrics.map(x=>metricRow(...x)).join("");

  renderMoveExplorer();

  const types=state.data.moveTypes
    .filter(r=>+r.training_generation===t.train && +r.evaluation_generation===t.eval && r.opponent===t.opponent)
    .sort((a,b)=>b.mean_selected_share-a.mean_selected_share).slice(0,8);
  const max=Math.max(...types.map(x=>x.mean_selected_share),.01);
  document.getElementById("typeExplorer").innerHTML=types.map((r,i)=>
    metricRow(r.move_type,r.mean_selected_share,max,METRIC_COLOR[i%METRIC_COLOR.length],`${pct(r.mean_selection_given_available)} when available`)
  ).join("");

  const cards=[
    ["Win rate",pct(d?.mean_raw_win_rate_rl)],
    ["Avg turns",d?Number(d.mean_mean_turns).toFixed(1):"—"],
    ["Avg switches",d?Number(d.mean_mean_switches_rl).toFixed(2):"—"],
    ["Voluntary switches",d?Number(d.mean_mean_voluntary_switches_rl).toFixed(2):"—"],
    ["Own remaining",d?Number(d.mean_mean_remaining_rl).toFixed(2):"—"],
    ["Opponent reveal",pct(d?.mean_mean_opponent_reveal_fraction)],
  ];
  document.getElementById("dynamicsCards").innerHTML=cards.map(([a,b])=>`
    <div class="stat-card"><strong>${b}</strong><span>${a}</span></div>`).join("");
}

function renderMoveExplorer(){
  const t=state.telemetry;
  const rows=state.data.moves.filter(r=>+r.training_generation===t.train && +r.evaluation_generation===t.eval && r.opponent===t.opponent);
  const key=state.moveMode==="selected"?"mean_selected_share":"mean_selection_given_available";
  const top=[...rows].sort((a,b)=>(b[key]||0)-(a[key]||0)).slice(0,10);
  const max=Math.max(...top.map(x=>x[key]||0),.01);
  document.getElementById("moveExplorer").innerHTML=top.map(r=>`
    <div class="move-row">
      <code>${pretty(r.move_id)}</code>
      <div class="move-mini-bar"><span style="width:${(r[key]||0)/max*100}%"></span></div>
      <strong>${pct(r[key])}</strong>
    </div>
  `).join("");
}

function renderGenerations(){
  const gens=[1,4,9];
  document.getElementById("generationCards").innerHTML=gens.map(g=>{
    const d=one(state.data.dexOverview,r=>+r.generation===g);
    return `<article class="panel pixel-frame generation-card">
      <div class="gen-card-head gen${g}">
        <div class="eyebrow" style="color:white">GENERATION PROFILE</div>
        <h3>${GEN_LABEL[g]}</h3>
        <span>${g===1?"The Origins":g===4?"The Expansion":"The Modern System"}</span>
      </div>
      <div class="gen-card-body">
        <div class="dna-row"><span>Base species</span><strong>${d.pokemon_base_species_entries}</strong></div>
        <div class="dna-row"><span>Standard moves</span><strong>${d.move_entries_standard}</strong></div>
        <div class="dna-row"><span>Dual-type share</span><strong>${pct(d.dual_type_share)}</strong></div>
        <div class="dna-row"><span>Damaging moves</span><strong>${pct(d.damaging_move_share)}</strong></div>
        <div class="dna-row"><span>Status moves</span><strong>${pct(d.status_move_share)}</strong></div>
        <div class="dna-row"><span>Buff-capable moves</span><strong>${pct(d.any_buff_share)}</strong></div>
        <div class="dna-row"><span>Debuff-capable moves</span><strong>${pct(d.any_debuff_share)}</strong></div>
      </div>
    </article>`;
  }).join("");

  const features=["damaging","status","any_buff","any_debuff","hazard_set","pivot_like","priority"];
  document.getElementById("poolFeatureCompare").innerHTML=features.map(f=>{
    return `<div class="compare-row">
      <strong>${pretty(f)}</strong>
      ${gens.map(g=>{
        const r=one(state.data.fixedPoolFeatures,x=>+x.generation===g && x.feature===f);
        return `<div class="compare-block gen${g}">${pct(r?.share_of_move_slots)}</div>`;
      }).join("")}
    </div>`;
  }).join("");

  const allTypes=[...new Set(state.data.fixedPoolPokemonTypes.map(x=>x.pokemon_type))];
  const topTypes=allTypes.map(type=>{
    const total=gens.reduce((s,g)=>{
      const r=one(state.data.fixedPoolPokemonTypes,x=>+x.generation===g && x.pokemon_type===type);
      return s+(r?.share_of_type_slots||0);
    },0);
    return {type,total};
  }).sort((a,b)=>b.total-a.total).slice(0,8);

  document.getElementById("poolTypeCompare").innerHTML=topTypes.map(({type})=>`
    <div class="compare-row">
      <strong>${type}</strong>
      ${gens.map(g=>{
        const r=one(state.data.fixedPoolPokemonTypes,x=>+x.generation===g && x.pokemon_type===type);
        return `<div class="compare-block gen${g}">${pct(r?.share_of_type_slots)}</div>`;
      }).join("")}
    </div>`).join("");
}

function rosterHTML(rows){
  return rows.sort((a,b)=>a.team_slot-b.team_slot).map(r=>{
    const type=r.type_1||"UNKNOWN";
    const initials=String(r.species).slice(0,2).toUpperCase();
    return `<div class="roster-mon">
      <div class="roster-icon type-${type}">${initials}</div>
      <div>
        <strong>${r.species}</strong>
        <small>${type}${r.type_2?` / ${r.type_2}`:""}</small>
      </div>
    </div>`;
  }).join("");
}

function renderInspector(){
  const s=state.inspector;
  ["inspectGen","inspectTrain"].forEach(id=>{
    document.getElementById(id).value=id==="inspectGen"?s.gen:s.train;
  });
  document.getElementById("inspectOpponent").value=s.opponent;
  document.getElementById("inspectPair").value=s.pair;

  const pool=state.data.fixedPoolPokemon.filter(r=>+r.generation===s.gen && r.pair_id===s.pair);
  const teamA=pool.filter(r=>r.side==="A");
  const teamB=pool.filter(r=>r.side==="B");
  document.getElementById("teamA").innerHTML=rosterHTML(teamA);
  document.getElementById("teamB").innerHTML=rosterHTML(teamB);

  const b=currentBehavior(s.train,s.gen,s.opponent);
  const d=currentDynamics(s.train,s.gen,s.opponent);
  document.getElementById("inspectorSummary").innerHTML=`
    <span class="cursor-dot">▶</span>
    ${GEN_LABEL[s.train]} policy enters ${GEN_LABEL[s.gen]} using fixed ${s.pair}.
    Aggregate profile: ${pct(b?.mean_selected_damaging_rate)} damaging,
    ${pct(b?.mean_selected_status_rate)} status, ${Number(b?.mean_mean_selected_base_power||0).toFixed(1)} mean base power.
  `;

  document.getElementById("inspectMetrics").innerHTML=[
    ["Damage",b?.mean_selected_damaging_rate,1,"#f45b69"],
    ["Status",b?.mean_selected_status_rate,1,"#9c6ade"],
    ["Buff",b?.mean_selected_any_buff_rate,1,"#5edc84"],
    ["Debuff",b?.mean_selected_any_debuff_rate,1,"#ff984f"],
    ["STAB",b?.mean_stab_move_rate,1,"#ffd95b"],
    ["Super-effective",b?.mean_super_effective_rate,1,"#39c6e8"],
  ].map(x=>metricRow(...x)).join("");

  document.getElementById("pairNotes").innerHTML=`
    <p><strong>${s.pair.toUpperCase()}</strong> belongs to the frozen ${GEN_LABEL[s.gen]} evaluation pool.</p>
    <p>Team A and Team B above are real benchmark rosters from the telemetry catalog.</p>
    <p>Against <strong>${pretty(s.opponent)}</strong>, this train→test condition averages
    <strong>${pct(d?.mean_raw_win_rate_rl)}</strong> raw wins and
    <strong>${d?Number(d.mean_mean_turns).toFixed(1):"—"}</strong> turns.</p>
    <p><strong>Replay note:</strong> the aggregate ZIP does not contain raw per-turn action logs, so this MVP does not invent a fake replay.</p>
  `;
}

function render(){
  if(state.view==="overview")renderOverview();
  if(state.view==="transfer")renderTransfer();
  if(state.view==="telemetry")renderTelemetry();
  if(state.view==="generations")renderGenerations();
  if(state.view==="inspector")renderInspector();
}

function bindFilters(){
  const gens=[1,4,9];
  fillSelect(document.getElementById("transferOpponent"),state.data.meta.opponents,pretty);
  fillSelect(document.getElementById("telemetryTrain"),gens,g=>GEN_LABEL[g]);
  fillSelect(document.getElementById("telemetryEval"),gens,g=>GEN_LABEL[g]);
  fillSelect(document.getElementById("telemetryOpponent"),state.data.meta.opponents,pretty);
  fillSelect(document.getElementById("inspectGen"),gens,g=>GEN_LABEL[g]);
  fillSelect(document.getElementById("inspectTrain"),gens,g=>GEN_LABEL[g]);
  fillSelect(document.getElementById("inspectOpponent"),state.data.meta.opponents,pretty);

  const pairs=[...new Set(state.data.fixedPoolPokemon.map(x=>x.pair_id))].sort();
  fillSelect(document.getElementById("inspectPair"),pairs,p=>p.toUpperCase());

  document.getElementById("telemetryTrain").onchange=e=>{state.telemetry.train=+e.target.value;renderTelemetry();};
  document.getElementById("telemetryEval").onchange=e=>{state.telemetry.eval=+e.target.value;renderTelemetry();};
  document.getElementById("telemetryOpponent").onchange=e=>{state.telemetry.opponent=e.target.value;renderTelemetry();};

  document.querySelectorAll(".mini-tab").forEach(btn=>{
    btn.onclick=()=>{
      state.moveMode=btn.dataset.moveMode;
      document.querySelectorAll(".mini-tab").forEach(b=>b.classList.toggle("active",b===btn));
      renderMoveExplorer();
    };
  });

  document.getElementById("inspectGen").onchange=e=>{state.inspector.gen=+e.target.value;renderInspector();};
  document.getElementById("inspectTrain").onchange=e=>{state.inspector.train=+e.target.value;renderInspector();};
  document.getElementById("inspectOpponent").onchange=e=>{state.inspector.opponent=e.target.value;renderInspector();};
  document.getElementById("inspectPair").onchange=e=>{state.inspector.pair=e.target.value;renderInspector();};
}

async function init(){
  const response=await fetch("data/battle_lab_data.json");
  state.data=await response.json();

  document.getElementById("seedBadge").textContent=state.data.meta.seeds;
  document.getElementById("battleBadge").textContent=state.data.meta.totalControlledBattles.toLocaleString();
  document.getElementById("sourceNote").textContent=`DATA: ${state.data.meta.source}`;

  bindFilters();
  nav();
  render();
}

init().catch(err=>{
  document.body.innerHTML=`<pre style="padding:24px">Battle Lab failed to load.\n${err}\n\nRun it through a local HTTP server, not file://.</pre>`;
});
