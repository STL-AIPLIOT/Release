# -*- coding: utf-8 -*-
"""DogFightEnv 통합 대시보드 — Training 탭 + Replay(DogFight Log Playback) 탭.

기존 동작을 유지한다: 벤더 패키지(DogFightEnv/tools/dogfight_dashboard)가 있으면
그쪽 main() 을 그대로 호출한다. 그 패키지가 없는 배포본에서는 내장 서버로 대체한다.
(2026-08-04 기준 벤더 드롭에는 DogFightEnv/tools 자체가 없어 원본 래퍼는 ImportError 로
실행되지 않았다. tools/web_log_viewer.py 와 tools/training_dashboard/server.py 도
같은 패키지를 import 하므로 둘 다 실행되지 않는다.)

내장 서버는 표준 라이브러리만 쓴다. 새 의존성이 없다.

    # 학습 지표만
    python tools/dashboard.py --logdir artifacts/logs/stil/<tag> --port 7860

    # 대표 경기 복기만 (Replay 탭)
    python tools/dashboard.py --playback-dir analysis/playback_cases --port 7860

    # 둘 다
    python tools/dashboard.py --logdir artifacts/logs/stil \
        --playback-dir analysis/playback_cases --port 7860

--logdir 아래에 training_log.csv 가 여러 개면 실험별로 나누어 함께 보여준다.
--playback-dir 는 tools/export_playback_cases.py 가 만든 디렉터리다
(manifest.json + case_*/playback.json).

Replay 탭이 보여주는 것
    - 위에서 본 궤적(북/동)과 고도 프로파일, 현재 시점 표시
    - Own ATA / Target AA / 거리 / 고도 / 속도 HUD
    - WEZ 활성 badge (내가 조준 중 / 표적이 나를 조준 중)
    - BFM 모드와 SCISSORS badge (PredictManeuver CSV 를 붙인 케이스에만 있다)
    - ATA/AA 시간 그래프와 이벤트 타임라인 marker
    - 재생 / 일시정지 / 배속 / 시간 이동
    - playback.json / trajectory.csv 내려받기

값의 단위와 부호 규약은 playback.json 의 units / angle_conventions 에 들어 있고,
화면에도 그대로 표시한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ENV_ROOT = Path(__file__).resolve().parents[1]
VENDOR_TOOLS = ENV_ROOT.parent / "tools"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import (  # noqa: E402
    DASHBOARD_COMBOS,
    DASHBOARD_METRICS,
    find_training_logs,
    iter_training_log,
    warn,
)


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _try_vendor() -> bool:
    """벤더 대시보드가 있으면 실행하고 True. 없으면 False."""
    if not (VENDOR_TOOLS / "dogfight_dashboard").exists():
        return False
    sys.path.insert(0, str(VENDOR_TOOLS))
    try:
        from dogfight_dashboard.server import main as vendor_main  # type: ignore
    except ImportError as exc:
        warn(f"벤더 대시보드 import 실패, 내장 서버로 전환한다: {exc}")
        return False
    if not _has_option("--env-root"):
        sys.argv.extend(["--env-root", str(ENV_ROOT)])
    vendor_main()
    return True


# --------------------------------------------------------------------------- 데이터
def moving_average(values: list[float | None], window: int) -> list[float | None]:
    """None 을 건너뛴 이동평균. 창 안에 값이 없으면 None."""
    out: list[float | None] = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def read_run(path: Path, metrics: tuple[str, ...], window: int) -> dict[str, object]:
    """training_log.csv 하나를 읽어 지표별 raw/smooth 계열을 만든다."""
    columns = ("iter",) + metrics
    iters: list[float | None] = []
    series: dict[str, list[float | None]] = {m: [] for m in metrics}
    present: set[str] = set()

    for row in iter_training_log(path, columns):
        iters.append(row.get("iter"))
        for m in metrics:
            if m in row:
                present.add(m)
                series[m].append(row[m])
            else:
                series[m].append(None)

    missing = [m for m in metrics if m not in present]
    if missing:
        warn(f"{path.parent.name}: 지표 없음 {missing} (해당 지표만 건너뛴다)")

    payload: dict[str, object] = {
        "name": path.parent.name,
        "path": str(path),
        "iterations": iters,
        "rows": len(iters),
        "missing": missing,
        "metrics": {},
    }
    for m in metrics:
        if m in missing:
            payload["metrics"][m] = {"available": False}
            continue
        raw = series[m]
        smooth = moving_average(raw, window)
        latest = next((v for v in reversed(raw) if v is not None), None)
        valid = [v for v in raw if v is not None]
        payload["metrics"][m] = {
            "available": True,
            "raw": raw,
            "smooth": smooth,
            "latest": latest,
            "latest_smooth": next((v for v in reversed(smooth) if v is not None), None),
            "count": len(valid),
            "min": min(valid) if valid else None,
            "max": max(valid) if valid else None,
            "mean": (sum(valid) / len(valid)) if valid else None,
            "trend": (valid[-1] - valid[0]) if len(valid) >= 2 else None,
        }
    return payload


def collect(logdir: Path, metrics: tuple[str, ...], window: int) -> dict[str, object]:
    """logdir 아래 모든 실험을 읽는다."""
    logs = find_training_logs(logdir)
    hint = ""
    if not logs:
        # 조용히 빈 화면을 띄우지 않는다. 어디에 있는지 찾아서 알려 준다.
        hint = suggest_logdir(logdir)
        warn(f"training_log.csv 를 찾지 못했다: {logdir}")
        if hint:
            warn(hint)
    return {
        "logdir": str(logdir),
        "window": window,
        "hint": hint,
        "combos": [list(c) for c in DASHBOARD_COMBOS],
        "runs": [read_run(p, metrics, window) for p in logs],
    }


def suggest_logdir(missing: Path) -> str:
    """training_log.csv 가 어디에 있는지 찾아 안내 문구를 만든다.

    artifacts/dashboard 에는 metrics.jsonl 만 있고 training_log.csv 는
    artifacts/logs/<name>/<tag>/ 에 있다. 이 둘을 헷갈리기 쉬워 안내를 붙인다.
    """
    for base in (ENV_ROOT / "artifacts" / "logs", Path("artifacts") / "logs"):
        found = sorted(base.rglob("training_log.csv")) if base.exists() else []
        if found:
            dirs = sorted({str(f.parent.parent) for f in found})
            return ("training_log.csv 는 artifacts/logs/<name>/<tag>/ 에 있다. "
                    f"예: --logdir {dirs[0]}")
    return ""


# --------------------------------------------------------------------------- 복기
def load_manifest(playback_dir: Path | None) -> dict[str, object]:
    """playback 케이스 목록을 읽는다. 없으면 사유를 담아 돌려준다."""
    if playback_dir is None:
        return {"available": False,
                "reason": "--playback-dir 을 주지 않았다.", "cases": []}
    path = playback_dir / "manifest.json"
    if not path.exists():
        warn(f"manifest.json 이 없다: {path}")
        return {"available": False,
                "reason": f"manifest.json 이 없다: {path}. "
                          "tools/export_playback_cases.py 를 먼저 실행하라.",
                "cases": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warn(f"{path} 파싱 실패: {exc}")
        return {"available": False, "reason": f"manifest.json 파싱 실패: {exc}",
                "cases": []}
    data["available"] = True
    data["playback_dir"] = str(playback_dir)
    return data


def load_case(playback_dir: Path | None, case_id: str) -> dict[str, object] | None:
    """케이스 하나의 playback.json 을 읽는다.

    case_id 는 manifest 에 있는 것만 허용한다. 임의 경로 접근을 막는다.
    """
    if playback_dir is None:
        return None
    manifest = load_manifest(playback_dir)
    known = {str(c.get("case_id")) for c in manifest.get("cases", [])}
    if case_id not in known:
        warn(f"manifest 에 없는 case_id: {case_id}")
        return None
    path = playback_dir / case_id / "playback.json"
    if not path.exists():
        warn(f"playback.json 이 없다: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        warn(f"{path} 파싱 실패: {exc}")
        return None


# --------------------------------------------------------------------------- 서버
PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>DogFightEnv 지표 대시보드</title><style>
:root{--bg:#15171a;--panel:#1e2126;--line:#2f343b;--ink:#e6e6e3;--dim:#98a0a8;--acc:#e0a942;--raw:#5a6270}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI","Malgun Gothic",system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:15px;margin:0;letter-spacing:.02em}
.meta{color:var(--dim);font:12px ui-monospace,Consolas,monospace}
.run{margin:20px;border:1px solid var(--line);border-radius:3px;background:var(--panel)}
.run>h2{margin:0;padding:12px 16px;font-size:14px;border-bottom:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1px;background:var(--line)}
.card{background:var(--panel);padding:14px 16px}
.k{font:11px ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.v{font:20px ui-monospace,monospace;font-variant-numeric:tabular-nums;margin:4px 0}
.sub{font:11px ui-monospace,monospace;color:var(--dim)}
.na{color:#c9756c}
.combo{grid-column:1/-1}
.combo canvas{height:150px}
.sw{display:inline-block;width:9px;height:9px;margin-right:4px;vertical-align:middle}
canvas{width:100%;height:90px;display:block;margin-top:8px}
nav{display:flex;gap:2px;margin-left:auto}
nav button{background:var(--panel);color:var(--dim);border:1px solid var(--line);
  padding:5px 14px;font:12px ui-monospace,monospace;cursor:pointer;border-radius:2px}
nav button.on{color:var(--ink);border-color:var(--acc)}
.hide{display:none}
.pb{margin:20px;display:grid;grid-template-columns:230px 1fr;gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:3px}
.pb-list{background:var(--panel);padding:10px;max-height:78vh;overflow:auto}
.pb-list button{display:block;width:100%;text-align:left;background:transparent;
  color:var(--ink);border:1px solid var(--line);border-radius:2px;padding:8px;
  margin-bottom:6px;cursor:pointer;font:12px ui-monospace,monospace}
.pb-list button.on{border-color:var(--acc)}
.pb-main{background:var(--panel);padding:14px 16px;min-width:0}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}
.ctl button,.ctl select{background:#262a30;color:var(--ink);border:1px solid var(--line);
  padding:4px 10px;font:12px ui-monospace,monospace;cursor:pointer;border-radius:2px}
.ctl input[type=range]{flex:1;min-width:200px}
.hud{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:1px;
  background:var(--line);border:1px solid var(--line);margin:10px 0}
.hud>div{background:var(--panel);padding:8px 10px}
.badge{display:inline-block;padding:2px 8px;border-radius:2px;font:11px ui-monospace,monospace;
  border:1px solid var(--line);color:var(--dim)}
.badge.on{background:#5c2b26;border-color:#c9756c;color:#ffd9d3}
.badge.ok{background:#2b4a2e;border-color:#6fa873;color:#d7f0d9}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.plots canvas{height:220px}
#pbTimeline{height:46px}
#pbAngles{height:150px}
.legend{font:11px ui-monospace,monospace;color:var(--dim);margin-top:4px}
.note{font:11px/1.6 ui-monospace,monospace;color:var(--dim);border-left:2px solid var(--line);
  padding-left:10px;margin:10px 0}
a.dl{color:var(--acc);font:11px ui-monospace,monospace;margin-right:12px}
</style></head><body>
<header><h1>DogFightEnv 대시보드</h1>
<span class="meta" id="meta">불러오는 중…</span>
<span class="meta na" id="hint"></span>
<nav><button id="tabTraining" class="on">Training</button><button id="tabReplay">Replay</button></nav>
</header>
<div id="root"></div>
<div id="replay" class="hide"></div>
<script>
const METRICS = __METRICS__, REFRESH = __REFRESH__, PLAYBACK_DIR = __PLAYBACK__;
function draw(cv, raw, smooth){
  const dpr = window.devicePixelRatio||1, w=cv.clientWidth, h=90;
  cv.width=w*dpr; cv.height=h*dpr; const c=cv.getContext("2d"); c.scale(dpr,dpr); c.clearRect(0,0,w,h);
  const all=[...raw,...smooth].filter(v=>v!==null);
  if(all.length<2) return;
  const lo=Math.min(...all), hi=Math.max(...all), sp=(hi-lo)||1;
  const px=(i,n)=>n<2?0:i/(n-1)*(w-2)+1, py=v=>h-4-((v-lo)/sp)*(h-10);
  const line=(arr,col,wd)=>{c.beginPath();c.strokeStyle=col;c.lineWidth=wd;let st=false;
    arr.forEach((v,i)=>{if(v===null){st=false;return;}const x=px(i,arr.length),y=py(v);
      if(!st){c.moveTo(x,y);st=true;}else c.lineTo(x,y);});c.stroke();};
  line(raw,"#5a6270",1); line(smooth,"#e0a942",1.8);
}
function fmt(v,d=4){return v===null||v===undefined?"N/A":Number(v).toFixed(d);}
let trainingMeta="";
async function tick(){
  if(activeTab!=="training") return;
  let d; try{ d=await (await fetch("/data")).json(); }catch(e){ document.getElementById("meta").textContent="데이터 로드 실패: "+e; return; }
  trainingMeta=`${d.logdir} · 실험 ${d.runs.length}개 · 창 ${d.window} · ${new Date().toLocaleTimeString()}`;
  document.getElementById("meta").textContent=trainingMeta;
  const root=document.getElementById("root"); root.innerHTML="";
  for(const run of d.runs){
    const sec=document.createElement("section"); sec.className="run";
    sec.innerHTML=`<h2>${run.name} <span class="sub">· ${run.rows} iteration</span></h2>`;
    const g=document.createElement("div"); g.className="grid";
    for(const m of METRICS){
      const s=run.metrics[m], card=document.createElement("div"); card.className="card";
      if(!s||!s.available){ card.innerHTML=`<div class="k">${m}</div><div class="v na">N/A</div><div class="sub">로그에 없음</div>`; }
      else{ card.innerHTML=`<div class="k">${m}</div><div class="v">${fmt(s.latest)}</div>
        <div class="sub">이동평균 ${fmt(s.latest_smooth)} · 표본 ${s.count} · 추세 ${fmt(s.trend,2)}</div>
        <div class="sub">min ${fmt(s.min,2)} · max ${fmt(s.max,2)} · mean ${fmt(s.mean,2)}</div>`;
        const cv=document.createElement("canvas"); card.appendChild(cv);
        setTimeout(()=>draw(cv,s.raw,s.smooth),0); }
      g.appendChild(card);
    }
    // 보상 성분 겹침 패널 (pursuit vs position + 차이선)
    for(const [a,b,label] of (d.combos||[])) addComboCard(g, run, a, b, label);
    sec.appendChild(g); root.appendChild(sec);
  }
  const hint=document.getElementById("hint");
  if(hint) hint.textContent = d.runs.length ? "" : (d.hint||"");
}

// pursuit 과 position 을 한 축에 겹쳐 그리고, 둘 다 있으면 차이선을 덧그린다.
// 차이선이 0 에 붙으면 두 성분이 서로 상쇄되고 있다는 뜻이다.
function addComboCard(grid, run, keyA, keyB, label){
  const A=run.metrics[keyA], B=run.metrics[keyB];
  const card=document.createElement("div"); card.className="card combo";
  const na=[];
  if(!A||!A.available) na.push(keyA);
  if(!B||!B.available) na.push(keyB);
  card.innerHTML=`<div class="k">${label}</div>
    <div class="sub"><span class="sw" style="background:#e0a942"></span>${keyA}
      <span class="sw" style="background:#5b8dd6;margin-left:10px"></span>${keyB}
      <span class="sw" style="background:#6fa873;margin-left:10px"></span>차이(${keyA} − ${keyB})</div>
    ${na.length?`<div class="sub na">로그에 없는 컬럼: ${na.join(", ")}
      — train_rllib.py 의 _CSV_FIELDS 가 하드코딩이라 해당 성분은 CSV 로 나오지 않는다.
      있는 계열만 그린다.</div>`:""}`;
  const cv=document.createElement("canvas"); card.appendChild(cv);
  grid.appendChild(card);
  setTimeout(()=>drawCombo(cv, A, B),0);
}

function drawCombo(cv, A, B){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth, h=150;
  cv.width=w*dpr; cv.height=h*dpr; const c=cv.getContext("2d");
  c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,w,h);
  const sa=(A&&A.available)?A.smooth:null, sb=(B&&B.available)?B.smooth:null;
  let diff=null;
  if(sa&&sb){ diff=sa.map((v,i)=>(v===null||sb[i]===null)?null:v-sb[i]); }
  const all=[...(sa||[]),...(sb||[]),...(diff||[])].filter(v=>v!==null);
  if(all.length<2){
    c.fillStyle="#98a0a8"; c.font="12px ui-monospace,monospace";
    c.fillText("표시할 값이 없다", 8, 20); return;
  }
  const lo=Math.min(...all,0), hi=Math.max(...all,0), sp=(hi-lo)||1;
  const px=(i,n)=>n<2?0:i/(n-1)*(w-2)+1, py=v=>h-6-((v-lo)/sp)*(h-14);
  // 0 기준선: 차이선이 여기 붙으면 상쇄 구간이다.
  c.strokeStyle="#3a4048"; c.setLineDash([3,3]);
  c.beginPath(); c.moveTo(1,py(0)); c.lineTo(w-1,py(0)); c.stroke(); c.setLineDash([]);
  const line=(arr,col,wd)=>{ if(!arr) return;
    c.beginPath(); c.strokeStyle=col; c.lineWidth=wd; let st=false;
    arr.forEach((v,i)=>{ if(v===null){st=false;return;}
      const x=px(i,arr.length), y=py(v);
      if(!st){c.moveTo(x,y);st=true;} else c.lineTo(x,y);});
    c.stroke(); c.lineWidth=1; };
  line(sa,"#e0a942",1.6); line(sb,"#5b8dd6",1.6); line(diff,"#6fa873",1.2);
}

// ---------------------------------------------------------------- Replay 탭
let activeTab = "training";
let PB = null;          // 현재 케이스 playback.json
let pbIndex = 0;        // 현재 프레임
let pbPlaying = false;
let pbSpeed = 1;
let pbTimer = null;

function setTab(name){
  activeTab = name;
  document.getElementById("tabTraining").className = name==="training"?"on":"";
  document.getElementById("tabReplay").className = name==="replay"?"on":"";
  document.getElementById("root").className = name==="training"?"":"hide";
  document.getElementById("replay").className = name==="replay"?"":"hide";
  if(name==="training"){ document.getElementById("meta").textContent=trainingMeta; tick(); }
  else { loadManifest(); }
}

async function loadManifest(){
  const box=document.getElementById("replay");
  let m; try{ m=await (await fetch("/playback/manifest")).json(); }
  catch(e){ box.innerHTML=`<div class="note">manifest 로드 실패: ${e}</div>`; return; }
  if(!m.available){
    document.getElementById("meta").textContent="복기 데이터 없음";
    box.innerHTML=`<div class="note">${m.reason}<br><br>
      다음을 먼저 실행하라:<br>
      <code>python tools/export_playback_cases.py --logdir &lt;로그루트&gt; --output analysis/playback_cases</code><br>
      그리고 대시보드를 <code>--playback-dir analysis/playback_cases</code> 로 실행하라.</div>`;
    return;
  }
  document.getElementById("meta").textContent=`${m.playback_dir} · 케이스 ${m.cases.length}개`;
  const un=(m.unavailable_reasons||[]).map(r=>`<div>· ${r}</div>`).join("");
  box.innerHTML=`<div class="pb"><div class="pb-list" id="pbList"></div>
    <div class="pb-main" id="pbMain"><div class="note">왼쪽에서 케이스를 고르라.</div>
    ${un?`<div class="note"><b>담지 못한 값</b>${un}</div>`:""}</div></div>`;
  const list=document.getElementById("pbList");
  m.cases.forEach((c,i)=>{
    const b=document.createElement("button");
    b.innerHTML=`<b>${c.case_id}</b> · ${c.result}<br><span class="sub">${c.case_type}</span>
      <br><span class="sub">${c.episode_id}</span>`;
    b.onclick=()=>{ [...list.children].forEach(x=>x.className=""); b.className="on"; openCase(c.case_id); };
    list.appendChild(b);
    if(i===0) setTimeout(()=>b.click(),0);
  });
}

async function openCase(id){
  stopPlay();
  let d; try{ d=await (await fetch("/playback/case?id="+encodeURIComponent(id))).json(); }
  catch(e){ document.getElementById("pbMain").innerHTML=`<div class="note">로드 실패: ${e}</div>`; return; }
  PB=d; pbIndex=0;
  const conv=Object.entries(d.angle_conventions||{}).map(([k,v])=>`<div>· <b>${k}</b> — ${v}</div>`).join("");
  const hasBfm = (d.frames||[]).some(f=>f.bfm_mode!==undefined);
  document.getElementById("pbMain").innerHTML=`
    <h2 style="margin:0 0 4px;font-size:14px">${d.case_id} · ${d.case_type}</h2>
    <div class="sub">${d.episode_id} · 결과 <b>${d.result}</b> · 종료 <code>${d.end_condition_raw}</code>
      · 길이 ${fmt(d.real_duration_sec,2)}s(실제) / ${fmt(d.duration_sec,2)}s(Time 컬럼)
      · 샘플 ${d.sample_count}(stride ${d.sample_stride})</div>
    <div class="note">선정 이유: ${d.reason_selected}</div>
    <div class="ctl">
      <button id="pbPlay">▶ 재생</button>
      <button id="pbStep-">◀</button><button id="pbStep+">▶</button>
      <select id="pbSpeed">
        <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
        <option value="1" selected>1×</option><option value="2">2×</option>
        <option value="4">4×</option><option value="8">8×</option></select>
      <input type="range" id="pbSeek" min="0" max="${(d.frames||[]).length-1}" value="0">
      <span class="sub" id="pbClock">0.000 s</span>
    </div>
    <div class="hud" id="pbHud"></div>
    <canvas id="pbTimeline"></canvas>
    <div class="legend">타임라인 marker — 빨강 WEZ(표적→나) · 초록 WEZ(나→표적) · 노랑 BFM/SCISSORS · 흰색 피격/종료</div>
    <div class="plots">
      <div><canvas id="pbMap"></canvas><div class="legend">위에서 본 궤적 (파랑 아군 · 빨강 표적 · 굵은 구간 = 표적이 나를 WEZ 안에 둔 구간)</div></div>
      <div><canvas id="pbAlt"></canvas><div class="legend">고도 프로파일 (m)</div></div>
    </div>
    <canvas id="pbAngles"></canvas>
    <div class="legend">노랑 |Own ATA| · 하늘 |Target AA| · 회색 거리(정규화) — 전부 파생값</div>
    <div class="note"><b>값의 출처</b> — 원본은 Tacview CSV 의 위치/자세/체력뿐이다.
      ATA·AA·WEZ·속도·에너지는 다시 계산한 <code>derived_*</code> 값이다.
      ${hasBfm?"BFM/SCISSORS 는 PredictManeuver CSV 에서 왔다.":"BFM/SCISSORS 는 이 케이스에 없다(PredictManeuver CSV 미첨부)."}
      <br>단위: 각 ${d.units.angle} · 거리 ${d.units.distance} · 속도 ${d.units.speed} · 시간 ${d.units.time}
      <br><b>시간축</b>: ${d.time_base_note||"(정보 없음)"}
      <br>WEZ 게이트: ${d.wez_config.min_range_m}~${d.wez_config.max_range_m} m, ${d.wez_config.note}
      ${conv}</div>
    <div><a class="dl" href="/playback/file?id=${encodeURIComponent(id)}&name=playback.json" download>playback.json 내려받기</a>
      <a class="dl" href="/playback/file?id=${encodeURIComponent(id)}&name=trajectory.csv" download>trajectory.csv 내려받기</a></div>`;

  document.getElementById("pbPlay").onclick=togglePlay;
  document.getElementById("pbStep-").onclick=()=>seek(pbIndex-1);
  document.getElementById("pbStep+").onclick=()=>seek(pbIndex+1);
  document.getElementById("pbSpeed").onchange=e=>{pbSpeed=Number(e.target.value); if(pbPlaying){stopPlay();startPlay();}};
  document.getElementById("pbSeek").oninput=e=>seek(Number(e.target.value));
  // 시간축 차트는 hover 로 상세를 보여주고 click 으로 그 시점으로 이동한다.
  ["pbTimeline","pbAlt","pbAngles"].forEach(id=>{
    const cv=document.getElementById(id);
    cv.onmousemove=ev=>hoverTime(ev,cv,id==="pbTimeline");
    cv.onclick=ev=>{ stopPlay(); seek(indexFromX(ev,cv,id==="pbTimeline")); };
    cv.style.cursor="crosshair";
  });
  // 핵심 시각이 있으면 거기로 이동해 둔다.
  if(d.key_timestamp_sec!==null && d.key_timestamp_sec!==undefined){
    seek(nearestFrame(d.key_timestamp_sec));
  } else seek(0);
}

function indexFromX(ev,cv,byTime){
  const F=PB.frames, r=cv.getBoundingClientRect();
  const rel=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
  if(!byTime) return Math.round(rel*(F.length-1));
  const t0=F[0].time_sec, t1=F[F.length-1].time_sec;
  return nearestFrame(t0+rel*((t1-t0)||0));
}
function hoverTime(ev,cv,byTime){
  if(!PB) return;
  const f=PB.frames[indexFromX(ev,cv,byTime)];
  const wez=f.derived_target_in_wez?"표적→나 WEZ 안":(f.derived_own_in_wez?"나→표적 WEZ 안":"WEZ 밖");
  cv.title=`t=${fmt(f.time_sec,3)}s\n`
    +`Own ATA ${fmt(f.derived_own_ata_deg,2)}° / Target AA ${fmt(f.derived_target_aa_deg,2)}°\n`
    +`거리 ${fmt(f.derived_distance_m,1)} m / 내 고도 ${fmt(f.own_alt_m,1)} m\n`
    +`${wez}${f.bfm_mode!==undefined?" / BFM "+f.bfm_mode:""}\n`
    +`클릭하면 이 시점으로 이동한다.`;
}

function nearestFrame(t){
  let best=0,bd=Infinity;
  PB.frames.forEach((f,i)=>{const dd=Math.abs(f.time_sec-t); if(dd<bd){bd=dd;best=i;}});
  return best;
}
function seek(i){
  if(!PB) return;
  pbIndex=Math.max(0,Math.min(PB.frames.length-1,i));
  document.getElementById("pbSeek").value=pbIndex;
  renderAll();
}
function startPlay(){
  // 프레임 간 실제 시간 간격을 배속으로 나눠 재생한다.
  pbPlaying=true; document.getElementById("pbPlay").textContent="⏸ 일시정지";
  const step=()=>{
    if(!pbPlaying||!PB) return;
    if(pbIndex>=PB.frames.length-1){ stopPlay(); return; }
    // 실제 경과 시각이 있으면 그것으로 재생 속도를 맞춘다(Time 컬럼은 step_ratio 배 느리다).
    const key = PB.frames[0].derived_real_time_sec!==undefined ? "derived_real_time_sec" : "time_sec";
    const dt=(PB.frames[pbIndex+1][key]-PB.frames[pbIndex][key])*1000/pbSpeed;
    seek(pbIndex+1);
    pbTimer=setTimeout(step, Math.max(8, isFinite(dt)?dt:33));
  };
  step();
}
function stopPlay(){
  pbPlaying=false; if(pbTimer) clearTimeout(pbTimer); pbTimer=null;
  const b=document.getElementById("pbPlay"); if(b) b.textContent="▶ 재생";
}
function togglePlay(){ pbPlaying?stopPlay():startPlay(); }

function setupCanvas(cv,h){
  const dpr=window.devicePixelRatio||1, w=cv.clientWidth;
  cv.width=w*dpr; cv.height=h*dpr; const c=cv.getContext("2d");
  c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,w,h); return [c,w,h];
}

function renderAll(){
  if(!PB) return;
  const f=PB.frames[pbIndex];
  // Time 컬럼은 실제 경과가 아니다. 실제 시각을 먼저 보이고 원본을 괄호로 덧붙인다.
  const real=f.derived_real_time_sec;
  document.getElementById("pbClock").textContent =
    (real===undefined||real===null ? fmt(f.time_sec,3)+" s"
     : `${fmt(real,2)} s (Time ${fmt(f.time_sec,3)})`);
  renderHud(f); renderTimeline(); renderMap(); renderAlt(); renderAngles();
}

function badge(label,on,okColor){
  return `<span class="badge ${on?(okColor?"ok":"on"):""}">${label}: ${on===null||on===undefined?"N/A":(on?"YES":"no")}</span>`;
}
function renderHud(f){
  const rows=[
    ["Own ATA (deg)", fmt(f.derived_own_ata_deg,2)],
    ["Target AA (deg)", fmt(f.derived_target_aa_deg,2)],
    ["거리 (m)", fmt(f.derived_distance_m,1)],
    ["내 고도 (m)", fmt(f.own_alt_m,1)],
    ["표적 고도 (m)", fmt(f.target_alt_m,1)],
    ["내 속도 (m/s)", fmt(f.derived_own_speed_ms,1)],
    ["내 체력", fmt(f.own_health,4)],
    ["표적 체력", fmt(f.target_health,4)],
  ];
  let html=rows.map(([k,v])=>`<div><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  html+=`<div><div class="k">WEZ</div><div style="margin-top:6px">
    ${badge("나→표적", f.derived_own_in_wez, true)}<br>
    ${badge("표적→나", f.derived_target_in_wez, false)}</div></div>`;
  const bfm=f.bfm_mode;
  html+=`<div><div class="k">BFM</div><div style="margin-top:6px">
    <span class="badge ${bfm==="SCISSORS"?"on":""}">${bfm===undefined?"N/A (로그 없음)":bfm}</span></div></div>`;
  if(f.avg_delta_deg!==undefined){
    html+=`<div><div class="k">avgDelta (deg)</div><div class="v">${fmt(f.avg_delta_deg,2)}</div></div>`;
  }
  if(f.derived_ata_sign_degenerate){
    html+=`<div><div class="k">경고</div><div class="sub na">ATA 부호 붕괴 구간<br>(플랫폼 결함 1)</div></div>`;
  }
  document.getElementById("pbHud").innerHTML=html;
}

const EVENT_COLOR={
  WEZ_ENTER_TARGET:"#c9756c", WEZ_ENTER_OWN:"#6fa873",
  BFM_TRANSITION:"#e0a942", SCISSORS_ENTER:"#e0a942",
  OWN_DAMAGE:"#e6e6e3", EPISODE_END:"#e6e6e3"};
function renderTimeline(){
  const [c,w,h]=setupCanvas(document.getElementById("pbTimeline"),46);
  const t0=PB.frames[0].time_sec, t1=PB.frames[PB.frames.length-1].time_sec;
  const sp=(t1-t0)||1, x=t=>((t-t0)/sp)*(w-2)+1;
  c.strokeStyle="#2f343b"; c.beginPath(); c.moveTo(1,h-10); c.lineTo(w-1,h-10); c.stroke();
  (PB.events||[]).forEach(e=>{
    if(e.time_sec===null||e.time_sec===undefined) return;
    const col=EVENT_COLOR[e.type]||"#98a0a8";
    if(e.end_sec!==null&&e.end_sec!==undefined){
      c.fillStyle=col+"55"; c.fillRect(x(e.time_sec),8,Math.max(2,x(e.end_sec)-x(e.time_sec)),h-20);
    }
    c.strokeStyle=col; c.beginPath(); c.moveTo(x(e.time_sec),6); c.lineTo(x(e.time_sec),h-10); c.stroke();
  });
  const cx=x(PB.frames[pbIndex].time_sec);
  c.strokeStyle="#ffffff"; c.lineWidth=1.5;
  c.beginPath(); c.moveTo(cx,2); c.lineTo(cx,h-6); c.stroke(); c.lineWidth=1;
}

function bounds(vals){const v=vals.filter(x=>x!==null&&x!==undefined&&isFinite(x));
  return v.length?[Math.min(...v),Math.max(...v)]:[0,1];}

function renderMap(){
  const [c,w,h]=setupCanvas(document.getElementById("pbMap"),220);
  const F=PB.frames;
  const lats=F.map(f=>f.own_lat).concat(F.map(f=>f.target_lat));
  const lons=F.map(f=>f.own_lon).concat(F.map(f=>f.target_lon));
  const [la0,la1]=bounds(lats), [lo0,lo1]=bounds(lons);
  const pad=12, sx=(w-2*pad)/((lo1-lo0)||1e-9), sy=(h-2*pad)/((la1-la0)||1e-9);
  const s=Math.min(sx,sy);
  const X=lo=>pad+(lo-lo0)*s, Y=la=>h-pad-(la-la0)*s;
  const path=(latK,lonK,col)=>{
    c.beginPath(); c.strokeStyle=col; c.lineWidth=1.2; let st=false;
    F.forEach(f=>{const la=f[latK],lo=f[lonK];
      if(la===null||lo===null){st=false;return;}
      const x=X(lo),y=Y(la); if(!st){c.moveTo(x,y);st=true;}else c.lineTo(x,y);});
    c.stroke();
  };
  path("own_lat","own_lon","#5b8dd6"); path("target_lat","target_lon","#c9756c");
  // 표적이 나를 WEZ 안에 둔 구간은 굵게 덧그린다.
  c.beginPath(); c.strokeStyle="#ff6b5e"; c.lineWidth=3; let st=false;
  F.forEach(f=>{ if(!f.derived_target_in_wez||f.own_lat===null){st=false;return;}
    const x=X(f.own_lon),y=Y(f.own_lat); if(!st){c.moveTo(x,y);st=true;}else c.lineTo(x,y);});
  c.stroke(); c.lineWidth=1;
  const f=F[pbIndex];
  const dot=(la,lo,col)=>{ if(la===null||lo===null) return;
    c.fillStyle=col; c.beginPath(); c.arc(X(lo),Y(la),4,0,7); c.fill(); };
  dot(f.own_lat,f.own_lon,"#9dc0f0"); dot(f.target_lat,f.target_lon,"#f0a79d");
  if(f.own_lat!==null&&f.target_lat!==null){
    c.strokeStyle="#98a0a8"; c.setLineDash([3,3]); c.beginPath();
    c.moveTo(X(f.own_lon),Y(f.own_lat)); c.lineTo(X(f.target_lon),Y(f.target_lat));
    c.stroke(); c.setLineDash([]);
  }
}

function renderAlt(){
  const [c,w,h]=setupCanvas(document.getElementById("pbAlt"),220);
  const F=PB.frames;
  const [a0,a1]=bounds(F.map(f=>f.own_alt_m).concat(F.map(f=>f.target_alt_m)));
  const sp=(a1-a0)||1, X=i=>(i/((F.length-1)||1))*(w-2)+1, Y=v=>h-8-((v-a0)/sp)*(h-16);
  const line=(k,col)=>{c.beginPath();c.strokeStyle=col;let st=false;
    F.forEach((f,i)=>{const v=f[k]; if(v===null){st=false;return;}
      const x=X(i),y=Y(v); if(!st){c.moveTo(x,y);st=true;}else c.lineTo(x,y);});c.stroke();};
  line("own_alt_m","#5b8dd6"); line("target_alt_m","#c9756c");
  c.strokeStyle="#ffffff"; c.beginPath(); c.moveTo(X(pbIndex),2); c.lineTo(X(pbIndex),h-2); c.stroke();
}

function renderAngles(){
  const [c,w,h]=setupCanvas(document.getElementById("pbAngles"),150);
  const F=PB.frames;
  const X=i=>(i/((F.length-1)||1))*(w-2)+1, Y=v=>h-8-(v/180)*(h-16);
  // 배경: WEZ(표적→나) 구간
  c.fillStyle="#5c2b2655";
  F.forEach((f,i)=>{ if(f.derived_target_in_wez) c.fillRect(X(i)-1,2,3,h-4); });
  const line=(get,col,wd)=>{c.beginPath();c.strokeStyle=col;c.lineWidth=wd||1.2;let st=false;
    F.forEach((f,i)=>{const v=get(f); if(v===null||v===undefined||!isFinite(v)){st=false;return;}
      const x=X(i),y=Y(v); if(!st){c.moveTo(x,y);st=true;}else c.lineTo(x,y);});c.stroke();c.lineWidth=1;};
  line(f=>f.derived_own_ata_deg===null?null:Math.abs(f.derived_own_ata_deg),"#e0a942",1.5);
  line(f=>f.derived_target_aa_deg===null?null:Math.abs(f.derived_target_aa_deg),"#5b8dd6",1.2);
  const [d0,d1]=bounds(F.map(f=>f.derived_distance_m));
  line(f=>f.derived_distance_m===null?null:((f.derived_distance_m-d0)/((d1-d0)||1))*180,"#5a6270",1);
  // 1도(WEZ 반각) 기준선
  c.strokeStyle="#6fa873"; c.setLineDash([4,4]);
  c.beginPath(); c.moveTo(1,Y(PB.wez_config.angle_deg/2)); c.lineTo(w-1,Y(PB.wez_config.angle_deg/2));
  c.stroke(); c.setLineDash([]);
  c.strokeStyle="#ffffff"; c.beginPath(); c.moveTo(X(pbIndex),2); c.lineTo(X(pbIndex),h-2); c.stroke();
}

window.addEventListener("resize", ()=>{ if(activeTab==="replay") renderAll(); });

document.getElementById("tabTraining").onclick=()=>setTab("training");
document.getElementById("tabReplay").onclick=()=>setTab("replay");
if(PLAYBACK_DIR && !__HAS_LOGDIR__) setTab("replay"); else tick();
setInterval(tick, REFRESH*1000);
</script></body></html>"""


# 케이스 디렉터리에서 내려받기를 허용하는 파일. 임의 경로 접근을 막는다.
DOWNLOADABLE = {
    "playback.json": "application/json; charset=utf-8",
    "trajectory.csv": "text/csv; charset=utf-8",
    "source_summary.json": "application/json; charset=utf-8",
    "case_report.md": "text/markdown; charset=utf-8",
}


def _query(path: str) -> dict[str, str]:
    """?a=1&b=2 를 dict 로. 값이 없는 키는 빈 문자열."""
    parsed = urlparse(path)
    out: dict[str, str] = {}
    for part in parsed.query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        out[unquote(key)] = unquote(value.replace("+", " "))
    return out


def make_handler(logdir: Path | None, metrics: tuple[str, ...], window: int,
                 refresh: int, playback_dir: Path | None):
    """요청마다 로그를 다시 읽는다. 파일이 갱신되면 재시작 없이 반영된다."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: object, status: int = 200) -> None:
            self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path

            if route == "/data":
                if logdir is None:
                    self._json({"logdir": "", "window": window, "runs": [],
                                "reason": "--logdir 을 주지 않았다."})
                    return
                self._json(collect(logdir, metrics, window))
                return

            if route == "/playback/manifest":
                self._json(load_manifest(playback_dir))
                return

            if route == "/playback/case":
                case = load_case(playback_dir, _query(self.path).get("id", ""))
                if case is None:
                    self._json({"error": "케이스를 찾지 못했다"}, 404)
                    return
                self._json(case)
                return

            if route == "/playback/file":
                q = _query(self.path)
                name = q.get("name", "")
                if name not in DOWNLOADABLE or playback_dir is None:
                    self._send(b"not found", "text/plain; charset=utf-8", 404)
                    return
                # case_id 는 manifest 에 있는 것만 허용한다(경로 탈출 방지).
                known = {str(c.get("case_id"))
                         for c in load_manifest(playback_dir).get("cases", [])}
                case_id = q.get("id", "")
                target = playback_dir / case_id / name
                if case_id not in known or not target.exists():
                    self._send(b"not found", "text/plain; charset=utf-8", 404)
                    return
                self._send(target.read_bytes(), DOWNLOADABLE[name])
                return

            page = (PAGE
                    .replace("__METRICS__", json.dumps(list(metrics)))
                    .replace("__REFRESH__", str(refresh))
                    .replace("__PLAYBACK__",
                             json.dumps(str(playback_dir) if playback_dir else ""))
                    .replace("__HAS_LOGDIR__", "true" if logdir is not None else "false"))
            self._send(page.encode("utf-8"), "text/html; charset=utf-8")

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 지표 + 경기 복기 대시보드")
    # --training-logdir 은 벤더 래퍼(tools/training_dashboard/server.py)의 이름이다.
    # 예전 문서의 명령을 그대로 붙여넣어도 조용히 무시되지 않도록 별칭으로 받는다.
    # 벤더 쪽은 artifacts/dashboard(metrics.jsonl)를 가리켰지만 내장 서버는
    # training_log.csv 를 읽는다. 못 찾으면 collect() 가 올바른 경로를 안내한다.
    ap.add_argument("--logdir", "--training-logdir", dest="logdir", type=Path,
                    help="training_log.csv 가 있는 디렉터리 "
                         "(예: artifacts/logs/stil)")
    ap.add_argument("--playback-dir", type=Path,
                    help="export_playback_cases.py 가 만든 디렉터리 (Replay 탭)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="외부 접속 허용 시 0.0.0.0")
    ap.add_argument("--window", type=int, default=20, help="이동평균 창 크기")
    ap.add_argument("--refresh-sec", type=int, default=5, help="브라우저 갱신 주기")
    ap.add_argument("--metrics", nargs="*", default=list(DASHBOARD_METRICS))
    args, _unknown = ap.parse_known_args()

    if _try_vendor():
        return 0

    if args.logdir is None and args.playback_dir is None:
        ap.error("--logdir 또는 --playback-dir 중 하나는 필요하다 "
                 "(벤더 대시보드가 없어 내장 서버로 실행한다)")

    metrics = tuple(args.metrics)
    if args.logdir is not None:
        snapshot = collect(args.logdir, metrics, args.window)
        print(f"logdir : {args.logdir}")
        print(f"실험    : {len(snapshot['runs'])}개")
        for run in snapshot["runs"]:
            miss = f"  없는 지표: {run['missing']}" if run["missing"] else ""
            print(f"  - {run['name']} ({run['rows']} iteration){miss}")
    if args.playback_dir is not None:
        manifest = load_manifest(args.playback_dir)
        if manifest.get("available"):
            print(f"복기    : 케이스 {len(manifest['cases'])}개 ({args.playback_dir})")
            for c in manifest["cases"]:
                print(f"  - {c['case_id']}  {c['case_type']}  {c['episode_id']}  "
                      f"{c['result']}")
        else:
            print(f"복기    : 없음 — {manifest.get('reason')}")

    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/"
    print(f"\n{url}  (Ctrl+C 로 종료, {args.refresh_sec}초마다 자동 갱신)")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.logdir, metrics, args.window, args.refresh_sec,
                     args.playback_dir),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
