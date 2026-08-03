# -*- coding: utf-8 -*-
"""DogFightEnv 학습 지표 대시보드.

기존 동작을 유지한다: 벤더 패키지(DogFightEnv/tools/dogfight_dashboard)가 있으면
그쪽 main() 을 그대로 호출한다. 그 패키지가 없는 배포본에서는 내장 서버로 대체한다.
(2026-08-04 기준 벤더 드롭에는 DogFightEnv/tools 자체가 없어 원본 래퍼는 ImportError 로
실행되지 않았다.)

내장 서버는 표준 라이브러리만 쓴다. 새 의존성이 없다.

    python tools/dashboard.py --logdir artifacts/logs/stil/<tag> --port 7860
    python tools/dashboard.py --logdir artifacts/logs/stil --host 0.0.0.0 --port 7860 \
        --window 20 --refresh-sec 5

--logdir 아래에 training_log.csv 가 여러 개면 실험별로 나누어 함께 보여준다.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ENV_ROOT = Path(__file__).resolve().parents[1]
VENDOR_TOOLS = ENV_ROOT.parent / "tools"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_analysis import DASHBOARD_METRICS, find_training_logs, iter_training_log, warn  # noqa: E402


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
    if not logs:
        warn(f"training_log.csv 를 찾지 못했다: {logdir}")
    return {
        "logdir": str(logdir),
        "window": window,
        "runs": [read_run(p, metrics, window) for p in logs],
    }


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
canvas{width:100%;height:90px;display:block;margin-top:8px}
</style></head><body>
<header><h1>DogFightEnv 지표 대시보드</h1>
<span class="meta" id="meta">불러오는 중…</span>
<span class="meta">회색 = raw · 노랑 = 이동평균</span></header>
<div id="root"></div>
<script>
const METRICS = __METRICS__, REFRESH = __REFRESH__;
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
async function tick(){
  let d; try{ d=await (await fetch("/data")).json(); }catch(e){ document.getElementById("meta").textContent="데이터 로드 실패: "+e; return; }
  document.getElementById("meta").textContent=`${d.logdir} · 실험 ${d.runs.length}개 · 창 ${d.window} · ${new Date().toLocaleTimeString()}`;
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
    sec.appendChild(g); root.appendChild(sec);
  }
}
tick(); setInterval(tick, REFRESH*1000);
</script></body></html>"""


def make_handler(logdir: Path, metrics: tuple[str, ...], window: int, refresh: int):
    """요청마다 로그를 다시 읽는다. 파일이 갱신되면 재시작 없이 반영된다."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/data"):
                body = json.dumps(collect(logdir, metrics, window),
                                  ensure_ascii=False).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            else:
                page = (PAGE
                        .replace("__METRICS__", json.dumps(list(metrics)))
                        .replace("__REFRESH__", str(refresh)))
                body = page.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 지표 대시보드")
    ap.add_argument("--logdir", type=Path, help="training_log.csv 가 있는 디렉터리")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="외부 접속 허용 시 0.0.0.0")
    ap.add_argument("--window", type=int, default=20, help="이동평균 창 크기")
    ap.add_argument("--refresh-sec", type=int, default=5, help="브라우저 갱신 주기")
    ap.add_argument("--metrics", nargs="*", default=list(DASHBOARD_METRICS))
    args, _unknown = ap.parse_known_args()

    if _try_vendor():
        return 0

    if args.logdir is None:
        ap.error("--logdir 이 필요하다 (벤더 대시보드가 없어 내장 서버로 실행한다)")

    metrics = tuple(args.metrics)
    snapshot = collect(args.logdir, metrics, args.window)
    print(f"logdir : {args.logdir}")
    print(f"실험    : {len(snapshot['runs'])}개")
    for run in snapshot["runs"]:
        miss = f"  없는 지표: {run['missing']}" if run["missing"] else ""
        print(f"  - {run['name']} ({run['rows']} iteration){miss}")
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/"
    print(f"\n{url}  (Ctrl+C 로 종료, {args.refresh_sec}초마다 자동 갱신)")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(args.logdir, metrics, args.window, args.refresh_sec),
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
