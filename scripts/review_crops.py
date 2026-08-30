"""Adjudicate review_queue.csv in a browser: show the crop, take the verdict.

remap_to_av14.py parks every box whose COCO class maps to two or more AV-14
classes (or to none) in review_queue.csv with an empty `decision` column. All
219 of them sat unreviewed because the one thing that decides them -- what the
crop actually shows -- is the one thing a CSV editor cannot display.

  python scripts/review_crops.py

Opens http://localhost:8765. Number keys pick a class, D drops the box, S skips
it, B steps back. Every verdict rewrites the CSV immediately, so closing the tab
never loses work and re-running resumes where you left off. Then:

  python scripts/merge_review_queue.py --report
  python scripts/merge_review_queue.py --out datasets/av14_merged
"""
import argparse
import csv
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.classes_av import AV_CLASSES  # noqa: E402

PAD = 0.6           # context around the box in the zoomed crop, as a fraction
MAX_SIDE = 700      # cap served image size; these are viewed, not measured


def load(queue: Path):
    with queue.open(encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), r.fieldnames


def save(queue: Path, rows, fields):
    """Rewrite via a temp file: a crash mid-write must not eat 200 verdicts."""
    tmp = queue.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(queue)


def candidates(row):
    """'signboard/door/pole' -> ['signboard', 'door', 'pole']."""
    return [c for c in (row.get("suspected_av14") or "").split("/") if c]


def render(img_dir: Path, row, zoom: bool):
    """Full frame with the box drawn, or a padded crop around it."""
    img = cv2.imread(str(img_dir / row["image"]))
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy, bw, bh = (float(row[k]) for k in ("cx", "cy", "w", "h"))
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    if zoom:
        px, py = int(bw * w * PAD), int(bh * h * PAD)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
        img = img[cy1:cy2, cx1:cx2].copy()
        x1, y1, x2, y2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    if img.size == 0:
        return None
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),
                  max(2, img.shape[1] // 250))
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


PAGE = """<!doctype html><meta charset=utf-8>
<title>Review queue</title>
<style>
 body{margin:0;font:15px system-ui,sans-serif;background:#14161a;color:#e8e8ea}
 header{padding:10px 16px;background:#1d2026;display:flex;gap:16px;
   align-items:center;flex-wrap:wrap}
 #bar{flex:1;height:8px;background:#2c3038;border-radius:9px;min-width:120px}
 #bar div{height:100%;background:#4ea1ff;border-radius:9px;width:0}
 main{display:flex;gap:10px;padding:10px;flex-wrap:wrap}
 figure{margin:0;flex:1;min-width:280px;text-align:center}
 figcaption{font-size:12px;color:#8b93a1;padding:4px}
 img{max-width:100%;border-radius:8px;background:#000}
 #keys{padding:10px 16px;display:flex;gap:8px;flex-wrap:wrap}
 button{font:600 15px system-ui;padding:11px 15px;border-radius:9px;
   border:1px solid #3a3f4a;background:#242833;color:#e8e8ea;cursor:pointer}
 button:hover{border-color:#4ea1ff}
 button b{color:#4ea1ff;margin-right:6px}
 .drop{border-color:#7a2f2f} .all button{font-size:13px;padding:8px 11px}
 .all{padding:0 16px 16px;border-top:1px solid #262a33;margin-top:6px}
 h3{font-size:12px;color:#8b93a1;font-weight:600;padding:10px 16px 0;margin:0}
 #done{padding:40px;text-align:center;font-size:18px;display:none}
</style>
<header>
  <strong>Review queue</strong>
  <span id=pos></span><div id=bar><div></div></div>
  <span id=meta></span>
</header>
<div id=app>
<main>
  <figure><img id=zoom alt="Zoomed crop with the box outlined">
    <figcaption>crop + context</figcaption></figure>
  <figure><img id=full alt="Whole frame with the box outlined">
    <figcaption>whole frame</figcaption></figure>
</main>
<div id=keys></div>
<h3>ALL AV-14 CLASSES</h3>
<div id=all class="all"></div>
</div>
<div id=done>All rows decided. Now run:<br><br>
  <code>python scripts/merge_review_queue.py --out datasets/av14_merged</code>
</div>
<script>
let cur=null, hot=[];
async function load(){
  const r=await fetch('/next').then(r=>r.json());
  if(!r.row){document.getElementById('app').style.display='none';
    document.getElementById('done').style.display='block';
    document.getElementById('pos').textContent=r.total+' / '+r.total;
    document.getElementById('bar').firstElementChild.style.width='100%';return;}
  cur=r;
  document.getElementById('zoom').src='/img?i='+r.i+'&zoom=1&t='+Date.now();
  document.getElementById('full').src='/img?i='+r.i+'&zoom=0&t='+Date.now();
  document.getElementById('pos').textContent=(r.done+1)+' / '+r.total;
  document.getElementById('bar').firstElementChild.style.width=
    (100*r.done/r.total)+'%';
  document.getElementById('meta').textContent='COCO "'+r.row.coco_class+'"';
  hot=r.candidates.slice(0,9);
  const keys=document.getElementById('keys');
  keys.innerHTML='';
  hot.forEach((c,n)=>keys.appendChild(btn(String(n+1),c,c)));
  keys.appendChild(btn('D','drop','drop','drop'));
  keys.appendChild(btn('S','skip',null));
  keys.appendChild(btn('B','back','__back__'));
  const all=document.getElementById('all');
  all.innerHTML='';
  r.classes.forEach(c=>all.appendChild(btn('',c,c)));
}
function btn(key,label,val,cls){
  const b=document.createElement('button');
  if(cls)b.className=cls;
  b.innerHTML=(key?'<b>'+key+'</b>':'')+label;
  b.onclick=()=>decide(val);
  return b;
}
async function decide(v){
  if(v===null){await fetch('/skip',{method:'POST'});return load();}
  await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({i:cur.i,decision:v})});
  load();
}
addEventListener('keydown',e=>{
  if(!cur)return;
  const k=e.key.toLowerCase();
  if(k>='1'&&k<='9'){const c=hot[+k-1]; if(c){e.preventDefault();decide(c);}}
  else if(k==='d'){e.preventDefault();decide('drop');}
  else if(k==='s'){e.preventDefault();decide(null);}
  else if(k==='b'){e.preventDefault();decide('__back__');}
});
load();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    rows, fields, queue, img_dir = [], [], None, None
    skipped = set()
    history = []

    def log_message(self, *a):
        pass                       # the progress bar in the page is the log

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _pending(self):
        return [i for i, r in enumerate(self.rows)
                if not (r.get("decision") or "").strip()
                and i not in self.skipped]

    def do_GET(self):
        if self.path == "/":
            return self._send(200, "text/html; charset=utf-8",
                              PAGE.encode("utf-8"))
        if self.path.startswith("/next"):
            pending = self._pending()
            total = len(self.rows)
            done = total - len(pending) - len(self.skipped)
            if not pending:
                return self._send(200, "application/json",
                                  json.dumps({"row": None, "total": total,
                                              "done": done}).encode())
            i = pending[0]
            row = self.rows[i]
            return self._send(200, "application/json", json.dumps({
                "i": i, "row": row, "candidates": candidates(row),
                "classes": list(AV_CLASSES), "total": total,
                "done": done}).encode())
        if self.path.startswith("/img"):
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&"))
            jpg = render(self.img_dir, self.rows[int(q["i"])], q["zoom"] == "1")
            if jpg is None:
                return self._send(404, "text/plain", b"image not found")
            return self._send(200, "image/jpeg", jpg)
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/skip":
            pending = self._pending()
            if pending:
                self.skipped.add(pending[0])
            return self._send(200, "application/json", b'{"ok":true}')
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        i, decision = data.get("i"), data.get("decision")
        if decision == "__back__":
            # step back over the last verdict and re-offer it
            if self.history:
                prev = self.history.pop()
                self.rows[prev]["decision"] = ""
                self.skipped.discard(prev)
            else:                       # nothing decided yet: un-skip instead
                if self.skipped:
                    self.skipped.discard(max(self.skipped))
        else:
            self.rows[i]["decision"] = decision
            self.history.append(i)
        save(self.queue, self.rows, self.fields)
        self._send(200, "application/json", b'{"ok":true}')


def main(queue: Path, img_dir: Path, port: int, open_browser: bool):
    rows, fields = load(queue)
    if "decision" not in (fields or []):
        raise SystemExit(f"{queue} has no 'decision' column")
    missing = {r["image"] for r in rows if not (img_dir / r["image"]).exists()}
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) named in {queue} are not under {img_dir}, "
            f"e.g. {sorted(missing)[:3]}. Pass --images pointing at the set the "
            "queue was generated from.")
    Handler.rows, Handler.fields = rows, fields
    Handler.queue, Handler.img_dir = queue, img_dir
    todo = sum(1 for r in rows if not (r.get("decision") or "").strip())
    url = f"http://localhost:{port}"
    print(f"{todo} of {len(rows)} rows still undecided -> {url}")
    print("keys: 1-9 class - D drop - S skip - B back. Ctrl+C when done.")
    if open_browser:
        webbrowser.open(url)
    try:
        # Threading, not plain HTTPServer: the page pulls TWO images per row and
        # Chrome opens parallel keep-alive connections, so a single-threaded
        # server blocks on the first idle one and the tool silently stops
        # responding -- verdicts then look like they were never pressed.
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        left = sum(1 for r in rows if not (r.get("decision") or "").strip())
        print(f"\nsaved {queue} - {len(rows) - left} decided, {left} left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path,
                    default=Path("datasets/av14_seed/review_queue.csv"))
    ap.add_argument("--images", type=Path,
                    default=Path("datasets/av14_seed/images"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    main(a.queue, a.images, a.port, not a.no_browser)
