/* Render docs/ICIICTE2026_paper.md to an IEEE-style two-column .docx.
 * Tailored to this manuscript's markdown conventions:
 *   ## SECTION   ### A. Subsection   | tables |   *Fig. N. caption* + (`path`)
 * Wide figures (fig1, fig3) get single-column section breaks. Run from repo root:
 *   node docs/figures/render_paper.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, ImageRun, BorderStyle, HeadingLevel,
} = require("docx");

const SRC = "docs/ICIICTE2026_paper.md";
const OUT = "docs/ICIICTE2026_paper.docx";
const WIDE = new Set(["fig1_architecture.png", "fig3_dataset.png"]);
const PAGE = { width: 12240, height: 15840 }; // US Letter
const MARGIN = 1080;                          // 0.75in
const COLW = (PAGE.width - 2 * MARGIN - 360) / 2; // two cols, 0.25in gutter

// ---- inline markdown -> TextRuns ------------------------------------------
function runs(text, base = {}) {
  const out = [];
  // tokenise **bold**, *italic*, `code`
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  const push = (t, extra) => t && out.push(new TextRun({
    text: t, font: "Times New Roman", size: base.size || 20,
    bold: base.bold || extra === "b", italics: base.italics || extra === "i",
  }));
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), "b");
    else if (tok.startsWith("`")) push(tok.slice(1, -1));
    else push(tok.slice(1, -1), "i");
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const para = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 80, line: 232, lineRule: "auto" },
  ...(opts.extra || {}),
});

// ---- block parser ----------------------------------------------------------
const raw = fs.readFileSync(SRC, "utf8");
// manuscript = from the title "# Hazard-First" to the checklist separator
const start = raw.indexOf("# Hazard-First");
const end = raw.indexOf("## POST-TRAINING INSERTION CHECKLIST");
const body = raw.slice(start, end).replace(/\n---\n---\n/g, "\n");
const lines = body.split("\n");

const blocks = [];
let i = 0;
while (i < lines.length) {
  const L = lines[i];
  if (!L.trim()) { i++; continue; }
  if (L.startsWith("|")) {                       // table
    const rows = [];
    while (i < lines.length && lines[i].startsWith("|")) {
      const cells = lines[i].split("|").slice(1, -1).map(c => c.trim());
      if (!cells.every(c => /^[-: ]+$/.test(c))) rows.push(cells);
      i++;
    }
    blocks.push({ t: "table", rows });
    continue;
  }
  if (L.startsWith("### ")) { blocks.push({ t: "h3", text: L.slice(4) }); i++; continue; }
  if (L.startsWith("## ")) { blocks.push({ t: "h2", text: L.slice(3) }); i++; continue; }
  if (L.startsWith("# ")) { blocks.push({ t: "h1", text: L.slice(2) }); i++; continue; }
  // paragraph: gather until blank
  const buf = [];
  while (i < lines.length && lines[i].trim() && !lines[i].startsWith("|") &&
         !lines[i].startsWith("#")) { buf.push(lines[i].trim()); i++; }
  let text = buf.join(" ");
  const fig = text.match(/\(`(docs\/figures\/[^`]+)`\)/);
  if (fig) {
    const caption = text.replace(fig[0], "").trim();
    blocks.push({ t: "img", src: fig[1],
                  caption: caption ? caption.replace(/^\*|\*$/g, "") : null });
    continue;
  }
  if (/^[-\s]+$/.test(text)) continue;         // stray horizontal rules
  blocks.push({ t: "p", text });
}

// ---- docx assembly ---------------------------------------------------------
function mkTable(rows) {
  const n = rows[0].length;
  const total = COLW - 40;
  const w = Math.floor(total / n);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: Array(n).fill(w),
    rows: rows.map((r, ri) => new TableRow({
      children: r.map(c => new TableCell({
        width: { size: w, type: WidthType.DXA },
        margins: { top: 20, bottom: 20, left: 40, right: 40 },
        children: [new Paragraph({
          children: runs(c, { size: 15, bold: ri === 0 }),
          alignment: ri === 0 ? AlignmentType.CENTER : AlignmentType.LEFT,
          spacing: { after: 0 },
        })],
      })),
    })),
  });
}

function mkImage(src, wide) {
  const buf = fs.readFileSync(src);
  const px = require("child_process");
  // read PNG dimensions from IHDR
  const wpx = buf.readUInt32BE(16), hpx = buf.readUInt32BE(20);
  const wIn = wide ? 6.6 : 3.25;
  const hIn = wIn * hpx / wpx;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 40 },
    children: [new ImageRun({
      type: "png", data: buf,
      transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
    })],
  });
}

const sections = [];
let cur = [];                      // current two-column children
const flushTwoCol = () => {
  if (cur.length) {
    sections.push({
      properties: {
        page: { size: PAGE, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
        column: { count: 2, space: 360 },
        type: "continuous",
      },
      children: cur,
    });
    cur = [];
  }
};
const oneCol = (children) => sections.push({
  properties: {
    page: { size: PAGE, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
    type: "continuous",
  },
  children,
});

// title block (single column): title + abstract + keywords
const titleIdx = blocks.findIndex(b => b.t === "h1");
const firstSectionIdx = blocks.findIndex(b => b.t === "h2");
const head = [];
head.push(new Paragraph({
  children: runs(blocks[titleIdx].text, { size: 40, bold: true }),
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
}));
for (let k = titleIdx + 1; k < firstSectionIdx; k++) {
  const b = blocks[k];
  if (b.t === "p") head.push(para(b.text, { run: { size: 18 }, after: 120 }));
}
oneCol(head);

let pendingCaption = null;
for (let k = firstSectionIdx; k < blocks.length; k++) {
  const b = blocks[k];
  if (b.t === "h2") {
    cur.push(new Paragraph({
      children: runs(b.text, { size: 20 }),
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 100 },
    }));
  } else if (b.t === "h3") {
    cur.push(new Paragraph({
      children: [new TextRun({ text: b.text, font: "Times New Roman", size: 20, italics: true })],
      spacing: { before: 140, after: 80 },
    }));
  } else if (b.t === "table") {
    cur.push(mkTable(b.rows));
    cur.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
  } else if (b.t === "img") {
    const base = path.basename(b.src);
    const cap = b.caption || pendingCaption;
    const capPara = cap ? new Paragraph({
      children: runs(cap, { size: 16 }), alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
    }) : null;
    if (WIDE.has(base)) {
      flushTwoCol();
      const kids = [mkImage(b.src, true)];
      if (capPara) kids.push(capPara);
      oneCol(kids);
    } else {
      cur.push(mkImage(b.src, false));
      if (capPara) cur.push(capPara);
    }
    pendingCaption = null;
  } else if (b.t === "p") {
    if (/^\*Fig\. \d+\./.test(b.text)) { pendingCaption = b.text.replace(/^\*|\*$/g, ""); continue; }
    if (/^\*\*TABLE/.test(b.text)) {
      cur.push(new Paragraph({
        children: runs(b.text, { size: 16 }), alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 60 },
      }));
      continue;
    }
    cur.push(para(b.text));
  }
}
flushTwoCol();

const doc = new Document({
  styles: { default: { document: { run: { font: "Times New Roman", size: 20 } } } },
  sections,
});
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, buf.length, "bytes");
});
