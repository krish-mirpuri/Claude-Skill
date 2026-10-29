#!/usr/bin/env python3
"""Normalise one month's Tira/Ajio export into paste-ready Sales Dump rows.

    python3 scripts/append_month.py <month.xlsx> <soh_bridge.xlsx> <out.tsv>

Writes columns A-D and F and H-M of the Sales Dump (the typed ones). The
derived columns are ARRAYFORMULA-driven in the sheet, so pasting A:D, F, H:M
is enough - everything else fills itself in.
"""
import sys, datetime as dt, openpyxl

def grab(p, s):
    wb = openpyxl.load_workbook(p, data_only=True, read_only=True)
    try: ws = wb[s]
    except KeyError: wb.close(); return None
    r = [list(x) for x in ws.iter_rows(values_only=True)]; wb.close(); return r

def dd(s):
    s = str(s).strip()
    for f in ("%b %d, %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try: return dt.datetime.strptime(s, f).date()
        except ValueError: pass
    raise SystemExit(f"unparseable date: {s!r}")

MONTH, BRIDGE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

# SAP <-> LB bridge: prefer a SOH tab carrying both codes
lb2sap, sap2lb = {}, {}
for src in (BRIDGE, MONTH):
    rows = grab(src, "SOH")
    if not rows: continue
    for r in rows[2:]:
        a, b = (str(r[0]).strip() if r[0] else ""), (str(r[1]).strip() if r[1] else "")
        if a.startswith("4948") and b.startswith("1207"):
            lb2sap[b] = a; sap2lb[a] = b
if not lb2sap: raise SystemExit("no SAP<->LB bridge found in either file's SOH tab")

tira, ajio = grab(MONTH, "Tira Sales"), grab(MONTH, "Ajio Sales")
out = []
if tira:
    wide = len(tira[0]) > 5
    for r in tira[1:]:
        if not r[0]: continue
        lb = str(r[2] if wide else r[1]).strip()
        out.append(["Tira", dd(r[0]), lb2sap.get(lb, ""), lb,
                    r[3] if wide else r[2], str(r[1]).strip() if wide else "BAU",
                    (r[5] if wide else ""), (r[6] if wide else ""), (r[7] if wide else ""),
                    r[9] if wide else r[4], r[8] if wide else r[3]])
if ajio:
    wide = len(ajio[0]) > 4
    for r in ajio[1:]:
        if not r[0]: continue
        sap = str(r[1]).strip()
        out.append(["Ajio", dd(r[0]), sap, sap2lb.get(sap, ""),
                    r[2] if wide else "", "BAU",
                    (r[4] if wide else ""), (r[5] if wide else ""), "",
                    r[7] if wide else r[3], r[6] if wide else r[2]])
out.sort(key=lambda x: (x[1], x[0], x[2]))

with open(OUT, "w") as f:
    for r in out:
        f.write("\t".join("" if v is None else
                (v.strftime("%d-%b-%y") if isinstance(v, dt.date) else str(v)) for v in r) + "\n")

months = sorted({r[1].strftime("%b %Y") for r in out})
miss   = sorted({r[2] for r in out if not r[2]} | {r[3] for r in out if not r[3]})
print(f"rows        : {len(out)}  (Tira {sum(1 for r in out if r[0]=='Tira')}, "
      f"Ajio {sum(1 for r in out if r[0]=='Ajio')})")
print(f"months      : {months}")
print(f"qty / gmv   : {sum(r[9] for r in out):,.0f} / {sum(r[10] for r in out):,.2f}")
print(f"unbridged   : {miss or 'none'}")
print(f"written     : {OUT}")
