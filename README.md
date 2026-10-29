# Tira / Ajio Cat. View builder

Rebuilds the Reliance (Tira + Ajio) reporting tabs that mirror the Meesho
`Cat. View` / `Sales Dump` pattern.

## Usage

```bash
python3 scripts/build_tira_ajio.py <sep.xlsx> <aug.xlsx> <master.tsv> <out.xlsx>
```

Monthly refresh: drop the new `Tuco_Kids_<Mon>_Sales_SOH.xlsx` in as the first
argument and re-run. The script auto-detects the wide (full-column) and narrow
(stripped) variants of both `Tira Sales` and `Ajio Sales`, so either export
shape works.

`master.tsv` is tab-separated, no header, one row per SKU:

```
SKU ID <tab> ARTICLE (SAP) <tab> Product Name <tab> MRP <tab> Landing Rate
```

It is not committed — it carries pricing.

## Tabs produced

| Tab | What it holds |
|---|---|
| `Dropdown` | CATEGORY / TAGS / PLATFORM / EVENT TYPE / GST % validation lists |
| `TIRA AJIO MAPPER` | One row per SKU: SAP + LB + internal SKU ID, category, MRP, landing rate, COGS, SOH |
| `SOH Tira Ajio` | Aug vs Sep stock on hand with movement |
| `Tira Ajio Sales Dump` | Merged Tira + Ajio line items, normalised to one schema |
| `Cat.View Tira Ajio` | GMV / QTY / ASP / Net Revenue / COGS / GM / GM% by month |

## Join key

`ARTICLE (SAP)` (`4948…`) is the spine. Ajio reports it directly as `Article`;
Tira reports `ItemCode` (the LB code, `1207…`), bridged via the SAP↔LB pairs in
the August `SOH` tab. Every derived figure is a formula over the dump, so
editing the mapper reflows the whole workbook.

## Known gaps

- `COGS W/o GST` is blank by design — it is the only true input. GM stays blank
  until it is filled.
- SKU `494836841` has no Landing Rate in the master, so its Net Revenue and GM
  are blank (~17% of GMV).
- Neither platform reports customer state, so there is no `ZONE` column.
