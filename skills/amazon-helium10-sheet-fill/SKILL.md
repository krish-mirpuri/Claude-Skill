---
name: amazon-helium10-sheet-fill
description: Populate a Google Sheet competitor-analysis tab with Amazon.in product data (name, price, gramage, price/gm, L30 sales, L30 units, DRR sales, DRR units) sourced from Amazon search results using the Helium 10 Chrome extension. Use when the user wants to fill a "Category Analysis" style sheet with competitor products for a given keyword (e.g. shampoo, toothpaste, soap).
---

# Amazon + Helium 10 → Google Sheet competitor data fill

## Goal
Given a Google Sheet with columns `Competitor Link | Name | Price | Gramage | Price per GM | L30 Sales | L30 Units | DRR Sales | DRR Units` and a search keyword, fill one row per unique product found on Amazon.in search results for that keyword, using the Helium 10 extension's on-page data panel for sales figures.

## Preconditions
- Use the **Claude in Chrome** browser tools (`mcp__claude-in-chrome__*`), not the sandboxed Browser pane — the user's real Chrome has the Helium 10 extension logged in and active.
- Load tools first: `ToolSearch` with `select:mcp__claude-in-chrome__tabs_context_mcp,mcp__claude-in-chrome__navigate,mcp__claude-in-chrome__computer,mcp__claude-in-chrome__read_page,mcp__claude-in-chrome__tabs_create_mcp,mcp__claude-in-chrome__tabs_close_mcp,mcp__claude-in-chrome__find,mcp__claude-in-chrome__form_input,mcp__claude-in-chrome__get_page_text,mcp__claude-in-chrome__browser_batch`
- If Helium 10 shows a logged-out/login state on a product page, STOP and tell the user immediately so they can log back in — do not guess or skip data.

## Per-product data to capture
On each Amazon product detail page, Helium 10 injects a "Product Summary for '<ASIN>'" panel near the top with:
- **30-Day Revenue** → this is `L30 Sales`
- **Unit Sales** (under the revenue figure) → this is `L30 Units`
This panel can take 2-3 seconds to load after navigation — `wait` ~3s before reading it, then screenshot to confirm real numbers replaced the loading placeholders.

From the standard Amazon listing itself, capture:
- **Name**: the full product title (trim to something reasonably descriptive)
- **Price**: the current selling price (not the struck-through MRP)
- **Gramage**: pack size/volume as advertised (e.g. `340ml`, `1200ml`, `1 Ltr` → normalize to ml)
- **Competitor Link**: canonical product URL, `https://www.amazon.in/<slug>/dp/<ASIN>` (drop tracking query params)

## Derived columns (enter as sheet formulas, not hardcoded values)
- `Price per GM` = `=C{row}/{gramage_in_ml}` (e.g. `=C5/340`)
- `DRR Sales` = `=F{row}/30`
- `DRR Units` = `=G{row}/30`

## Workflow loop
1. Open (or reuse) a Chrome tab on `https://www.amazon.in/s?k=<keyword>`.
2. Scroll down in increments (max `scroll_amount: 10` per call) past the Helium 10 "Keyword Niche Summary" and sponsored carousel to reach the actual results grid.
3. Screenshot to identify the next unique product tile not yet added to the sheet (track by ASIN or product name to avoid duplicates — sponsored carousels often repeat the same 4 products).
4. Click the product tile/title — this opens a **new tab**. Note its tabId from the tool result.
5. `wait` ~3s, then `screenshot` the new tab to confirm the Helium 10 panel has loaded (revenue/rating fields show numbers, not gray placeholder bars). If still loading, wait again.
6. Read off Name, Price, Gramage from the listing area, and L30 Sales / L30 Units from the Helium 10 panel.
7. Switch to the Google Sheet tab. Click the first empty row's Competitor Link cell and type the row via keyboard, using `Tab` to move across columns in one `browser_batch` call:
   `link → Tab → name → Tab → price → Tab → gramage → Tab → =C{row}/{ml} → Tab → l30sales → Tab → l30units → Tab → =F{row}/30 → Tab → =G{row}/30 → Enter`
8. Click elsewhere to deselect, screenshot to verify the row rendered correctly (no `#VALUE!`, no stray link-popup covering cells).
9. Close the product's Chrome tab (`tabs_close_mcp`) — don't let completed product tabs pile up.
10. Repeat from step 3 for the next unique product, keeping strict listing order (top-to-bottom, left-to-right as products first appear) — do not skip around or re-order after the fact.

## Gotchas learned the hard way
- Amazon's Indian-locale numbers (e.g. `89,90,970`) sometimes get typed into the sheet as literal text with commas, which breaks arithmetic formulas referencing that cell (`#VALUE!`). Always type L30 Sales / L30 Units as plain digit strings with no commas.
- Google Sheets' right-click **"Insert 1 row above"** can silently fail to register before a subsequent click lands, causing you to type over an existing row's data instead of a new blank one. After any insert, screenshot to confirm a genuinely blank row exists before typing into it. If data gets overwritten, immediately re-enter the lost row after finishing the current one.
- The sponsored carousel at the top of search results reshuffles order between page loads — re-fetching the same search URL can show the same products in a different sequence. Track products by ASIN (visible in Helium 10's per-tile overlay, e.g. `B0H6BV9FPY`) to avoid double-counting when the order shifts.
- Keep exactly ONE product tab open at a time beyond the search-results tab and the sheet tab; close each as soon as its row is committed.
- If the user asks for N unique products, keep scrolling deeper into results (past sponsored, into organic, then further pages if needed) rather than stopping at the first screen's worth.

## New sheet setup
If no existing sheet was provided, create a new Google Sheet first with a header row:
`Competitor Link | Name | Price | Gramage | Price per GM | L30 Sales | L30 Units | DRR Sales | DRR Units`
then run the workflow loop above starting at row 2.
