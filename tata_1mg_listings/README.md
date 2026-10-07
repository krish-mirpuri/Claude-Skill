# Tata 1mg listings — split by product

Source: `tata_1mg_listings_1.zip` (64 images, 1080x1080).

The zip was one flat sequence where a blank white page carrying the product
name marked the start of each product. Those name pages were used as the split
points; each one is kept inside its product folder as `_name-page.jpg`, and the
actual listing images are renamed `01.jpg`, `02.jpg`, ... in their original
order.

`manifest.csv` maps every new filename back to its original filename.

| # | Folder | Product | Listing images |
|---|--------|---------|----------------|
| 1 | `01_dull-skin-body-wash` | dull skin body wash | 6 |
| 2 | `02_hand-sanitiser-50ml` | hand sanitiser 50 ml | 6 |
| 3 | `03_mosquito-lotion` | mosquito lotion | 6 |
| 4 | `04_3-in-1-super-wash` | 3-in-1 super wash | 6 |
| 5 | `05_daily-face-wash` | daily face wash | 7 |
| 6 | `06_kumkumadi-soap` | kumkumadi soap | 6 |
| 7 | `07_kumkumadi-lotion` | kumkumadi lotion | 6 |
| 8 | `08_face-cream-sunscreen-spf-50` | face cream + sunscreen SPF 50 | 7 |
| 9 | `09_gel-sunscreen-spf-50` | gel sunscreen SPF 50 | 6 |

Total: 9 products, 64 images (55 listing images + 8 name pages + 1 product
with no name page).

## Notes on the source zip

- **Product 8 has no name page.** The zip contains only 8 blank name pages but
  9 products. Images `52.jpg`–`57.jpg` are clearly a separate product (face
  cream + sunscreen SPF 50) from both kumkumadi lotion before them and gel
  sunscreen SPF 50 after them. `08_face-cream-sunscreen-spf-50` therefore has
  no `_name-page.jpg`.
- **Two files were never renumbered** in the source and were slotted back by
  content:
  - `3-in-1 super wash 1080x1080.jpg` -> `04_3-in-1-super-wash/01.jpg` (the
    hero shot, the gap at `24.jpg`)
  - `spf 50 this_that_that.jpg` -> `08_face-cream-sunscreen-spf-50/07.jpg`
    (the "you might receive any of these tubes" page, the gap at `58.jpg`)
- `1.jpg` is absent from the zip; the sequence starts at `2.jpg`.
