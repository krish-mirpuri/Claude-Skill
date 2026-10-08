# Tiraajio listings — split by product

Source: `tiraajio_listings_2026_3.zip` (31 images, 1080x1080).

Same convention as `tata_1mg_listings/`: the zip was one flat sequence in
which a blank white page carrying the product name marked the start of each
product. Those name pages are the split points; each is kept inside its
product folder as `_name-page.jpg`, and the listing images are renamed
`01.jpg`, `02.jpg`, ... in their original order.

`manifest.csv` maps every new filename back to its original.

| # | Folder | Product | Listing images |
|---|--------|---------|----------------|
| 1 | `01_nail-paint-pen-soft-pink` | nail paint pen: soft pink | 11 |
| 2 | `02_nail-paint-pen-berry-pink` | nail paint pen: berry pink | 10 |
| 3 | `03_3-in-1-super-wash` | 3-in-1 super wash | 7 |

Total: 3 products, 31 images (28 listing images + 3 name pages).

## Notes on the source zip

- **Only 3 of the 5 small files are name pages.** `4.jpg` and `16.jpg` are
  comparably small but are colour-swatch images (the pink blob) belonging to
  the soft pink and berry pink products respectively, not separators. They
  are kept in sequence as `02.jpg` in their folders.
- **`3-in-1 super wash 1080x1080.jpg` was never renumbered** in the source.
  By content it is the hero pack shot and fills the gap at `26.jpg`, so it
  becomes `03_3-in-1-super-wash/01.jpg`.
- `1.jpg` is absent from the zip; the sequence starts at `2.jpg`.
- The soft pink product has a barcode image (`13.jpg` -> `11.jpg`) that the
  berry pink product has no equivalent of, which is why the two otherwise
  parallel sets differ by one image.
