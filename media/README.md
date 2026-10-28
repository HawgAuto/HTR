# The HTR visual system

Graphite, porcelain, copper, and cool mineral tones connect a sculptural trellis motif to precise technical graphics. The halo expresses the project's identity; the diagrams explain it; the charts show the retained observations.

## Assets

- **`hero.png`** — generated conceptual trellis sculpture, with locally typeset lettering and the small selected-path mark.
- **`method-map.png`** — a schematic overview of conditioning, encoding, and row-dependent execution. It is an explanatory diagram, not a captured tensor or hardware layout.
- **`codec-fidelity.png`** — every quantized row with teacher KL in `evidence/codec-screen.json`, retaining original labels. BF16 has no KL row and is outside this low-bit scatter plot.
- **`serving-tradeoff.png`** — every configuration in `evidence/historical-serving.json`; generation and prompt-wall metrics remain separate. Bars are medians of three observations, whiskers are their min–max, not confidence intervals.
- **`chart-data.json`** — exact plotted values, original repetition arrays, and source SHA-256 bindings.

## Rebuild the graphics

From the repository root, with Pillow installed:

```bash
python media/render.py --out /path/to/rebuilt-media
```

The renderer reads the unchanged evidence files, bundled font, and `source/trellis-halo.webp`. It does not download models, generate benchmark observations, call an image API, or initialize a GPU. Final rendering was checked with Pillow 10.2.0. Small glyph/raster differences may occur with other Pillow/FreeType versions; the data and layout are explicit.

## Artwork provenance

The source sculpture was generated through the authorized OpenAI Codex subscription image endpoint. The request named `gpt-image-2.5-sunburst`, but that endpoint manages model selection server-side and returned no model identity; the artwork is therefore not labeled as verified GPT Image 2.5 output. The original response reported 1774×887 pixels and `low` quality despite the requested `high` setting. The retained visual was inspected before publication. The prompt is in `source/trellis-halo.prompt.txt`; the public source is a high-quality WebP derivative, and the complete generation receipt is retained in the operator's release records.

The related model-card heroes use paired-ribbon and branching-canopy compositions in the same material palette. Generated imagery is conceptual; benchmark numbers and technical lettering are rendered deterministically, never supplied by an image model.

## Typography and licenses

Space Grotesk is bundled unmodified from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/spacegrotesk), by Florian Karsten and its contributors. Its complete SIL Open Font License and copyright notice are preserved in [`fonts/OFL.txt`](fonts/OFL.txt). The renderer and original layout code follow the project's MIT license; the font retains its own license.
