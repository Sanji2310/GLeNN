"""Draw the GLeNN Racing suit decals into one mask atlas.

Runs with plain Python and Pillow (not inside Blender):
    python make_decals.py

Writes textures/suit_decals.png, 1024 x 512, used as data (Non-Color):
  R  fill of the lettering, G  outline ring around the fill.
Cells (UV, origin bottom left):
  number    u 0.0-0.5, v 0.0-1.0   "61"
  stacked   u 0.5-1.0, v 0.5-1.0   "GLeNN" over "RACING"
  line      u 0.5-1.0, v 0.0-0.5   "GLeNN RACING"

Fonts are open-licensed (SIL OFL): Inter and Big Shoulders. The numerals are
sheared for a forward lean. Nothing here copies a real team's marks.
"""

import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "textures", "suit_decals.png")
FONT_NUMBER = ["/mnt/skills/examples/canvas-design/canvas-fonts/BigShoulders-Bold.ttf",
               "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"]
FONT_WORD = ["/usr/share/fonts/opentype/inter/Inter-BoldItalic.otf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError(paths)


def text_mask(w, h, lines, paths, fill=0.80, shear=0.0, gap=0.10):
    """White text centred in a w x h mask, scaled to fill the cell."""
    big = 400
    f = font(paths, big)
    probe = Image.new("L", (1, 1))
    d = ImageDraw.Draw(probe)
    boxes = [d.textbbox((0, 0), t, font=f) for t in lines]
    tw = max(b[2] - b[0] for b in boxes)
    th = sum(b[3] - b[1] for b in boxes) + gap * big * (len(lines) - 1)
    s = min(w * fill / (tw * (1 + abs(shear))), h * fill / th)
    f = font(paths, max(8, int(big * s)))
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    boxes = [d.textbbox((0, 0), t, font=f) for t in lines]
    total = sum(b[3] - b[1] for b in boxes) + gap * f.size * (len(lines) - 1)
    y = (h - total) / 2
    for t, b in zip(lines, boxes):
        x = (w - (b[2] - b[0])) / 2 - b[0]
        d.text((x, y - b[1]), t, font=f, fill=255)
        y += (b[3] - b[1]) + gap * f.size
    if shear:
        img = img.transform(img.size, Image.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0), Image.BICUBIC)
    return img


def outline(mask, width):
    grown = mask.filter(ImageFilter.MaxFilter(2 * width + 1))
    return ImageChops.subtract(grown, mask)


def main():
    W, H = 1024, 512
    R = Image.new("L", (W, H), 0)
    Gc = Image.new("L", (W, H), 0)
    num = text_mask(512, 512, ["61"], FONT_NUMBER, fill=0.78, shear=0.16)
    R.paste(num, (0, 0))
    Gc.paste(outline(num, 13), (0, 0))
    stacked = text_mask(512, 256, ["GLeNN", "RACING"], FONT_WORD, fill=0.86, gap=0.06)
    R.paste(stacked, (512, 0))
    line = text_mask(512, 256, ["GLeNN RACING"], FONT_WORD, fill=0.90)
    R.paste(line, (512, 256))
    img = Image.merge("RGBA", (R, Gc, Image.new("L", (W, H), 0), Image.new("L", (W, H), 255)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, optimize=True)
    print(OUT, os.path.getsize(OUT))


if __name__ == "__main__":
    main()
