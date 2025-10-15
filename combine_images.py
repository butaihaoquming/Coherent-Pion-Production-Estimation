# make_collage_by_rules.py
# pip install pillow
import os, glob, re, math
from PIL import Image, ImageDraw

# ========= USER SETTINGS =========
IMG_DIR   = r"/Users/jerry/Desktop/root/Root file/results_by_date/2025-09-10_Ver2"
OUT_PATH  = r"//Users/jerry/Desktop/root/Root file/results_by_date/2025-09-10_Ver2/combination.png"
ROWS_DEF  = [(0,30),(30,60),(60,100),(100,150),(150,200),(200,250)]   # Tπ rows
COLS_DEF  = [(0,25),(25,50),(50,75),(75,150)]                         # AE−Tπ groups
ORDER     = "row-major"  # keep it simple L→R each row
MARGIN    = 20
GAP       = 6
BG_COLOR  = (255,255,255)
RESIZE_MODE = "contain"  # keep image fully visible
UPSCALE   = False        # NEVER enlarge images
TILE_SIZE = "auto"       # "auto" = use first image's size (recommended for uniform 1192x756)

# ========= REGEX (robust for your filenames) =========
# Example: 0 < T_{#pi^{+}} < 30 MeV0 < Available Energy - T_{#pi^{+}} < 25 MeV aUNSCALED.png
T_PAT  = re.compile(r'(\d+)\s*<\s*T[^<]*<\s*(\d+)\s*MeV', re.I)

# AE−Tπ: grab the two numbers around "Available Energy - T" … "< … MeV"
AE_PAT = re.compile(r'(\d+)\s*<\s*Available\s*Energy\s*-\s*T[^<]*<\s*(\d+)\s*MeV', re.I)

def list_images(folder):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(folder, e)))
    return files

def detect_scaled_flag(name_upper):
    # IMPORTANT: check UNSCALED first (since "UNSCALED" contains "SCALED")
    if "UNSCALED" in name_upper:
        return "UNSCALED"
    if "SCALED" in name_upper:
        return "SCALED"
    raise ValueError("Scaled/Unscaled flag not found in: " + name_upper)

def parse_filename(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    # Try primary patterns
    mT  = T_PAT.search(name)
    mAE = AE_PAT.search(name)
    if not (mT and mAE):
        # fallback: very robust — pull four integers in order:
        # [T_low, T_high, AE_low, AE_high]
        nums = re.findall(r'\d+', name)
        if len(nums) >= 4:
            t_low, t_high, ae_low, ae_high = map(int, nums[:4])
            flag = detect_scaled_flag(name.upper())
            return (t_low, t_high), (ae_low, ae_high), flag
        raise ValueError(f"Cannot parse ranges from: {base}")

    t_low, t_high   = int(mT.group(1)), int(mT.group(2))
    ae_low, ae_high = int(mAE.group(1)), int(mAE.group(2))
    flag = detect_scaled_flag(name.upper())  # "UNSCALED" contains "SCALED", so check UNSCALED first
    return (t_low, t_high), (ae_low, ae_high), flag


def find_row_idx(t_pair):
    try:
        return ROWS_DEF.index(t_pair)
    except ValueError:
        raise ValueError(f"Tπ range {t_pair} not in defined rows {ROWS_DEF}")

def find_col_base(ae_pair):
    try:
        return COLS_DEF.index(ae_pair) * 2  # each AE group takes 2 columns
    except ValueError:
        raise ValueError(f"AE−Tπ range {ae_pair} not in defined cols {COLS_DEF}")

def fit_image(im, target_wh, mode="contain", bg=(255,255,255)):
    tw, th = target_wh
    iw, ih = im.size
    if mode == "contain":
        scale = min(tw/iw, th/ih)
        if not UPSCALE:
            scale = min(scale, 1.0)
        nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
        im2 = im.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (iw, ih) else im
        canvas = Image.new("RGB", (tw, th), bg)
        canvas.paste(im2, ((tw - nw)//2, (th - nh)//2))
        return canvas
    elif mode == "cover":
        scale = max(tw/iw, th/ih)
        if not UPSCALE:
            scale = max(min(scale, 1.0), 1.0)
        nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
        im2 = im.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (iw, ih) else im
        ox, oy = (nw - tw)//2, (nh - th)//2
        return im2.crop((ox, oy, ox+tw, oy+th))
    else:
        raise ValueError("RESIZE_MODE must be 'contain' or 'cover'")

def main():
    files = list_images(IMG_DIR)
    if not files:
        raise SystemExit("No images found.")

    # Parse and bucket every file
    buckets = {}  # (row, col) -> path
    imgs = {}     # cache opened images
    first_size = None
    for f in files:
        try:
            t_rng, ae_rng, flag = parse_filename(f)
            r = find_row_idx(t_rng)
            c = find_col_base(ae_rng) + (0 if flag == "UNSCALED" else 1)
            if (r, c) in buckets:
                print(f"[warn] duplicate for cell {(r,c)}; keeping first: {os.path.basename(buckets[(r,c)])}")
            else:
                buckets[(r, c)] = f
                im = Image.open(f).convert("RGB")
                imgs[f] = im
                if first_size is None:
                    first_size = im.size
        except Exception as e:
            print("[skip]", f, "->", e)

    rows = len(ROWS_DEF)
    cols = len(COLS_DEF) * 2  # 4 AE groups × 2 (U,S) = 8 columns

    # Decide tile size
    if TILE_SIZE == "auto":
        if first_size is None:
            raise SystemExit("Could not determine tile size.")
        tile_w, tile_h = first_size
    else:
        tile_w, tile_h = TILE_SIZE

    # Canvas size
    W = MARGIN*2 + cols*tile_w + (cols-1)*GAP
    H = MARGIN*2 + rows*tile_h + (rows-1)*GAP
    canvas = Image.new("RGB", (W, H), BG_COLOR)

    # Place images row-major
    missing = []
    for r in range(rows):
        for c in range(cols):
            x = MARGIN + c*(tile_w + GAP)
            y = MARGIN + r*(tile_h + GAP)
            key = (r, c)
            if key not in buckets:
                missing.append(key)
                continue
            f = buckets[key]
            tile = fit_image(imgs[f], (tile_w, tile_h), RESIZE_MODE, BG_COLOR)
            canvas.paste(tile, (x, y))

    # Save
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    canvas.save(OUT_PATH, optimize=True)
    print(f"Saved collage: {OUT_PATH}  ({W}x{H}px)")
    if missing:
        print("Missing cells (row, col):", missing)

if __name__ == "__main__":
    main()
