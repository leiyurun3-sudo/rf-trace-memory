from pathlib import Path
import os
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(os.environ.get('R85_ARCHIVE', str(Path(__file__).resolve().parents[2] / 'results/R85')))
OUT = ROOT / 'figures' / 'final_qualitative'
ROWS = [
    ('821000000008', 'Grass to water'),
    ('613000000003', 'White wall to red'),
    ('000000000021', 'Make the parrots kiss'),
    ('122000000002', 'Replace camera with phone'),
]
COLS = [('Source', None), ('CORR25', 'RELEASE_CORR25'),
        ('BOTH25', 'RELEASE_BOTH25'), ('ENERGY', 'ENERGY_MATCH25'),
        ('FlowEdit N26', 'FLOWEDIT_N26')]
# Width is 7.1 inches at 300 dpi; all labels are at least 8 pt at that width.
dpi = 300
width = 2130
margin, gap, header, row_label, row_gap = 12, 10, 56, 47, 12
cell = (width - 2 * margin - 4 * gap) // 5
height = header + 4 * (row_label + cell + row_gap)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 34)
bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 34)
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)
for j, (label, _) in enumerate(COLS):
    x = margin + j * (cell + gap) + cell / 2
    draw.text((x, 8), label, fill='black', font=bold, anchor='mt')
for i, (case, label) in enumerate(ROWS):
    y = header + i * (row_label + cell + row_gap)
    draw.text((margin, y), f'{label}  |  seed 0', fill='black', font=font)
    for j, (_, arm) in enumerate(COLS):
        path = (ARCHIVE / 'sources' / f'{case}.jpg' if arm is None else
                ARCHIVE / f'R85_pie_{case}_s0_{arm}' / 'preview.jpg')
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as raw:
            raw = raw.convert('RGB')
            scale = min(cell / raw.width, cell / raw.height)
            raw = raw.resize((round(raw.width * scale), round(raw.height * scale)), Image.Resampling.LANCZOS)
            x = margin + j * (cell + gap) + (cell - raw.width) // 2
            image.paste(raw, (x, y + row_label + (cell - raw.height) // 2))
image.save(OUT.with_suffix('.png'), dpi=(dpi, dpi))
image.save(OUT.with_suffix('.jpg'), quality=95, subsampling=0, dpi=(dpi, dpi))
page_w = 7.1 * 72
page_h = page_w * height / width
pdf = canvas.Canvas(str(OUT.with_suffix('.pdf')), pagesize=(page_w, page_h))
pdf.setTitle('Seed-0 qualitative comparisons, archived JPEG previews')
pdf.drawImage(ImageReader(image), 0, 0, width=page_w, height=page_h)
pdf.save()
print(f'Created 5-column figure: {width} x {height} px, {page_w/72:.1f} x {page_h/72:.2f} inches')

