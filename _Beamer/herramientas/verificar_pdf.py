"""Auditoría de navegación del Beamer. Requiere pypdf.

Uso: python herramientas/verificar_pdf.py [ruta/al/main.pdf]
Si existe un .log contiguo, comprueba también avisos de compilación.
"""
from pathlib import Path
import json
import re
import sys
from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / 'main.pdf'
pdf = PdfReader(path)
dest = {k: pdf.get_destination_page_number(v) + 1 for k, v in pdf.named_destinations.items()}
indices = {dest[k] for k in dest if re.fullmatch(r'nav-index-\d+-\d+', k)}
assert indices, 'Falta el índice detallado'
indexed = set()
count = 0
for n, page in enumerate(pdf.pages, 1):
    home = False
    for ref in page.get('/Annots', []):
        item = ref.get_object()
        if item.get('/Subtype') != '/Link':
            continue
        action = item.get('/A', {})
        target = action.get('/D') if action.get('/S') == '/GoTo' else item.get('/Dest')
        if isinstance(target, str):
            assert target in dest, (n, 'Destino inexistente', target)
            count += 1
            home |= target == 'nav-content'
            if n in indices:
                indexed.add(dest[target])
        x0, y0, x1, y1 = map(float, item['/Rect'])
        assert x0 >= -1 and y0 >= -1 and x1 <= float(page.mediabox.width) + 1 and y1 <= float(page.mediabox.height) + 1, (n, 'Enlace fuera de página')
    assert home, (n, 'Falta regreso a Contenido')
assert set(range(3, min(indices))) <= indexed, 'Hay contenido sin acceso desde el índice'
assert all(f'nav-map-{n}-0' in dest for n in range(1, 9)), 'Falta un mapa de apertura'
summaries = ['nav-frame-023', 'nav-frame-049', 'nav-frame-052', 'holo-ll-10', 'nav-frame-090', 'nav-frame-115', 'holo-gen-10', 'nav-frame-122']
assert all(k in dest for k in summaries), 'Falta un resumen de capítulo'
lengths = [dest['nav-map-5-0'] - dest['nav-map-4-0'], dest['nav-map-8-0'] - dest['nav-map-7-0']]
assert all(0 < n <= 10 for n in lengths), ('Límite holográfico excedido', lengths)
log = path.with_suffix('.log')
if log.exists():
    assert not re.search(r'Overfull|There were undefined|multiply defined|destination with the same|Rerun to get', log.read_text(encoding='utf-8', errors='replace')), 'Revisar avisos de compilación'
print(json.dumps({'pages': len(pdf.pages), 'index_pages': len(indices), 'internal_links': count, 'opening_maps': 8, 'chapter_summaries': 8, 'holography_pages': lengths, 'status': 'PASS'}, indent=2))
