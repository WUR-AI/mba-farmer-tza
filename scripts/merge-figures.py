import sys
import os
from pypdf import PdfReader, PdfWriter, PageObject

DATA_PATH = "/data/mba-tza"
os.chdir(f'{DATA_PATH}/results/figures')

PDFS = [
    ('fertilizer-marginal-dose-response.pdf', 'v'), # Stack vertically
    ('fertilizer-dose-response-curve.pdf', 'h'),    # Stack horizontally
    ('fertilizer-timing-effects.pdf', 'v'),
    ('fertilizer-gps-support-strata.pdf', 'v'),
    ('fertilizer-predictive.pdf', 'h'), 
    ('fertilizer-ols.pdf', 'h'),    
    ('fertilizer-marginal-dose-response-ols.pdf', 'v'), # Stack vertically
]
 
for pdf_suffix, stacking in PDFS:
    reader_n = PdfReader(f'N-{pdf_suffix}')
    reader_p = PdfReader(f'P-{pdf_suffix}')
    page_n = reader_n.pages[0]
    page_p = reader_p.pages[0]

    if stacking == 'v':
        new_width = max(page_n.mediabox.width, page_p.mediabox.width)
        new_height = page_n.mediabox.height + page_p.mediabox.height
        dst = PageObject.create_blank_page(width=new_width, height=new_height)
        x_offset_p = new_width - page_p.mediabox.width
        dst.merge_translated_page(page_p, x_offset_p, 0)
        x_offset_n = new_width - page_n.mediabox.width
        dst.merge_translated_page(page_n, x_offset_n, page_p.mediabox.height)
    else:
        new_width = page_n.mediabox.width + page_p.mediabox.width
        new_height = max(page_n.mediabox.height, page_p.mediabox.height)
        dst = PageObject.create_blank_page(width=new_width, height=new_height)
        dst.merge_page(page_n)
        dst.merge_translated_page(page_p, page_n.mediabox.width, 0)

    writer = PdfWriter()
    writer.add_page(dst)
    with open(pdf_suffix, "wb") as f:
        writer.write(f)