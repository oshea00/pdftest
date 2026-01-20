# PDF Text Extraction Testing Tool

Test OpenAI's GPT vision models' ability to extract text from PDF pages at different DPIs.

## Features

- Extract any page from a PDF document
- Create a PDF baseline by extracting the page as a single-page PDF
- Use OpenAI's API to extract text directly from the PDF baseline
- Rasterize the page at multiple DPIs (default: 25, 50, 100, 150, 300, 600)
- Save PNG images to an `images/` subdirectory
- Use OpenAI's Vision API to extract text from each image
- Save extracted text to individual `.txt` files for comparison
- Auto-generate unified diff files comparing each DPI image extraction against the PDF baseline

## Installation

1. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

## Setup

Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file with:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

Basic usage:
```bash
pdf-extract <pdf_file> <page_number>
```

Example:
```bash
pdf-extract AplExamples.pdf 0
```

This will:
1. Extract page 0 (first page) from `AplExamples.pdf` as a single-page PDF (`pdf_page.pdf`)
2. Send the PDF to OpenAI's API for text extraction (baseline)
3. Save the PDF baseline text to `pdf_page_pdf.txt`
4. Create PNG images at 25, 50, 100, 150, 300, and 600 DPI in the `images/` folder
5. Send each image to OpenAI's Vision API for text extraction
6. Save results to `AplExamples_page0_25dpi.txt`, `AplExamples_page0_50dpi.txt`, etc.
7. Generate diff files comparing each DPI image extraction to the PDF baseline

### Options

- `--dpis`: Comma-separated DPI values (default: 25,50,100,150,300,600)
  ```bash
  pdf-extract document.pdf 0 --dpis 72,150,300
  ```

- `--model`: OpenAI model to use (default: gpt-4o)
  ```bash
  pdf-extract document.pdf 0 --model gpt-4o
  ```

- `--api-key`: OpenAI API key (if not set as environment variable)
  ```bash
  pdf-extract document.pdf 0 --api-key sk-...
  ```

- `--images-dir`: Directory to save images (default: images)
  ```bash
  pdf-extract document.pdf 0 --images-dir my_images
  ```

### Page Numbering

Pages are 0-indexed:
- Page 0 = first page
- Page 1 = second page
- etc.

## Output Files

Generated files:
- `pdf_page.pdf` - Single-page PDF extracted from the source document
- `pdf_page_pdf.txt` - Text extracted from the PDF baseline (used for comparison)
- `images/{pdf_name}_page{num}_{dpi}dpi.png` - Rasterized images at various DPIs
- `{pdf_name}_page{num}_{dpi}dpi.txt` - Extracted text for each DPI image
- `{pdf_name}_page{num}_diff_{dpi}dpi.txt` - Unified diff comparing each DPI image to the PDF baseline

## Approach & Reasoning

### PDF Baseline vs. Highest DPI Image

This tool uses a **PDF-based baseline** for comparison rather than comparing against the highest DPI image. Here's why:

#### Why PDF Baseline is Superior

1. **Native Format Preservation**: PDFs preserve the original document structure, fonts, and text encoding. When the LLM extracts text from a PDF, it can leverage this native representation rather than trying to reconstruct text from pixels.

2. **No Rasterization Artifacts**: Even at very high DPI (e.g., 600 or 1200), rasterized images introduce:
   - Anti-aliasing artifacts
   - Lossy compression (PNG/JPEG)
   - Font rendering variations
   - Potential loss of fine details or subtle formatting

3. **True Ground Truth**: The PDF baseline represents the closest approximation to the source document's actual text content. Comparing image extractions against this baseline shows how much information is lost during the rasterization process at different DPIs.

4. **More Accurate Quality Assessment**: By comparing against the PDF baseline, we can:
   - Measure the actual degradation caused by converting to images at various DPIs
   - Identify the minimum DPI threshold where image-based extraction becomes acceptable
   - Understand the trade-offs between image quality (DPI) and extraction accuracy

#### Use Cases

This approach is particularly valuable for:
- **Document digitization workflows** - Determining optimal scanning/rasterization DPI
- **OCR quality testing** - Comparing vision-based extraction against native PDF text
- **Cost optimization** - Finding the lowest DPI that maintains acceptable extraction quality (lower DPI = smaller images = lower API costs)
- **Performance benchmarking** - Understanding LLM capabilities with different input modalities (PDF vs. images)

## Example Workflow

```bash
# Extract page 2 from a PDF
pdf-extract report.pdf 2

# Review the PDF baseline text
cat pdf_page_pdf.txt

# Review the auto-generated diff files to see how each DPI image extraction compares to the PDF baseline
cat report_page2_diff_25dpi.txt
cat report_page2_diff_100dpi.txt
cat report_page2_diff_300dpi.txt

# Compare file sizes to understand cost trade-offs
ls -lh images/report_page2_*.png
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: PyMuPDF, openai, click, pillow
