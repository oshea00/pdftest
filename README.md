# PDF Text Extraction Testing Tool

Test OpenAI's GPT vision models' ability to extract text from PDF pages at different DPIs.

## Features

- Extract any page from a PDF document
- Rasterize the page at multiple DPIs (default: 25, 50, 100, 150, 300, 600)
- Save PNG images to an `images/` subdirectory
- Use OpenAI's Vision API to extract text from each image
- Save extracted text to individual `.txt` files for comparison
- Auto-generate unified diff files comparing each DPI extraction against the highest DPI

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
1. Extract page 0 (first page) from `AplExamples.pdf`
2. Create PNG images at 25, 50, 100, 150, 300, and 600 DPI in the `images/` folder
3. Send each image to OpenAI's API for text extraction
4. Save results to `AplExamples_page0_25dpi.txt`, `AplExamples_page0_50dpi.txt`, etc.
5. Generate diff files comparing each lower DPI to the 600 DPI reference

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
- `images/{pdf_name}_page{num}_{dpi}dpi.png` - Rasterized images
- `{pdf_name}_page{num}_{dpi}dpi.txt` - Extracted text for each DPI
- `{pdf_name}_page{num}_diff_{dpi}dpi.txt` - Unified diff comparing each DPI to the highest DPI

## Example Workflow

```bash
# Extract page 2 from a PDF
pdf-extract report.pdf 2

# Review the auto-generated diff files to see extraction differences
cat report_page2_diff_25dpi.txt
cat report_page2_diff_100dpi.txt
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: PyMuPDF, openai, click, pillow
