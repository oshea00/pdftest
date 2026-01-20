# PDF Text Extraction Testing Tool

Test OpenAI's GPT vision models' ability to extract text from PDF pages at different DPIs.

## Features

- Extract any page from a PDF document
- Create a PDF baseline by extracting the page as a single-page PDF
- Use OpenAI's API to extract text directly from the PDF baseline
- Rasterize the page at multiple DPIs (default: 25, 50, 100, 150, 300, 600)
- Save PNG images to an `images/` subdirectory
- Use OpenAI's Vision API to extract text from each image
- **Structured output**: Uses JSON schema to separate page title/header from body content, saving only the body text for consistent comparisons
- Remove blank lines from extracted text for cleaner comparisons
- **N-gram F1 scoring**: Layout-agnostic similarity metric robust to line wrapping and formatting differences

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
7. Compute n-gram F1 similarity scores comparing each extraction to the PDF baseline
8. Save detailed scores to `AplExamples_page0_scores.txt`

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
- `{pdf_name}_page{num}_scores.txt` - N-gram F1 similarity scores for all DPI extractions

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

### Structured Output for Consistent Comparisons

OpenAI's API often prepends a page title or header to extracted text. To ensure consistent comparisons between PDF and image extractions, this tool uses **structured output** with a JSON schema that separates:

- **title**: The page title or header (discarded)
- **body**: The main text content (saved for comparison)

This approach ensures that both PDF and image extractions return only the body text, making comparisons meaningful and avoiding false positives from inconsistent title handling.

### N-Gram F1 Scoring

Traditional diff-based comparisons are sensitive to line breaks and formatting differences, which can produce noisy results when comparing OCR outputs. This tool uses **character n-gram F1 scoring**, a layout-agnostic metric widely used in OCR and speech recognition evaluation.

#### How It Works

1. **Text Normalization**: Both reference and candidate texts are normalized:
   - Convert to lowercase
   - Collapse all whitespace (spaces, tabs, newlines) to single spaces
   - Remove leading/trailing whitespace

2. **N-Gram Extraction**: Character n-grams (substrings of length n) are extracted from both texts. For example, "hello" with n=3 produces: `["hel", "ell", "llo"]`

3. **Precision/Recall/F1 Computation**:
   - **Precision**: What fraction of candidate n-grams appear in the reference?
   - **Recall**: What fraction of reference n-grams appear in the candidate?
   - **F1**: Harmonic mean of precision and recall

4. **Multiple N Values**: Scores are computed for n=3, 4, and 5, plus an average F1 across all three.

#### Why N-Gram F1 is Superior for OCR Evaluation

- **Robust to line wrapping**: Different line breaks don't affect scores
- **Smooths over small shifts**: Minor character position changes have limited impact
- **Captures local accuracy**: Character-level n-grams detect substitutions, insertions, and deletions
- **Widely validated**: Standard metric in OCR benchmarks and competitions

#### Interpreting Scores

| F1 Score | Quality Level |
|----------|---------------|
| 0.99+    | Perfect - virtually identical extraction |
| 0.95-0.99| Excellent - near-perfect extraction |
| 0.90-0.95| Good - minor errors |
| 0.80-0.90| Fair - some noticeable errors |
| < 0.80   | Poor - significant text loss or errors |

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

# Review the n-gram F1 scores to see extraction quality at each DPI
cat report_page2_scores.txt

# Example output:
# DPI      n=3 F1     n=4 F1     n=5 F1     Avg F1     Quality
# ----------------------------------------------------------
# 25       0.8234     0.7891     0.7543     0.7889     Poor
# 50       0.9012     0.8876     0.8721     0.8870     Fair
# 100      0.9534     0.9456     0.9378     0.9456     Excellent
# 150      0.9712     0.9654     0.9598     0.9655     Excellent
# 300      0.9834     0.9789     0.9745     0.9789     Excellent
# 600      0.9901     0.9876     0.9851     0.9876     Perfect

# Compare file sizes to understand cost trade-offs
ls -lh images/report_page2_*.png
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: PyMuPDF, openai, click, pillow
