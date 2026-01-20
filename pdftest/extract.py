"""Extract text from PDF pages at different DPIs using OpenAI Vision API, comparing against PDF baseline."""

import base64
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import click
import fitz  # PyMuPDF
from openai import OpenAI


# JSON schema for structured text extraction
EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "text_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The page title or header, if present"
                },
                "body": {
                    "type": "string",
                    "description": "The main text content of the page"
                }
            },
            "required": ["title", "body"],
            "additionalProperties": False
        }
    }
}


def extract_page_as_images(pdf_path: str, page_num: int, dpis: List[int], output_dir: Path) -> List[tuple]:
    """
    Extract a page from PDF and rasterize at multiple DPIs.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to extract (0-indexed)
        dpis: List of DPI values to rasterize at
        output_dir: Directory to save images

    Returns:
        List of tuples (dpi, image_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} does not exist. PDF has {len(doc)} pages (0-indexed).")

    page = doc[page_num]
    image_paths = []

    for dpi in dpis:
        # Calculate zoom factor (72 DPI is the base)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Save as PNG
        image_filename = f"{pdf_name}_page{page_num}_{dpi}dpi.png"
        image_path = output_dir / image_filename
        pix.save(str(image_path))

        image_paths.append((dpi, image_path))
        click.echo(f"Saved {image_filename} ({pix.width}x{pix.height} pixels)")

    doc.close()
    return image_paths


def extract_page_as_pdf(pdf_path: str, page_num: int, output_path: Path) -> Path:
    """
    Extract a single page from PDF and save as a new PDF file.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to extract (0-indexed)
        output_path: Path to save the extracted page PDF

    Returns:
        Path to the saved PDF file
    """
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} does not exist. PDF has {len(doc)} pages (0-indexed).")

    # Create a new PDF with just the requested page
    new_doc = fitz.open()
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    new_doc.save(str(output_path))
    new_doc.close()
    doc.close()

    return output_path


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pdf_to_base64(pdf_path: Path) -> str:
    """Encode PDF file to base64 string."""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode("utf-8")


def extract_text_with_openai(image_path: Path, api_key: str, model: str = "gpt-4o") -> str:
    """
    Extract text from image using OpenAI Vision API.

    Args:
        image_path: Path to the image file
        api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Extracted text (body only, excluding title)
    """
    client = OpenAI(api_key=api_key)

    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)

    # Create chat completion with vision and structured output
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text from this image. Return only the text content, preserving the layout and structure as much as possible. Do not add any commentary, delimiters or special characters to the extracted content of your own. Separate the page title/header from the main body content."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format=EXTRACTION_SCHEMA
    )

    # Parse JSON response and return body text
    result = json.loads(response.choices[0].message.content)
    return result["body"]


def extract_text_from_pdf_with_openai(pdf_path: Path, api_key: str, model: str = "gpt-4o") -> str:
    """
    Extract text from PDF using OpenAI API.

    Args:
        pdf_path: Path to the PDF file
        api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Extracted text (body only, excluding title)
    """
    client = OpenAI(api_key=api_key)

    # Encode PDF to base64
    base64_pdf = encode_pdf_to_base64(pdf_path)

    # Create chat completion with PDF using file type and structured output
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{base64_pdf}",
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please extract all text from this PDF. Return only the text content, preserving the layout and structure as much as possible. Do not add any commentary, delimiters or special characters to the extracted content of your own. Separate the page title/header from the main body content."
                    }
                ]
            }
        ],
        response_format=EXTRACTION_SCHEMA
    )

    # Parse JSON response and return body text
    result = json.loads(response.choices[0].message.content)
    return result["body"]


def remove_blank_lines(text: str) -> str:
    """Remove blank lines from text."""
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_blank_lines)


def normalize_text(text: str) -> str:
    """
    Normalize text for n-gram comparison.

    - Converts to lowercase
    - Collapses all whitespace (spaces, tabs, newlines) to single spaces
    - Removes leading/trailing whitespace
    - Preserves punctuation and special characters

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    # Collapse all whitespace to single spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def get_character_ngrams(text: str, n: int) -> Counter:
    """
    Extract character n-grams from text.

    Args:
        text: Input text (should be normalized)
        n: Size of n-grams

    Returns:
        Counter of n-gram frequencies
    """
    if len(text) < n:
        return Counter()
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return Counter(ngrams)


@dataclass
class NgramScores:
    """Container for n-gram similarity scores."""
    precision: float
    recall: float
    f1: float
    n: int


def compute_ngram_f1(reference: str, candidate: str, n: int = 4) -> NgramScores:
    """
    Compute character n-gram precision, recall, and F1 score.

    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text to evaluate
        n: Size of n-grams (default: 4)

    Returns:
        NgramScores with precision, recall, and F1
    """
    # Normalize both texts
    ref_normalized = normalize_text(reference)
    cand_normalized = normalize_text(candidate)

    # Get n-gram counts
    ref_ngrams = get_character_ngrams(ref_normalized, n)
    cand_ngrams = get_character_ngrams(cand_normalized, n)

    # Handle edge cases
    if not ref_ngrams and not cand_ngrams:
        return NgramScores(precision=1.0, recall=1.0, f1=1.0, n=n)
    if not cand_ngrams:
        return NgramScores(precision=0.0, recall=0.0, f1=0.0, n=n)
    if not ref_ngrams:
        return NgramScores(precision=0.0, recall=0.0, f1=0.0, n=n)

    # Compute overlap (intersection with minimum counts)
    overlap = sum((ref_ngrams & cand_ngrams).values())

    # Compute precision and recall
    precision = overlap / sum(cand_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())

    # Compute F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return NgramScores(precision=precision, recall=recall, f1=f1, n=n)


def compute_multi_ngram_scores(reference: str, candidate: str, n_values: List[int] = [3, 4, 5]) -> Dict[int, NgramScores]:
    """
    Compute n-gram scores for multiple n values.

    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text to evaluate
        n_values: List of n-gram sizes to compute (default: [3, 4, 5])

    Returns:
        Dictionary mapping n to NgramScores
    """
    return {n: compute_ngram_f1(reference, candidate, n) for n in n_values}


def get_quality_label(f1_score: float) -> str:
    """
    Get a quality label based on F1 score.

    Args:
        f1_score: The F1 score (0.0 to 1.0)

    Returns:
        Quality label string
    """
    if f1_score >= 0.99:
        return "Perfect"
    elif f1_score >= 0.95:
        return "Excellent"
    elif f1_score >= 0.90:
        return "Good"
    elif f1_score >= 0.80:
        return "Fair"
    else:
        return "Poor"


@click.command()
@click.argument('pdf_file', type=click.Path(exists=True))
@click.argument('page_number', type=int)
@click.option('--dpis', default='25,50,100,150,300,600', help='Comma-separated DPI values (default: 25,50,100,150,300,600)')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-4o', help='OpenAI model to use (default: gpt-4o)')
@click.option('--images-dir', default='images', help='Directory to save images (default: images)')
def main(pdf_file, page_number, dpis, api_key, model, images_dir):
    """
    Extract text from a PDF page at different DPIs using OpenAI Vision API.

    Creates a PDF baseline by extracting the page as a PDF and comparing
    all image-based extractions against this baseline.

    PDF_FILE: Path to the PDF file
    PAGE_NUMBER: Page number to extract (0-indexed)

    Example:
        pdf-extract document.pdf 0
        pdf-extract document.pdf 0 --dpis 72,150,300
        pdf-extract document.pdf 0 --model gpt-4o --api-key YOUR_API_KEY
    """
    if not api_key:
        click.echo("Error: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key option.", err=True)
        raise click.Abort()

    # Parse DPI values
    dpi_list = [int(d.strip()) for d in dpis.split(',')]

    click.echo(f"Processing PDF: {pdf_file}")
    click.echo(f"Page number: {page_number}")
    click.echo(f"DPI values: {dpi_list}")
    click.echo(f"Model: {model}\n")

    # Create output directory
    output_dir = Path(images_dir)

    # Extract page as images at different DPIs
    try:
        image_paths = extract_page_as_images(pdf_file, page_number, dpi_list, output_dir)
    except Exception as e:
        click.echo(f"Error extracting page: {e}", err=True)
        raise click.Abort()

    # Extract page as PDF and get baseline text
    click.echo(f"\nCreating PDF baseline...\n")
    pdf_page_path = Path("pdf_page.pdf")

    try:
        extract_page_as_pdf(pdf_file, page_number, pdf_page_path)
        click.echo(f"Saved page {page_number} to {pdf_page_path}")

        # Extract text from the PDF page
        click.echo(f"Extracting text from PDF using OpenAI {model}...")
        pdf_text = extract_text_from_pdf_with_openai(pdf_page_path, api_key, model)
        pdf_text = remove_blank_lines(pdf_text)

        # Save PDF-extracted text as baseline
        pdf_text_filename = "pdf_page_pdf.txt"
        with open(pdf_text_filename, 'w', encoding='utf-8') as f:
            f.write(pdf_text)

        click.echo(f"Saved PDF baseline text to {pdf_text_filename}")
        click.echo(f"Preview: {pdf_text[:100]}...\n")

    except Exception as e:
        click.echo(f"Error creating PDF baseline: {e}", err=True)
        raise click.Abort()

    click.echo(f"\nExtracting text from images using OpenAI {model}...\n")

    # Extract text from each image and store results
    pdf_name = Path(pdf_file).stem
    extracted_texts: Dict[int, str] = {}

    for dpi, image_path in image_paths:
        click.echo(f"Processing {image_path.name}...")

        try:
            extracted_text = extract_text_with_openai(image_path, api_key, model)
            # Remove blank lines from extracted text
            extracted_text = remove_blank_lines(extracted_text)
            extracted_texts[dpi] = extracted_text

            # Save to text file
            text_filename = f"{pdf_name}_page{page_number}_{dpi}dpi.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            click.echo(f"Saved extracted text to {text_filename}")
            click.echo(f"Preview: {extracted_text[:100]}...\n")

        except Exception as e:
            click.echo(f"Error extracting text from {image_path.name}: {e}", err=True)

    # Compute n-gram similarity scores against PDF baseline
    if extracted_texts:
        reference_text = pdf_text

        click.echo(f"\nComputing n-gram similarity scores against PDF baseline...\n")
        click.echo(f"{'DPI':<8} {'n=3 F1':<10} {'n=4 F1':<10} {'n=5 F1':<10} {'Avg F1':<10} {'Quality':<10}")
        click.echo("-" * 58)

        all_scores = []

        for dpi in sorted(extracted_texts.keys()):
            compare_text = extracted_texts[dpi]

            # Compute n-gram scores for n=3, 4, 5
            scores = compute_multi_ngram_scores(reference_text, compare_text, [3, 4, 5])

            # Calculate average F1 across n values
            avg_f1 = sum(s.f1 for s in scores.values()) / len(scores)

            # Get quality label
            quality = get_quality_label(avg_f1)

            # Display scores
            click.echo(f"{dpi:<8} {scores[3].f1:<10.4f} {scores[4].f1:<10.4f} {scores[5].f1:<10.4f} {avg_f1:<10.4f} {quality:<10}")

            all_scores.append({
                'dpi': dpi,
                'scores': scores,
                'avg_f1': avg_f1,
                'quality': quality
            })

        # Save detailed scores to file
        scores_filename = f"{pdf_name}_page{page_number}_scores.txt"
        with open(scores_filename, 'w', encoding='utf-8') as f:
            f.write(f"N-Gram Similarity Scores\n")
            f.write(f"========================\n")
            f.write(f"PDF: {pdf_file}\n")
            f.write(f"Page: {page_number}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Reference: PDF baseline (pdf_page_pdf.txt)\n\n")

            f.write(f"{'DPI':<8} {'n=3 F1':<10} {'n=4 F1':<10} {'n=5 F1':<10} {'Avg F1':<10} {'Quality':<10}\n")
            f.write("-" * 58 + "\n")

            for entry in all_scores:
                dpi = entry['dpi']
                scores = entry['scores']
                avg_f1 = entry['avg_f1']
                quality = entry['quality']
                f.write(f"{dpi:<8} {scores[3].f1:<10.4f} {scores[4].f1:<10.4f} {scores[5].f1:<10.4f} {avg_f1:<10.4f} {quality:<10}\n")

            f.write("\n\nDetailed Scores\n")
            f.write("===============\n\n")

            for entry in all_scores:
                dpi = entry['dpi']
                scores = entry['scores']
                f.write(f"DPI: {dpi}\n")
                for n, s in sorted(scores.items()):
                    f.write(f"  n={n}: Precision={s.precision:.4f}, Recall={s.recall:.4f}, F1={s.f1:.4f}\n")
                f.write("\n")

        click.echo(f"\nSaved detailed scores to {scores_filename}")

    click.echo("\nDone!")


if __name__ == "__main__":
    main()
