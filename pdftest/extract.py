"""Extract text from PDF pages at different DPIs using OpenAI Vision API, comparing against PDF baseline."""

import base64
import difflib
from pathlib import Path
from typing import Dict, List

import click
import fitz  # PyMuPDF
from openai import OpenAI


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
        Extracted text
    """
    client = OpenAI(api_key=api_key)

    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)

    # Create chat completion with vision
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text from this image. Return only the text content, preserving the layout and structure as much as possible. Do not add any commentary, delimiters or special characters to the the extracted content of your own."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


def extract_text_from_pdf_with_openai(pdf_path: Path, api_key: str, model: str = "gpt-4o") -> str:
    """
    Extract text from PDF using OpenAI API.

    Args:
        pdf_path: Path to the PDF file
        api_key: OpenAI API key
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Extracted text
    """
    client = OpenAI(api_key=api_key)

    # Encode PDF to base64
    base64_pdf = encode_pdf_to_base64(pdf_path)

    # Create chat completion with PDF
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please extract all text from this PDF. Return only the text content, preserving the layout and structure as much as possible. Do not add any commentary, delimiters or special characters to the the extracted content of your own."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{base64_pdf}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


def remove_blank_lines(text: str) -> str:
    """Remove blank lines from text."""
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_blank_lines)


def generate_diff(reference_text: str, compare_text: str, reference_label: str, compare_label: str) -> str:
    """
    Generate a unified diff between two text strings.

    Args:
        reference_text: The reference text (highest DPI)
        compare_text: The text to compare against reference
        reference_label: Label for reference file
        compare_label: Label for comparison file

    Returns:
        Unified diff as a string
    """
    reference_lines = reference_text.splitlines(keepends=True)
    compare_lines = compare_text.splitlines(keepends=True)

    # Ensure lines end with newline for proper diff formatting
    if reference_lines and not reference_lines[-1].endswith('\n'):
        reference_lines[-1] += '\n'
    if compare_lines and not compare_lines[-1].endswith('\n'):
        compare_lines[-1] += '\n'

    diff = difflib.unified_diff(
        compare_lines,
        reference_lines,
        fromfile=compare_label,
        tofile=reference_label,
        lineterm=''
    )

    return ''.join(diff)


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

    # Generate diff comparisons against PDF baseline
    if extracted_texts:
        reference_text = pdf_text
        reference_label = "pdf_page_pdf.txt"

        click.echo(f"\nComparing all image extractions against PDF baseline...\n")

        for dpi in sorted(extracted_texts.keys()):
            compare_text = extracted_texts[dpi]
            compare_label = f"{pdf_name}_page{page_number}_{dpi}dpi.txt"

            diff_output = generate_diff(reference_text, compare_text, reference_label, compare_label)

            # Save diff file
            diff_filename = f"{pdf_name}_page{page_number}_diff_{dpi}dpi.txt"
            with open(diff_filename, 'w', encoding='utf-8') as f:
                if diff_output:
                    f.write(diff_output)
                else:
                    f.write("No differences found.\n")

            diff_lines = len([l for l in diff_output.splitlines() if l.startswith('+') or l.startswith('-')]) if diff_output else 0
            click.echo(f"Saved diff to {diff_filename} ({diff_lines} changed lines)")

    click.echo("\nDone!")


if __name__ == "__main__":
    main()
