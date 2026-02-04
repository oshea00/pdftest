"""Extract text from PDF using OpenAI API with custom prompt and JSON schema."""

import base64
import json
import re
import time
from pathlib import Path

import click
import fitz  # PyMuPDF
from openai import OpenAI


def encode_pdf_to_base64(pdf_path: Path) -> str:
    """Encode PDF file to base64 string."""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode("utf-8")


def parse_markdown_config(md_path: Path) -> tuple[str, dict]:
    """
    Parse markdown file to extract prompt and JSON schema.
    
    Expected format:
        # Prompt
        Your prompt text here...
        
        # Schema
        ```schema
        {
          "type": "json_schema",
          ...
        }
        ```
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        Tuple of (prompt, schema_dict)
        
    Raises:
        ValueError: If markdown cannot be parsed or is missing required sections
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract prompt section
    prompt_match = re.search(r'#\s+Prompt\s*\n(.*?)(?=\n#\s+|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not prompt_match:
        raise ValueError("Markdown file must contain a '# Prompt' section")
    
    prompt = prompt_match.group(1).strip()
    if not prompt:
        raise ValueError("Prompt section cannot be empty")
    
    # Extract schema section from code block
    schema_match = re.search(r'#\s+Schema\s*\n.*?```schema\s*\n(.*?)\n```', content, re.DOTALL | re.IGNORECASE)
    if not schema_match:
        raise ValueError("Markdown file must contain a '# Schema' section with ```schema code block")
    
    schema_json = schema_match.group(1).strip()
    if not schema_json:
        raise ValueError("Schema code block cannot be empty")
    
    try:
        schema = json.loads(schema_json)
        return prompt, schema
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema block: {e}")


def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Get the number of pages in a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF
    """
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()
    return page_count


def extract_text_from_pdf(
    pdf_path: Path,
    api_key: str,
    prompt: str,
    json_schema: dict,
    model: str = "gpt-4o"
) -> dict:
    """
    Extract text from PDF using OpenAI API with custom prompt and schema.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: OpenAI API key
        prompt: Custom prompt for text extraction
        json_schema: JSON schema for structured output
        model: OpenAI model to use (default: gpt-4o)
        
    Returns:
        Parsed JSON response as dictionary
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
                        "text": prompt
                    }
                ]
            }
        ],
        response_format=json_schema
    )
    
    # Parse and return JSON response
    result = json.loads(response.choices[0].message.content)
    return result


@click.command()
@click.argument('pdf_file', type=click.Path(exists=True))
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key (or set OPENAI_API_KEY env var)')
@click.option('--model', default='gpt-4o', help='OpenAI model to use (default: gpt-4o)')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), default='json', 
              help='Output format: json (full response) or text (concatenated text fields)')
def main(pdf_file, config_file, output_file, api_key, model, output_format):
    """
    Extract text from PDF using OpenAI API with custom prompt and JSON schema.
    
    PDF_FILE: Path to the PDF file to process
    
    CONFIG_FILE: Path to markdown file with '# Prompt' and '# Schema' sections
    
    OUTPUT_FILE: Path to save the extracted text
    
    Example:
        pdfextract document.pdf config.md output.json
        
        pdfextract document.pdf config.md output.txt --format text
    """
    if not api_key:
        click.echo("Error: OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --api-key option.", err=True)
        raise click.Abort()
    
    pdf_path = Path(pdf_file)
    config_path = Path(config_file)
    output_path = Path(output_file)
    
    click.echo(f"Processing PDF: {pdf_file}")
    click.echo(f"Config file: {config_file}")
    click.echo(f"Model: {model}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Output format: {output_format}\n")
    
    # Parse markdown config to get prompt and schema
    try:
        prompt, json_schema = parse_markdown_config(config_path)
        click.echo(f"Loaded prompt and schema from config file")
        click.echo(f"Prompt preview: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
    except ValueError as e:
        click.echo(f"Error parsing config file: {e}", err=True)
        raise click.Abort()
    
    # Get page count
    try:
        page_count = get_pdf_page_count(pdf_path)
        click.echo(f"PDF pages: {page_count}\n")
    except Exception as e:
        click.echo(f"Error reading PDF: {e}", err=True)
        raise click.Abort()
    
    # Extract text with timing
    click.echo("Extracting text from PDF...")
    start_time = time.time()
    
    try:
        extracted_data = extract_text_from_pdf(
            pdf_path=pdf_path,
            api_key=api_key,
            prompt=prompt,
            json_schema=json_schema,
            model=model
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
    except Exception as e:
        click.echo(f"Error extracting text: {e}", err=True)
        raise click.Abort()
    
    # Save output based on format
    try:
        if output_format == 'json':
            # Save full JSON response
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        else:
            # Save as text (concatenate all string values)
            text_parts = []
            for key, value in extracted_data.items():
                if isinstance(value, str):
                    text_parts.append(value)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(text_parts))
        
        click.echo(f"\n✓ Extraction completed successfully!")
        click.echo(f"  Pages processed: {page_count}")
        click.echo(f"  Processing time: {elapsed_time:.2f} seconds")
        click.echo(f"  Output saved to: {output_file}")
        
        # Show preview of extracted data
        if output_format == 'json':
            preview = json.dumps(extracted_data, indent=2, ensure_ascii=False)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            click.echo(f"\nPreview:\n{preview}")
        else:
            with open(output_path, 'r', encoding='utf-8') as f:
                preview = f.read()
            if len(preview) > 200:
                preview = preview[:200] + "..."
            click.echo(f"\nPreview:\n{preview}")
            
    except Exception as e:
        click.echo(f"Error saving output: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
