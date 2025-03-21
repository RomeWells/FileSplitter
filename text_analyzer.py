import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in the text for the specified model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text(text: str, max_tokens: int = 3000, model: str = "gpt-3.5-turbo") -> List[str]:
    """Split text into chunks that don't exceed max_tokens."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph, model)
        
        if paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chunk_tokens = 0
            
            # Split large paragraphs into sentences
            sentences = paragraph.replace(". ", ".\n").split("\n")
            sentence_chunk = []
            sentence_chunk_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence, model)
                if sentence_chunk_tokens + sentence_tokens + 1 <= max_tokens:
                    sentence_chunk.append(sentence)
                    sentence_chunk_tokens += sentence_tokens + 1
                else:
                    chunks.append(". ".join(sentence_chunk) + ".")
                    sentence_chunk = [sentence]
                    sentence_chunk_tokens = sentence_tokens
            
            if sentence_chunk:
                chunks.append(". ".join(sentence_chunk) + ".")
                
        elif current_chunk_tokens + paragraph_tokens + 2 <= max_tokens:
            current_chunk.append(paragraph)
            current_chunk_tokens += paragraph_tokens + 2
        else:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_tokens = paragraph_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

def analyze_chunk(chunk: str, analysis_type: str, model: str = "gpt-3.5-turbo") -> Dict:
    """Analyze a chunk of text based on the specified analysis type."""
    print(f"Analyzing chunk with {count_tokens(chunk, model)} tokens...")
    
    prompts = {
        "summarize": "Provide a detailed summary of the following text that captures all the main points and important details:",
        "categorize": "Identify the main topics, themes, and categories in the following text. Provide a structured analysis:",
        "extract_entities": "Extract all named entities (people, organizations, locations, products, etc.) from the following text. Format as a structured list:",
        "key_points": "Identify and list all key points, arguments, and important information in the following text:",
        "questions": "Generate 5-10 important questions that could be answered based on the information in this text:",
    }
    
    if analysis_type not in prompts:
        analysis_type = "summarize"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a detailed text analyzer that provides comprehensive, structured analysis of text."},
            {"role": "user", "content": f"{prompts[analysis_type]}\n\n{chunk}"}
        ],
        temperature=0.3,
    )
    
    return {
        "chunk_text": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Preview of chunk
        "analysis": response.choices[0].message.content,
        "analysis_type": analysis_type,
        "token_count": count_tokens(chunk, model)
    }

def combine_analyses(analyses: List[Dict], analysis_type: str, model: str = "gpt-3.5-turbo") -> str:
    """Combine individual chunk analyses into a comprehensive analysis."""
    # Extract just the analyses without the chunk previews
    analysis_texts = [item["analysis"] for item in analyses]
    all_analyses = "\n\n---\n\n".join(analysis_texts)
    
    prompt_map = {
        "summarize": "Synthesize these summaries into one comprehensive summary:",
        "categorize": "Combine these category analyses into one unified set of topics and themes, removing duplicates and organizing logically:",
        "extract_entities": "Combine these entity lists into one comprehensive list, removing duplicates and organizing by entity type:",
        "key_points": "Combine these key points into one comprehensive list, removing duplicates and organizing logically:",
        "questions": "Select the most important and diverse questions from these lists, organizing them by topic:",
    }
    
    prompt = prompt_map.get(analysis_type, "Combine these analyses into one comprehensive analysis:")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a skilled analyst who can synthesize multiple analyses into one coherent document."},
            {"role": "user", "content": f"{prompt}\n\n{all_analyses}"}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Analyze text from a file using OpenAI API")
    parser.add_argument("file_path", help="Path to the text file to analyze")
    parser.add_argument("--analysis", "-a", choices=["summarize", "categorize", "extract_entities", "key_points", "questions", "all"], 
                        help="Type of analysis to perform", default="summarize")
    parser.add_argument("--output", "-o", help="Output file for the analysis", default="analysis_results")
    parser.add_argument("--model", "-m", help="OpenAI model to use", default="gpt-3.5-turbo")
    parser.add_argument("--max-tokens", "-t", help="Maximum tokens per chunk", type=int, default=3000)
    parser.add_argument("--save-chunks", "-s", action="store_true", help="Save individual chunk analyses")
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Load the text file
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Split text into chunks
    print(f"Splitting text into chunks (max {args.max_tokens} tokens per chunk)...")
    chunks = split_text(text, args.max_tokens, args.model)
    print(f"Text split into {len(chunks)} chunks.")
    
    # Define analysis types to run
    analysis_types = ["summarize", "categorize", "extract_entities", "key_points", "questions"] if args.analysis == "all" else [args.analysis]
    
    results = {}
    
    # Process each analysis type
    for analysis_type in analysis_types:
        print(f"\nPerforming {analysis_type} analysis...")
        
        # Analyze each chunk
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} for {analysis_type}...")
            try:
                analysis = analyze_chunk(chunk, analysis_type, args.model)
                chunk_analyses.append(analysis)
                print(f"Chunk {i+1} analyzed successfully.")
            except Exception as e:
                print(f"Error analyzing chunk {i+1}: {e}")
        
        # Save individual chunk analyses if requested
        if args.save_chunks:
            chunks_output = f"{args.output}_chunks_{analysis_type}.json"
            try:
                with open(chunks_output, 'w', encoding='utf-8') as f:
                    json.dump(chunk_analyses, f, indent=2)
                print(f"Individual chunk analyses saved to {chunks_output}")
            except Exception as e:
                print(f"Error saving chunk analyses: {e}")
        
        # Combine analyses if we have multiple chunks
        if len(chunk_analyses) > 1:
            print(f"Creating combined {analysis_type} analysis...")
            try:
                combined_analysis = combine_analyses(chunk_analyses, analysis_type, args.model)
                results[analysis_type] = combined_analysis
                print(f"Combined {analysis_type} analysis created successfully.")
            except Exception as e:
                print(f"Error creating combined analysis: {e}")
                # Fall back to individual analyses
                results[analysis_type] = "\n\n===== INDIVIDUAL ANALYSES =====\n\n".join(
                    [item["analysis"] for item in chunk_analyses]
                )
        else:
            results[analysis_type] = chunk_analyses[0]["analysis"]
    
    # Save the results
    for analysis_type, result in results.items():
        output_path = f"{args.output}_{analysis_type}.txt"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"{analysis_type.capitalize()} analysis saved to {output_path}")
        except Exception as e:
            print(f"Error saving {analysis_type} analysis: {e}")
    
    # Create a combined results file
    if len(analysis_types) > 1:
        combined_output = f"{args.output}_all.txt"
        try:
            with open(combined_output, 'w', encoding='utf-8') as f:
                for analysis_type in analysis_types:
                    f.write(f"\n\n{'=' * 50}\n{analysis_type.upper()}\n{'=' * 50}\n\n")
                    f.write(results[analysis_type])
            print(f"All analyses combined and saved to {combined_output}")
        except Exception as e:
            print(f"Error saving combined analyses: {e}")

if __name__ == "__main__":
    main()