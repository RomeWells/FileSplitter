import os
import argparse
from pathlib import Path
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

def ask_about_summary(summary_text, question, model="gpt-3.5-turbo"):
    """Ask a question about the summary text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about text summaries."},
            {"role": "user", "content": f"Here is a summary of a document:\n\n{summary_text}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Ask questions about a summary")
    parser.add_argument("summary_file", help="Path to the summary text file")
    parser.add_argument("question", help="Question to ask about the summary")
    parser.add_argument("--model", "-m", help="OpenAI model to use", default="gpt-3.5-turbo")
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Load the summary file
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"Error: File '{summary_path}' not found.")
        return
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Ask the question
    print(f"Asking: {args.question}")
    try:
        answer = ask_about_summary(summary_text, args.question, args.model)
        print("\nAnswer:")
        print(answer)
    except Exception as e:
        print(f"Error getting answer: {e}")

if __name__ == "__main__":
    main()