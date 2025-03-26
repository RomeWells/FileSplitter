
This is a Long Text Analyzer
Designed to Take Long Text File and summarize it into smaller chunks.
This is also used in Fathom for summaries of transacripts.
When creating long Zoom Videos sometimes there is a huge transcripts that takes place.

ChatGPT has a limit for the file size that you can import.
This script solves this problem by simply taking input of a long text transcript and summarizing it into smaller chunks.

You can use this for Youtube Transcripts and improve SEO for your sites.

***How to Use***

Save the script as text_analyzer.py
For basic usage:
bashCopypython text_analyzer.py your_large_file.txt

For comprehensive analysis of all types:
bashCopypython text_analyzer.py your_large_file.txt --analysis all

For a specific type of analysis:
bashCopypython text_analyzer.py your_large_file.txt --analysis categorize

To save individual chunk analyses:
bashCopypython text_analyzer.py your_large_file.txt --save-chunks


The script will generate several output files that give you different perspectives on your text:

analysis_results_summarize.txt
analysis_results_categorize.txt
analysis_results_extract_entities.txt
analysis_results_key_points.txt
analysis_results_questions.txt
analysis_results_all.txt (containing all analyses)


To use Text_clarifer.py file simply type in the command prompt the following

text_clarifier.py analysis_results_summarize.txt "tell me what the summary is about"
