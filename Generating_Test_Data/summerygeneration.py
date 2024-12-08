import openai
import pandas as pd

openai.api_key = "key should insert here"
def summarize_text(text):
    """
    Summarize the given text using GPT-3.5 API (ChatCompletion).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes texts."},
                {"role": "user", "content": f"Summarize the following text in less than 3 sentences:\n\n{text}"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

def process_excel(input_file, output_file):
    """
    Read an Excel file, summarize the text, and save results to a new Excel file.
    """
    # Read the Excel file
    df = pd.read_excel(input_file)

    # Ensure the required columns exist
    if 'ID' not in df.columns or 'Text' not in df.columns:
        raise ValueError("The Excel file must contain 'ID' and 'Text' columns.")

    # Create a new column for summaries
    summaries = []

    print("Starting summarization...")
    for idx, row in df.iterrows():
        text = row['Text']
        id_value = row['ID']
        print(f"Summarizing ID: {id_value}...")
        summary = summarize_text(text)
        summaries.append(summary)

    # Add the summaries to the DataFrame
    df['Summary'] = summaries

    # Save the DataFrame to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"Summarization completed. Results saved to {output_file}.")

# Example usage
input_file = "path"
output_file = "path"

process_excel(input_file, output_file)
