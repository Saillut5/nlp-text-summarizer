import torch
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, text: str, min_length: int = 30, max_length: int = 150) -> str:
        if not text.strip():
            return ""
        
        # Ensure the input text is not too short for summarization
        if len(text.split()) < min_length:
            return text # Return original text if it's already very short

        try:
            summary = self.summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Could not summarize text."

    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        if not text.strip():
            return ""

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # A very simple extractive summarization: take the first N sentences
        # In a real scenario, this would involve sentence scoring (e.g., TextRank)
        return " ".join(sentences[:num_sentences])

if __name__ == "__main__":
    summarizer = TextSummarizer()

    long_text = """
    Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals and humans. Example tasks in which AI is used include speech recognition, computer vision, translation between natural languages, and other mappings of inputs. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), and competing at the highest level in strategic game systems (such as chess and Go). 

    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI. For example, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. This phenomenon is known as the AI effect. 

    Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success, and renewed funding. AI research has tried and discarded many different approaches, including simulating the brain, modeling human problem solving, formal logic, large knowledge bases, and imitating animal behavior. 

    In the 21st century, AI research has seen a resurgence due to advances in computational power, large datasets, and theoretical breakthroughs, particularly in deep learning. This has led to significant progress in various subfields and widespread adoption of AI technologies across industries.
    """

    print("\n--- Original Text ---")
    print(long_text)

    print("\n--- Abstractive Summary (Default) ---")
    abstractive_summary = summarizer.summarize_text(long_text)
    print(abstractive_summary)

    print("\n--- Extractive Summary (First 2 sentences) ---")
    extractive_summary = summarizer.extractive_summarize(long_text, num_sentences=2)
    print(extractive_summary)

    short_text = "This is a very short text."
    print("\n--- Summarizing a short text ---")
    short_summary = summarizer.summarize_text(short_text)
    print(short_summary)
