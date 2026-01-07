from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

video_id = "dvx-DWWVTik"  # Example YouTube video ID

try:
    # Fetch the transcript for the given YouTube video ID
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id, languages=['en'])

    # Combine the transcript segments into a single string
    transcript = " ".join([segment.text for segment in transcript_data])
    print("Transcript fetched successfully.",transcript)
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")