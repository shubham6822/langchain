from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

video_id = "Rni7Fz7208c"  
transcript = ""

try:
    # Fetch the transcript for the given YouTube video ID
    api = YouTubeTranscriptApi()
    transcript_data = api.fetch(video_id, languages=['en'])

    # Combine the transcript segments into a single string
    transcript = " ".join([segment.text for segment in transcript_data])
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

# Split the transcript into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.create_documents([transcript])

# Generate embeddings for each chunk using Google Generative AI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])