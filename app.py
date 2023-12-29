from fastapi import FastAPI, File, UploadFile, HTTPException
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.llms import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()

app = FastAPI()

# Create the 'data' folder if it doesn't exist
DATA_FOLDER = "./data"


# Load data and create the index
def create_index():
    reader = SimpleDirectoryReader(input_dir=DATA_FOLDER, recursive=True)
    docs = reader.load_data()
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="You are an expert and give answer on the basis of pdf and focus on strong words and give context from PDF only",
        )
    )
    return VectorStoreIndex.from_documents(docs, service_context=service_context)



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to the 'data' folder
    file_path = os.path.join(DATA_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Recreate the index after uploading a new file
    global index
    index = create_index()

    return {"filename": file.filename, "file_path": file_path}

@app.post("/ask")
async def ask_question(data: dict):
    if "question" not in data:
        raise HTTPException(status_code=400, detail="Question not provided in JSON data")

    index = create_index()
    question = data["question"]
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    response = chat_engine.chat(question)

    return {"question": question, "answer": response.response}
