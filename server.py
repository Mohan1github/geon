from fastapi import FastAPI, requests
from pydantic import BaseModel
from fastapi import Form
app = FastAPI()

class chatprompt(BaseModel):
    prompt_string : str


class responseModel(BaseModel):
    username:str 
    power:list[str]
    age:int

class chatpromtcontent(BaseModel):
    prompt:str
    image:str
@app.post("/chat")
async def response_for_chat(prompt:chatprompt):
    print("the input give is ",prompt)
    return f"the given prompt is {prompt.prompt_string}"

@app.post("/chat-posting-to-llm")
async def response_generated(prompt:str=Form(...)):
    return f"the given prompt is {prompt}"

@app.get("/get-response",response_model= responseModel )
async def give_the_response() ->dict:
    data = {
    "username":"queenks",
    "power":["gorgeous","memory","intelligent","alagi"],
    "age":21
    }
@app.post("/api/v1/chat/response")
async def handleresponse(prompts:chatpromtcontent):
    user_promt = prompts.prompt
    image_url = prompts.image if prompts.image!= "" else None

    return data



