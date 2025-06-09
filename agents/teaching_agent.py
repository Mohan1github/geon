from agents import functional_tools
from abc import ABC
import openai
from configs.llm_config import openai_llm,groq_llm
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from configs.prompts import teaching_agent_system_prompt
import requests
from langchain.tools import WikipediaQuryRun
from langchain.utilities import WikipediaAPIWrapper
import base64

def llm_response(self,user_prompt:str)->str:
    response = self.llmChain.run(user_prompt)
    return response.content

def extractionformation_from_wikipedia(self,prompt:str):
    wiki_setup = WikipediaQuryRun(api_wrapper = WikipediaAPIWrapper())
    response = wiki_setup.run(prompt)

class TeachingAgent():
    def __init__(self,user_prompt:str,image_input = None):
        
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system",teaching_agent_system_prompt),
            ("human",{user_prompt},{image_input})
        ])
        self.llmChain = LLMChain(llm = openai_llm, prompt = self.system_prompt )
        self.user_prompt = user_prompt
        self.image_input = image_input
   
    #response from the llm combined

@functional_tools
async def combining_response(self) ->str:
    try:
        llm_response = await llm_response(self.user_prompt)
        wiki_response = await extractionformation_from_wikipedia(self.user_prompt)

        if(llm_response and wiki_response):
            combined_form = llm_response + wiki_response
            refined = self.llmChain("Give the very appropriate asnwer as mentioned in the system prompt",combined_form)
            if(refined):
                return refined.content
            else:
                return f"There have some error occurs"
    except Exception as err:
        return f"Exception:{err}"
        

    # summary with image 
    @functional_tools
    async def imagesummary(self) -> str:
        response = None
        try:
            pass
            #if both the prompt and the image is provided 
            if self.user_prompt and self.image_input:
                with open(self.image_input, "rb") as image_file:
                 base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": {self.user_prompt}},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
            #if only the image is present 
            elif self.image_input:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read the given image and provide the very essential information first about the image and then provide the over view and the extra informations.. "},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )



            return response['choices'][0]['message']['content']
        except Exception as e:
            return f" Exception occured and the exception is {e}"
        finally:
            print("Process finished.....!")

teaching_object = TeachingAgent()
if __name__ == "__main__":
    while True:
        input_from_user = input("Ask AI teaching agent:")
        teaching_object.examplefunction(input_from_user)
    
    
    

