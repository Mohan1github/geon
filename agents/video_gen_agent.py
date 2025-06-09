import requests
import openai
class VideoAgent():
    def __init__(self,prompt):
        self.prompt = prompt
        import openai

openai.api_key = "your-api-key"

response = openai.Image.create(
    model="dall-e-3",
    prompt="A futuristic city at sunset, with flying cars and neon lights",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']
print("Image URL:", image_url)



    