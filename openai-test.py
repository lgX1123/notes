#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
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output a prompt(about 50 words) to describe a picture of the object given by user."},
    {"role": "user", "content": "Dog."}
  ]
)

prompt = completion.choices[0].message.content
print(prompt)

image = client.images.generate(
    model="dall-e-2",
    prompt=prompt,
    n=1,
    size="256x256",
    style="natural"
)
image_url = image.data[0].url
print(image)
