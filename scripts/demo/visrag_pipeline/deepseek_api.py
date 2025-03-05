# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import openai
import base64
client = OpenAI(api_key="sk-47d4f6a08f6247b087d6e753b3897870", base_url="https://api.deepseek.com")
# 上传图片并获取文件 ID
def upload_image(file_path):
    with open(file_path, "rb") as file:
        response = openai.File.create(
            file=file,
            purpose="fine-tune"  # 可用于训练或存储
        )
    return response["id"]
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def deepseek_answer_question(image_paths, question):
    # img_ids = [upload_image(img_path) for img_path in image_paths]
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": question}
    ]
    for img_path in image_paths:
        image_data = encode_image(img_path)
        messages.append({"role": "user", "content": f"data:image/jpeg;base64,{image_data}"})
        
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages = messages,
        stream=False
    )


    ans = response.choices[0].message.content
    return ans

# ans = deepseek_answer_question(["scripts/demo/datastore/resume.pdf_0.png"], "What is the name of the person in the resume?")
# print(ans)