# import requests
# import base64
# # only accept one image
# def viscpm_answer_question(image_paths, question):
#     url = "http://34.143.180.202:3389/viscpm"
#     image_path = image_paths[0]
#     message = {
#         # need to modify
#         "image": base64.b64encode(open(image_path, "rb").read()).decode(),
#         "question": question,
#     }
#     resp = requests.post(url, json=message)
#     print("Response status code:", resp.status_code)
#     print("Response content:", resp.text)
#     try:
#         resp_json = resp.json()
#         ans = resp_json["answer"]
#         return ans
#     except requests.exceptions.JSONDecodeError as e:
#         print("JSON decode error:", e)
#         return None
#     resp = resp.json()
#     ans = resp["answer"]
#     return ans
# viscpm_answer_question(["I:\\Term8\\GraduationPJ\\RAG\\VisRAG\\scripts\\demo\\datastore\\resume.pdf_0.png"], "What is the name of the person in the resume?")


import requests
import base64

# this url will cause bad gateway?
url = "http://34.143.180.202:3389/viscpm"
resp = requests.post(url, json={
    # need to modify
    "image": base64.b64encode(open("I:\\Term8\\GraduationPJ\\RAG\\VisRAG\\scripts\\demo\\datastore\\resume.pdf_0.png", "rb").read()).decode(),
    "question": "描述一下这张图片",
})
resp = resp.json()
print(resp)