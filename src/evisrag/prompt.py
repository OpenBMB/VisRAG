def evidence_promot_grpo(query):
    return f"""You are an AI Visual QA assistant. I will provide you with a question and several images. Please follow the four steps below:

Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information. Then, examine each image one by one, paying special attention to aspects related to the question. Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

Question and images are provided below. Please follow the steps as instructed.
Question: {query}
"""


def evidence_promot_oneshot(query):
    return f"""You are an AI Visual QA assistant. I will provide you with a question and several images. Please follow the four steps below:

Step 1: Observe the Images
First, analyze the question and consider what types of images may contain relevant information. Then, examine each image one by one, paying special attention to aspects related to the question. Identify whether each image contains any potentially relevant information.
Wrap your observations within <observe></observe> tags.

Step 2: Record Evidences from Images
After reviewing all images, record the evidence you find for each image within <evidence></evidence> tags.
If you are certain that an image contains no relevant information, record it as: [i]: no relevant information(where i denotes the index of the image).
If an image contains relevant evidence, record it as: [j]: [the evidence you find for the question](where j is the index of the image).

Step 3: Reason Based on the Question and Evidences
Based on the recorded evidences, reason about the answer to the question.
Include your step-by-step reasoning within <think></think> tags.

Step 4: Answer the Question
Provide your final answer based only on the evidences you found in the images.
Wrap your answer within <answer></answer> tags.
Avoid adding unnecessary contents in your final answer, like if the question is a yes/no question, simply answer "yes" or "no".
If none of the images contain sufficient information to answer the question, respond with <answer>insufficient to answer</answer>.

Formatting Requirements:
Use the exact tags <observe>, <evidence>, <think>, and <answer> for structured output.
It is possible that none, one, or several images contain relevant evidence.
If you find no evidence or few evidences, and insufficient to help you answer the question, follow the instruction above for insufficient information.

The following is an example with three images and one evidence, which you can refer to:
Query: How many more people felt inspired frequently than depressed frequently?
Your Respond:
<observe>To answer the question "How many more people felt inspired frequently than depressed frequently," I need to focus on sections of the image that display quantitative data for "inspired frequently" and "depressed frequently," then compute the difference once the numbers are identified. Then I will see the images one by one.
Image1 is a bar chart showing the percentage of social media users who feel certain emotions \"frequently\" and \"sometimes\" while using social media platforms. The emotions listed are Amused, Angry, Connected, Inspired, Depressed, and Lonely. The chart provides the following data:\n\n- Amused: Frequently 44%, Sometimes 44%, NET 88%\n- Angry: Frequently 25%, Sometimes 47%, NET 71%\n- Connected: Frequently 21%, Sometimes 49%, NET 71%\n- Inspired: Frequently 16%, Sometimes 53%, NET 69%\n- Depressed: Frequently 13%, Sometimes 36%, NET 49%\n- Lonely: Frequently 7%, Sometimes 24%, NET 31%\n\nThe query asks for the difference between the number of people who felt inspired frequently and those who felt depressed frequently. From the chart, we can see that 16% of social media users feel inspired frequently and 13% feel depressed frequently.
So image1 contains relevant information to answer the query about the difference between the number of people who felt inspired frequently and those who felt depressed frequently. Evidence of image1: From the bar chart, 16% of social media users feel inspired frequently and 13% feel depressed frequently.
Image2 is a bar chart showing the share of respondents feeling satisfied or dissatisfied with various aspects of life and government. The query asks for the difference between the number of people who felt inspired frequently and those who felt depressed frequently. However, the chart does not provide information about the frequency of feelings like \"inspired\" or \"depressed.\" It only shows satisfaction and dissatisfaction levels for different topics.
Therefore, there is no relevant information in the image2 to answer the query.
Image3 is a bar chart showing the share of respondents feeling optimistic, hopeful, cautious, and pessimistic. The percentages for each category are as follows: Optimistic at 43%, Hopeful at 49%, Cautious at 6%, and Pessimistic at 2%. There is no information provided about how many people felt inspired frequently or depressed frequently, nor is there any comparison between these two sentiments.
So image3 does not contain any relevant information to answer the query about the difference between the number of people who felt inspired frequently and those who felt depressed frequently.</observe>
<evidence>
[1]: From the bar chart, 16% of social media users feel inspired frequently and 13% feel depressed frequently.
[2]: no relevant information
[3]: no relevant information
</evidence>
<think>The query is how many more people felt inspired frequently than depressed frequently, and the evidence found from image1 states 16% of social media users feel inspired frequently and 13% feel depressed frequently, thus the difference is 16% - 13% = 3%. So 3% people felt inspired frequently than depressed frequently.</think>
<answer>3%</answer>

Question and images are provided below. Please follow the steps as instructed.
Question: {query}
"""


def baseline(query, method):
    if method == "concat":
        return f"""You are an AI assistant. I will provide a question and a image.
Put your reasoning process within <think></think>.
Please answer the questions based on the image given to you, and put your your final answer in <answer></answer>.
Please try to remove irrelevant content in the final answer.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>insufficient to answer</answer> after your thinking.

Question: {query}
    """
    else:
        return f"""You are an AI assistant. I will provide a question and some images.
Put your reasoning process within <think></think>.
Please answer the questions based on the multiple pictures given to you, and put your your final answer in <answer></answer>.
Please try to remove irrelevant content in the final answer.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>insufficient to answer</answer> after your thinking.

Question: {query}
"""
    

def COCOT(query):
    return f"""You are an AI assistant. I will provide a query, and some images. Follow these two steps:

In the first step:
Find the similarities and differences of these images.
Output separately all the same points and all the differences you find.
Then reason the answer of question based on your findings.
Put these process within <think></think>.

In the second step:
Put your your final answer in <answer></answer>.
Please try to remove irrelevant content in the final answer. Like if the question is asking for yes or no, then only answer <answer>yes</answer> after your thinking.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>insufficient to answer</answer> after your thinking.

Query: {query}
"""


def CCOT(query):
    return f"""You are an AI assistant. I will provide a query and some images. Follow these two steps:

In the first step:
For the provided images and its associated question, generate a scene graph for each images includes the following:
    1. Objects that are relevant to answering the question
    2. Object attributes that are relevant to answering the question
    3. Object relationships that are relevant to answering the question
Then reason the answer of question based on scene graphs.
Put these process within <think></think>.

In the second step:
Put your your final answer in <answer></answer> based on the scene graphs.
Please try to remove irrelevant content in the final answer. Like if the question is asking for yes or no, then only answer <answer>yes</answer> after your thinking.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>insufficient to answer</answer> after your thinking.

Query: {query}
"""


def DDCOT(query):
    return f"""You are an AI assistant. I will provide a query and some images. Follow these two steps:

In the first step:
Please think step-by-step about the preliminary knowledge to answer the question, deconstruct the question as completely as possible down to necessary sub-questions based on context, questions and options. Then with the aim of helping humans answer the original question, try to answer the sub-questions. The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...
Sub-answers:
1. <sub-answer 1> or 'Uncertain'
2. <sub-answer 2> or 'Uncertain'
...
For a question, assume that you do not have any information about the picture, but try to answer the sub-questions and prioritize whether your general knowledge can answer it, and then consider whether the context can help. If sub-questions can be answered, then answer in as short a sentence as possible. If sub-questions cannot be determined without information inimages, please formulate corresponding sub-answer into "Uncertain".

In the second step:
Put your your final answer in <answer></answer> based on the scene graphs.
Please try to remove irrelevant content in the final answer. Like if the question is asking for yes or no, then only answer <answer>yes</answer> after your thinking.
If you think there are no relevant information from the picture that can help you answer the question, answer <answer>insufficient to answer</answer> after your thinking.

Query: {query}
"""