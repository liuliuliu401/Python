from openai import OpenAI
import numpy as np
from config import openai_base_url, openai_api_key


client = OpenAI(api_key = openai_api_key,
                base_url = openai_base_url)

model = "text-embedding-3-small"

def get_embedding(text, model=model):
    return client.embeddings.create(input=text, model=model).data[0].embedding



def cosine_sim(text_line: str):
    # 调用OpenAI Codex来计算文本之间的相似性
    embeddings = [get_embedding(text) for text in text_line]

    # 计算两个向量的余弦相似度

    len_text = len(text_line)
    cosine_sim = -0.01 * np.ones([len_text, len_text])
    for row in range(len_text): 
        for col in range(len_text):
            if row == col:
                cosine_sim[row, col] = 1
            elif cosine_sim[col, row] != -0.01:
                cosine_sim[row, col] = cosine_sim[col, row]
            else:
                cosine_sim[row, col] = np.dot(embeddings[row], embeddings[col]) / \
                            (np.linalg.norm(embeddings[row]) * np.linalg.norm(embeddings[col]))
    return cosine_sim        
        
            
if __name__ == "__main__":
    # 示例文本行
    text_line = ["男的", "在大专/末流985211上大学", "加入了学校的动漫社", "经常在游戏/动画/二次元同好群里活跃", "聊天栏基本上都是二次元群"]
    print(cosine_sim(text_line))


