import re

def format_math_explanation(text: str) -> str:
    # 提取原始答案
    answer_match = re.search(r'Answer:\s*([A-Z])', text)
    if answer_match:
        answer = answer_match.group(1)
    else:
        raise ValueError("No answer found in the text")
    
    # 移除原始 'Answer:' 部分
    text = re.sub(r'Answer:\s*[A-Z]', '', text).strip()
    
    # 格式化为目标格式
    formatted_text = f"<think>{text}</think> <answer>{answer}</answer>"
    
    return formatted_text

# 测试示例
# text = "To find the measure of angle B, we can use the property that the opposite angles formed by parallel lines are equal. Therefore, angle ADO is equal to angle ODC, which is equal to x/2. Angle ADC is the sum of angle ADO and angle ODC, so it is equal to x/2 + x/2 = x. Finally, angle B is equal to 180° - angle ADC, which is equal to 180° - x. Therefore, the degree measure of angle B is 180° - 50.0° = 130°. Hence, the answer is option D.\nAnswer:D"

# formatted_text = format_math_explanation(text)
# print(formatted_text)



import json
import pandas as pd
from datasets import Dataset, Features, Value, Image
from PIL import Image as PILImage
import io
import os
# import tqdm
from tqdm import tqdm
from PIL import Image as I

def has_valid_image_size_from_path(image_path):
    try:
        # Open the image from the path
        image = I.open(image_path)
        
        # Get the image dimensions
        width, height = image.size
        
        # Check if both dimensions are greater than or equal to 28
        return height >= 28 and width >= 28
    except Exception as e:
        # If there's an error opening the image (e.g., invalid file or path), return False
        print(f"Error opening image {image_path}: {e}")
        return False

def read_jsonl(file_path):
    """
    读取JSONL文件并返回一个包含所有项的字典列表。

    Args:
        file_path (str): JSONL文件的路径。

    Returns:
        list: 解析后的字典列表，每个字典代表JSONL文件中的一行。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  # 解析每一行并添加到列表中
    return data

# 读取k12.jsonl文件并解析
input_file = '/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/ReasoningData/geo170k/qa_tuning_20k.json'

data = {
    'image': [],
    'image_path': [],
    'problem': [],
    'solution': [],
    'original_question': [],
    'original_answer': []
}


with open(input_file,'r') as f:
    data_all = json.load(f)


data_all1 = []
for data_tmp in tqdm(data_all, desc="Processing images", unit="image"):
    image_path = os.path.join('/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/ReasoningData/geo170k/images',data_tmp['image'])  # 直接取出image路径或数据 
    is_valid = has_valid_image_size_from_path(image_path)
    if is_valid == True:
        data_all1.append(data_tmp)
    else:
        print(image_path)
     
print('len(data_all: ', len(data_all))   
print('len(data_all1): ', len(data_all1))


import random
random.seed(42)
data1 = data_all1

random.shuffle(data1)
for item in data1:

    # 假设item结构如下：item['image']、item['conversation'][0]['value']、item['conversation'][1]['value']
    image = os.path.join('/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/ReasoningData/geo170k/images',item.get('image'))  # 直接取出image路径或数据
    problem = item['question']
    solution = format_math_explanation(item['answer'])  # 第二个对话中的解决方案
    original_question = problem  # 假设原问题就是第一个对话的value
    original_answer = solution  # 假设原答案就是第二个对话的value

    # 填充数据
    data['image'].append(image)
    data['image_path'].append(image)
    data['problem'].append(problem)
    data['solution'].append(solution)
    data['original_question'].append(original_question)
    data['original_answer'].append(original_answer)

# 将数据转换为DataFrame
df = pd.DataFrame(data)



# 为了加载时能够正确解析图像，设置Feature
features = Features({
    'image': Image(),  # 将图像列指定为Image类型
    'image_path': Value('string'),
    'problem': Value('string'),
    'solution': Value('string'),
    'original_question': Value('string'),
    'original_answer': Value('string')
})

# 将训练集和测试集数据分别转换为Hugging Face的Dataset格式
train_dataset = Dataset.from_pandas(df, features=features)


# 保存为Parquet格式
train_save_path = "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/mengfanqing/open-r1-multimodal/k12data_20k_valid_fix_geo170k/train-00000-of-00001.parquet"


train_dataset.to_parquet(train_save_path)


print(f"训练集数据已保存到 {train_save_path}")
