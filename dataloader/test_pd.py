#%%
import pandas as pd
import json

# 假定JSON数据保存在data.json文件中
file_path = 'path_to_your_json_file.json'
file_path = "/storage/user/huju/transferred/ws_dditnach/DDIT/preprocess_output/experiment_1_class_alter_NptcUsdf_repro/preprocess.json"

# 加载JSON数据
with open(file_path, 'r') as file:
    data = json.load(file)

# 准备一个空列表来收集所有行的数据
rows = []
# 遍历JSON中的每个对象
for item in data:
    scene_name = item['scene_name']
    obj_id = item['id']
    category = item['category']
    neighbour_ids = item['neighbour_id']
    neighbour_ious = item['neighbour_iou']
    
    row = {
        'scene_name': scene_name,
        'id': obj_id,
        'category': category,
        'neighbour_ids': neighbour_ids,
        'neighbour_ious': neighbour_ious
    }
    rows.append(row)
# 使用rows列表创建DataFrame
df = pd.DataFrame(rows)

# 显示DataFrame的前几行
print(df.head())

# %%
