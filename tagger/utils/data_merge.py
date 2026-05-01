import json
with open("/root/test_prompts (7).json", 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    # 2. 读取第二个JSON文件
with open("/root/train_dataset (9).json", 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)
merged_data = data1 + data2

    # 3. 将合并后的数据写入新的JSON文件
with open("/root/train_dataset_all.json", 'w', encoding='utf-8') as out_f:
        # indent=4 让输出的JSON格式化，更易读
        # ensure_ascii=False 确保中文字符能正常显示而不是被转义
        json.dump(merged_data, out_f, indent=4, ensure_ascii=False)