import pandas as pd

# 读取CSV文件
file_path = 'dataset/electricity.csv'  # 请替换为你的CSV文件路径

# 使用pandas读取csv文件
df = pd.read_csv(file_path)

for n in {128, 512, 1024, 2048, 4096}:
    # 截取前n行，这里以10行为例，你可以根据需要修改这个数字
    output_path = F'dataset/test{n}.csv'  # 你想保存的新CSV文件路径

    df_subset = df.head(n)

    # 将截取的数据写入新的CSV文件
    df_subset.to_csv(output_path, index=False)

    print(f"已成功截取前{n}行并保存至{output_path}")
