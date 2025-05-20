import os
import glob
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import datetime
import re
import csv

# 安装必要的库
# pip install bertopic sentence-transformers transformers pillow

# 设置输出路径 - 保存到桌面
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
print(f"结果将保存到: {desktop_path}")

# 创建一个时间戳文件夹，避免覆盖之前的结果
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(desktop_path, f"image_classification_{timestamp}")
os.makedirs(output_folder, exist_ok=True)
print(f"创建结果文件夹: {output_folder}")

# 设置图像文件夹 - 使用原始字符串表示法
image_folder = r"C:\Users\misty\Desktop\.idea\twitter_download-main\#"  # 替换为你的图像文件夹路径

# 支持多种图像格式
images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif']:
    images.extend(glob.glob(os.path.join(image_folder, ext)))

print(f"找到 {len(images)} 张图片")

# 如果没有找到图像，检查路径是否存在
if len(images) == 0:
    if os.path.exists(image_folder):
        print(f"文件夹存在，但未找到图像文件。请检查图像扩展名。")
        # 显示文件夹中的所有文件
        print("文件夹内容:")
        for file in os.listdir(image_folder):
            print(f" - {file}")
    else:
        print(f"文件夹不存在，请检查路径: {image_folder}")
    exit()

# 新增：询问是否要结合CSV文本数据
use_csv = input("是否需要结合CSV文本数据？(y/n): ").strip().lower() == 'y'
text_data = {}
csv_columns = []

if use_csv:
    # 请求CSV文件路径
    csv_path = input("请输入CSV文件路径: ").strip()

    # 验证CSV文件存在
    if not os.path.exists(csv_path):
        print(f"CSV文件不存在: {csv_path}")
        use_csv = False
    else:
        # 读取CSV文件
        try:
            csv_df = pd.read_csv(csv_path)
            print(f"成功读取CSV文件，包含 {len(csv_df)} 条记录和 {len(csv_df.columns)} 列")

            # 显示列名并让用户选择需要的列
            print("CSV文件包含以下列:")
            for i, column in enumerate(csv_df.columns):
                print(f"{i + 1}. {column}")

            # 询问哪一列包含图像文件名或路径
            img_col_idx = int(input("请输入包含图像文件名或路径的列编号: ")) - 1
            img_col_name = csv_df.columns[img_col_idx]

            # 询问哪些列包含要分析的文本
            text_cols_input = input("请输入包含文本数据的列编号(多个用逗号分隔): ")
            text_cols_idx = [int(x.strip()) - 1 for x in text_cols_input.split(",")]
            text_cols_names = [csv_df.columns[idx] for idx in text_cols_idx]

            # 保存选择的列名
            csv_columns = [img_col_name] + text_cols_names

            # 构建文本数据字典
            print("正在处理CSV数据...")
            for _, row in tqdm(csv_df.iterrows(), total=len(csv_df), desc="处理CSV数据"):
                img_identifier = row[img_col_name]

                # 处理文件名还是完整路径
                if os.path.isabs(str(img_identifier)):
                    # 是完整路径，提取文件名
                    img_key = os.path.basename(img_identifier)
                else:
                    # 只是文件名
                    img_key = img_identifier

                # 合并所有文本列
                combined_text = " ".join([str(row[col]) for col in text_cols_names if pd.notna(row[col])])

                # 存储文本数据
                text_data[img_key] = combined_text.strip()

            print(f"成功处理 {len(text_data)} 条图像-文本关联数据")
        except Exception as e:
            print(f"处理CSV文件时出错: {e}")
            use_csv = False

# 开始处理图像
print("开始图像分类过程...")

try:
    # 导入需要的库
    from bertopic import BERTopic
    from bertopic.representation import VisualRepresentation, KeyBERTInspired
    from bertopic.backend import MultiModalBackend

    print("初始化CLIP模型...")
    # 使用MultiModalBackend，但改用更小的批量大小
    embedding_model = MultiModalBackend('clip-ViT-B-32', batch_size=4)

    print("初始化图像到文本的表示模型...")
    # 创建表示模型 - 区分是否使用CSV文本
    if use_csv and text_data:
        # 结合图像和文本的表示
        representation_model = {
            "Visual_Aspect": VisualRepresentation(
                image_to_text_model="nlpconnect/vit-gpt2-image-captioning"
            ),
            "Text_Content": KeyBERTInspired()
        }
    else:
        # 仅使用图像表示
        representation_model = {
            "Visual_Aspect": VisualRepresentation(
                image_to_text_model="nlpconnect/vit-gpt2-image-captioning"
            )
        }

    print("创建BERTopic模型...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        representation_model=representation_model,
        min_topic_size=5,  # 最小主题大小
        verbose=True
    )

    # 准备文档和图像的映射
    documents = None
    if use_csv and text_data:
        # 创建与图像对应的文本文档列表
        documents = []
        image_paths = []

        for img_path in images:
            img_filename = os.path.basename(img_path)
            # 尝试找到匹配的文本
            if img_filename in text_data:
                documents.append(text_data[img_filename])
                image_paths.append(img_path)
            else:
                # 尝试不带扩展名匹配
                name_without_ext = os.path.splitext(img_filename)[0]
                found = False
                for text_key in text_data:
                    if name_without_ext in text_key or text_key in name_without_ext:
                        documents.append(text_data[text_key])
                        image_paths.append(img_path)
                        found = True
                        break

                # 如果仍未找到匹配，使用空文本
                if not found:
                    documents.append("")
                    image_paths.append(img_path)

        # 更新图像列表为成功匹配的图像
        if len(image_paths) > 0:
            images = image_paths
            print(f"成功匹配 {len(documents)} 个图像-文本对")

    print("开始训练模型，这可能需要一些时间...")
    topics, probs = topic_model.fit_transform(documents=documents, images=images)
    print("模型训练完成！")


    # 修复概率处理 - 正确处理不同格式的概率值
    def safe_get_probability(prob):
        """安全地处理概率值，兼容不同的返回格式"""
        try:
            # 如果是列表或数组，返回最大值
            if hasattr(prob, '__iter__') and not isinstance(prob, (str, bytes)):
                return max(prob)
            # 如果是单个值（如numpy.float64），直接返回
            return float(prob)
        except Exception as e:
            # 如果出现任何错误，返回0
            print(f"处理概率值时出错: {e}, 值: {type(prob)}")
            return 0.0


    # 创建结果CSV - 使用安全的概率处理函数
    probability_values = [safe_get_probability(prob) for prob in probs]

    # 基础结果数据
    results_data = {
        'image_path': images,
        'topic': topics,
        'probability': probability_values
    }

    # 如果使用了CSV文本数据，添加文本列
    if use_csv and documents:
        results_data['text_content'] = documents

    results_df = pd.DataFrame(results_data)

    results_path = os.path.join(output_folder, "image_classification_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"分类结果已保存到: {results_path}")

    # 保存模型
    model_path = os.path.join(output_folder, "image_topic_model")
    topic_model.save(model_path)
    print(f"模型已保存到: {model_path}")

    # 获取主题信息并保存
    topic_info = topic_model.get_topic_info()
    print("\n主题分布:")
    print(topic_info)

    # 将主题信息保存到CSV
    topic_info_path = os.path.join(output_folder, "topic_info.csv")
    topic_info.to_csv(topic_info_path, index=False)
    print(f"主题信息已保存到: {topic_info_path}")

    # 保存各主题的图像清单
    topic_images = {}
    for i, topic in enumerate(topics):
        if topic not in topic_images:
            topic_images[topic] = []
        topic_images[topic].append(images[i])

    # 为每个主题创建一个图像列表文件，同时也保存关联的文本(如果有)
    for topic, img_list in topic_images.items():
        if topic != -1:  # 跳过异常值主题
            topic_file = os.path.join(output_folder, f"topic_{topic}_images.txt")

            # 如果有文本数据，创建一个包含文本的CSV文件
            if use_csv and documents:
                topic_csv = os.path.join(output_folder, f"topic_{topic}_images_text.csv")
                with open(topic_csv, "w", newline='', encoding="utf-8") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    # 写入标题行
                    csvwriter.writerow(["image_path", "text_content"])

                    # 为该主题的每张图片写入数据
                    for img_path in img_list:
                        img_idx = images.index(img_path)
                        text_content = documents[img_idx] if img_idx < len(documents) else ""
                        csvwriter.writerow([img_path, text_content])

                print(f"主题 {topic} 的图像和文本已保存到: {topic_csv}")

            # 始终创建纯图像路径列表
            with open(topic_file, "w", encoding="utf-8") as f:
                for img in img_list:
                    f.write(f"{img}\n")
            print(f"主题 {topic} 的图像列表已保存到: {topic_file}")

    # 尝试创建可视化（如果可能）
    try:
        print("创建主题可视化...")
        # 可视化主题分布
        fig = topic_model.visualize_topics()
        viz_path = os.path.join(output_folder, "topic_visualization.html")
        fig.write_html(viz_path)
        print(f"主题可视化已保存到: {viz_path}")

        # 可视化主题层次结构
        fig = topic_model.visualize_hierarchy()
        hierarchy_path = os.path.join(output_folder, "topic_hierarchy.html")
        fig.write_html(hierarchy_path)
        print(f"主题层次结构已保存到: {hierarchy_path}")

        # 如果有文本数据，尝试创建主题-文档可视化
        if use_csv and documents:
            try:
                fig = topic_model.visualize_documents(images)
                doc_viz_path = os.path.join(output_folder, "document_visualization.html")
                fig.write_html(doc_viz_path)
                print(f"文档可视化已保存到: {doc_viz_path}")
            except Exception as e:
                print(f"创建文档可视化时出错: {e}")
    except Exception as e:
        print(f"创建可视化时出错: {e}")

    # 创建一个简单的README文件，解释结果
    readme_path = os.path.join(output_folder, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("BERTopic图像分类结果\n")
        f.write("==================\n\n")
        f.write(f"分类时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"图像文件夹: {image_folder}\n")
        f.write(f"图像总数: {len(images)}\n")
        if use_csv:
            f.write(f"使用的CSV文件: {csv_path}\n")
            f.write(f"CSV列: {', '.join(csv_columns)}\n")
        f.write("\n主题分布:\n")
        for index, row in topic_info.iterrows():
            if row['Topic'] != -1:
                f.write(f"主题 {row['Topic']}: {row['Count']} 张图像\n")
        f.write("\n文件说明:\n")
        f.write("- image_classification_results.csv: 每张图像的分类结果\n")
        f.write("- topic_info.csv: 主题统计信息\n")
        f.write("- topic_X_images.txt: 每个主题包含的图像列表\n")
        if use_csv:
            f.write("- topic_X_images_text.csv: 每个主题的图像和相关文本\n")
        f.write("- topic_visualization.html: 主题可视化 (如果可用)\n")
        f.write("- topic_hierarchy.html: 主题层次结构 (如果可用)\n")
        if use_csv:
            f.write("- document_visualization.html: 文档可视化 (如果可用)\n")
        f.write("- topic_folders/: 包含按主题分类的图像副本\n")

    # 创建主题文件夹并复制图像
    print("\n开始创建主题文件夹并复制图像...")

    # 创建主题文件夹主目录
    topic_folders_dir = os.path.join(output_folder, "topic_folders")
    os.makedirs(topic_folders_dir, exist_ok=True)

    # 计算需要复制的图像总数
    total_images_to_copy = sum(len(img_list) for topic, img_list in topic_images.items() if topic != -1)
    print(f"将复制 {total_images_to_copy} 张图像到各自的主题文件夹")

    # 使用tqdm添加进度条
    with tqdm(total=total_images_to_copy, desc="复制图像") as pbar:
        # 为每个主题创建文件夹并复制图像
        for topic, img_list in topic_images.items():
            if topic != -1:  # 跳过异常值主题
                # 创建主题文件夹
                topic_dir = os.path.join(topic_folders_dir, f"Topic_{topic}")
                os.makedirs(topic_dir, exist_ok=True)

                # 获取主题关键词 (如果可用)
                topic_keywords = ""
                try:
                    topic_words = topic_model.get_topic(topic)
                    if topic_words:
                        # 提取前3个关键词
                        keywords = [word for word, _ in topic_words[:3]]
                        topic_keywords = "_".join(keywords)
                except:
                    pass

                # 如果有关键词，创建一个带关键词的信息文件
                if topic_keywords:
                    with open(os.path.join(topic_dir, "keywords.txt"), "w", encoding="utf-8") as f:
                        f.write(f"主题 {topic} 的关键词:\n")
                        for word, score in topic_model.get_topic(topic)[:10]:  # 前10个关键词
                            f.write(f"{word}: {score:.4f}\n")

                # 如果有CSV文本数据，为这个主题创建一个包含文本的信息文件
                if use_csv and documents:
                    text_info_path = os.path.join(topic_dir, "text_content.csv")
                    with open(text_info_path, "w", newline='', encoding="utf-8") as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(["image_filename", "text_content"])

                        for img_path in img_list:
                            img_filename = os.path.basename(img_path)
                            img_idx = images.index(img_path)
                            text_content = documents[img_idx] if img_idx < len(documents) else ""
                            csvwriter.writerow([img_filename, text_content])

                # 复制图像到主题文件夹
                for i, img_path in enumerate(img_list):
                    # 获取原始文件名
                    img_filename = os.path.basename(img_path)
                    # 创建目标路径
                    dest_path = os.path.join(topic_dir, img_filename)

                    # 如果文件名已存在，添加序号避免覆盖
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(img_filename)
                        dest_path = os.path.join(topic_dir, f"{name}_{i}{ext}")

                    # 复制图像
                    try:
                        shutil.copy2(img_path, dest_path)
                        pbar.update(1)  # 更新进度条
                    except Exception as e:
                        print(f"复制图像时出错: {e} - {img_path}")

    print(f"\n所有图像已成功复制到各自的主题文件夹: {topic_folders_dir}")
    print(f"\n所有结果已保存到桌面文件夹: {output_folder}")
    print(f"README文件包含结果说明: {readme_path}")

except Exception as e:
    print(f"处理过程中出现错误: {e}")
    import traceback

    traceback.print_exc()