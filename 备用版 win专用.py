from pathlib import Path
import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import seaborn as sns

# Get the desktop path using the user's home directory
desktop = str(Path.home() / "Desktop")

# Find all CSV files on desktop and get the first one
csv_files = [f for f in os.listdir(desktop) if f.endswith('.csv')]
file_path = os.path.join(desktop, csv_files[0])

# Try to read the CSV file with different encodings (UTF-8 first, then GBK)
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except:
    df = pd.read_csv(file_path, encoding='gbk')

# Perform Chinese word segmentation on the 'summary' column
# Convert each document into space-separated words
df['cut'] = df['摘要'].apply(lambda x: ' '.join(jieba.lcut(str(x))))

# Initialize the CountVectorizer for converting text to term frequency matrix
vectorizer = CountVectorizer(max_df=0.95, min_df=2)

# Transform text data into term frequency matrix
tf = vectorizer.fit_transform(df['cut'])

# 计算不同主题数的困惑度
n_topics_range = range(2, 11)  # 从2到10个主题
perplexities = []

print("Calculating perplexity for different numbers of topics...")
for n_topics in n_topics_range:
    # 训练LDA模型
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=0,
        max_iter=30,  # 减少迭代次数以加快速度
        learning_method='online'
    )
    lda_model.fit(tf)

    # 计算困惑度
    perplexity = lda_model.perplexity(tf)
    perplexities.append(perplexity)
    print(f"Topics: {n_topics}: Perplexity = {perplexity:.2f}")

# 绘制困惑度曲线
plt.figure(figsize=(10, 6))
plt.plot(n_topics_range, perplexities, 'o-')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('LDA Model Perplexity vs Number of Topics')
plt.grid(True)
plt.savefig(os.path.join(desktop, 'perplexity_curve.png'))
plt.close()
print(f"Perplexity curve saved to: {os.path.join(desktop, 'perplexity_curve.png')}")

# 找到最佳主题数 (困惑度曲线拐点)
optimal_topics = n_topics_range[np.argmin(perplexities)]
print(f"Optimal number of topics based on perplexity: {optimal_topics}")

n_final_topics = optimal_topics  # 使用自动确定的最佳主题数

# 重新训练最终的LDA模型
print(f"Training final LDA model with {n_final_topics} topics...")
lda = LatentDirichletAllocation(
    n_components=n_final_topics,
    random_state=0,
    max_iter=50  # 更多迭代次数以获得更好的结果
)

# Transform documents to topic space
doc_topics = lda.fit_transform(tf)

# Get the vocabulary (all unique words)
words = vectorizer.get_feature_names_out()

# 创建主题之间的距离可视化 (使用t-SNE)
print("Generating t-SNE visualization...")

# 计算文档的主题分布特征向量
doc_topic_features = doc_topics

# 使用t-SNE将文档映射到二维空间
tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(doc_topics) - 1))
tsne_results = tsne.fit_transform(doc_topic_features)

# 为每个文档分配主要主题
doc_main_topic = np.argmax(doc_topics, axis=1)

# 创建可视化
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, n_final_topics))

for topic_idx in range(n_final_topics):
    # 选择属于当前主题的点
    indices = np.where(doc_main_topic == topic_idx)[0]
    if len(indices) > 0:  # 确保该主题有文档
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            c=[colors[topic_idx]],
            label=f'Topic {topic_idx + 1}',
            alpha=1
        )

plt.legend(title="Topics")
plt.title('t-SNE: Document Distribution in Topic Space')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig(os.path.join(desktop, 'topic_distribution_tsne.png'))
plt.close()
print(f"t-SNE visualization saved to: {os.path.join(desktop, 'topic_distribution_tsne.png')}")

# Create a results file for topic distribution
with open(os.path.join(desktop, 'topic_distribution.txt'), 'w', encoding='utf-8') as f:
    # 写入困惑度分析结果
    f.write("LDA Topic Model Perplexity Analysis\n")
    f.write("=" * 50 + "\n")
    f.write("Number of Topics\tPerplexity\n")
    for n, p in zip(n_topics_range, perplexities):
        f.write(f"{n}\t{p:.2f}\n")
    f.write("\n")
    f.write(f"Recommended optimal number of topics: {optimal_topics}\n")
    f.write(f"Actual number of topics used: {n_final_topics}\n\n")

    # 自动生成每个主题的名称
    topic_names = []

    # For each topic
    for topic_idx, topic in enumerate(lda.components_):
        # Get top words for this topic
        top_words_idx = topic.argsort()[:-31:-1]  # 获取权重最高的30个词的索引，想要更多可以改
        top_words = [words[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]

        # 生成主题名称 (使用前3个词)
        topic_name = "-".join(top_words[:3])
        topic_names.append(topic_name)

        f.write(f'\nTopic {topic_idx + 1}: {topic_name}\n')
        f.write('-' * 50 + '\n')
        f.write(f'Keywords: {", ".join(top_words)}\n\n')

        # 创建一个有实际内容的词典
        top_word_weights = {}
        for word, weight in zip(top_words, top_weights):
            if weight > 0:  # 确保权重为正
                top_word_weights[word] = weight

        print(f"Topic {topic_idx + 1} ({topic_name}) top 10 words: {top_words}")

        try:
            # 尝试使用固定字体并打印详细信息
            print("Trying to use simhei.ttf...")
            wc = WordCloud(
                font_path='C:/Windows/Fonts/simhei.ttf',
                width=800, height=400,
                background_color='white',
                max_words=100
            )

            if top_word_weights:
                print(f"Generating wordcloud with {len(top_word_weights)} words...")
                wc.generate_from_frequencies(top_word_weights)

                # 保存词云图片
                plt.figure(figsize=(10, 5))
                plt.imshow(wc)
                plt.axis('off')
                plt.title(f'Topic {topic_idx + 1}')  # 英文标题
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(desktop, f'topic_{topic_idx + 1}_wordcloud.png'))
                plt.close()
            else:
                print("Error: Not enough word weights")

        except Exception as e:
            print(f"Failed to generate wordcloud with simhei.ttf: {str(e)}")

            # 如果上面的方法失败，尝试不指定字体
            try:
                print("Trying with default font...")
                wc = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100
                )

                if top_word_weights:
                    wc.generate(' '.join(top_words))  # 直接使用词频最高的词生成词云

                    # 保存词云图片
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc)
                    plt.axis('off')
                    plt.title(f'Topic {topic_idx + 1}')  # 英文标题
                    plt.tight_layout(pad=0)
                    plt.savefig(os.path.join(desktop, f'topic_{topic_idx + 1}_wordcloud.png'))
                    plt.close()
                else:
                    print("Error: Not enough words")
            except Exception as e2:
                print(f"Also failed with default font: {str(e2)}")

        # Get documents for this topic
        doc_scores = doc_topics[:, topic_idx]
        doc_indices = np.where(doc_scores > 0.3)[0]  # Get documents with >30% probability

        # Write document information
        f.write(f'Number of documents in this topic: {len(doc_indices)}\n\n')

        # 只显示前5个文档示例
        for idx in doc_indices[:5]:
            try:
                f.write(f'Document {idx + 1} (Probability: {doc_scores[idx]:.3f}):\n')
                f.write(f'{df["摘要"].iloc[idx][:200]}...\n\n')  # Show first 200 characters
            except Exception as e:
                f.write(f'Error processing document {idx + 1}: {str(e)}\n\n')

    # 生成主题相关性热图
    topic_correlation = np.corrcoef(doc_topics, rowvar=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        topic_correlation,
        annot=True,
        cmap="YlGnBu",
        xticklabels=[f"Topic {i + 1}" for i in range(n_final_topics)],
        yticklabels=[f"Topic {i + 1}" for i in range(n_final_topics)]
    )
    plt.title('Topic Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(desktop, 'topic_correlation.png'))
    plt.close()

    # 在结果文件中记录生成的所有可视化文件
    f.write("\nGenerated Visualization Files:\n")
    f.write("-" * 50 + "\n")
    f.write(f"1. Perplexity Curve: perplexity_curve.png\n")
    f.write(f"2. Topic Distribution t-SNE: topic_distribution_tsne.png\n")
    f.write(f"3. Topic Correlation Heatmap: topic_correlation.png\n")
    for i in range(n_final_topics):
        f.write(f"{i + 4}. Topic {i + 1} Wordcloud: topic_{i + 1}_wordcloud.png\n")

print("LDA Topic Analysis Complete!")
print(f"Results saved to: {os.path.join(desktop, 'topic_distribution.txt')}")