from pathlib import Path
import os
import pandas as pd
import jieba
import numpy as np
import matplotlib.pyplot as plt
from snownlp import SnowNLP
import warnings

warnings.filterwarnings('ignore')

# 获取桌面路径
desktop = str(Path.home() / "Desktop")

# 在桌面上寻找所有CSV文件并获取第一个
csv_files = [f for f in os.listdir(desktop) if f.endswith('.csv') and f != '情感分析结果.csv']
if not csv_files:
    print("桌面上没有找到CSV文件（除了情感分析结果.csv）")
    exit(1)

file_path = os.path.join(desktop, csv_files[0])
print(f"正在读取文件: {file_path}")

# 尝试使用不同的编码读取CSV文件（先尝试UTF-8，然后是GBK）
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except:
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except:
        print("无法读取CSV文件，请检查文件格式")
        exit(1)

print(f"已加载数据框，共{len(df)}行")
print(f"列名: {df.columns.tolist()}")

# 情感词典初始化
sentiment_words = {
    'positive': set(),
    'negative': set(),
    'degrees': {}  # 程度词及其权重
}

# 否定词，用于情感反转
negation_words = set([
    '不', '没', '无', '非', '莫', '弗', '毋', '勿', '未', '否',
    '别', '反', '难', '禁', '还是', '尚未', '并未', '并没', '不曾', '不太',
    '不怎么', '不很', '从不', '绝不', '几乎不', '从未', '尚无', '并非', '决非'
])
# 寻找情感词汇本体
print("寻找情感词汇本体...")
sentiment_lexicon_path = os.path.join(desktop, "情感词汇本体.xlsx")
if not os.path.exists(sentiment_lexicon_path):
    # 在整个桌面搜索可能的文件名
    for file in os.listdir(desktop):
        if file.endswith('.xlsx') and ('情感' in file or '词汇' in file or '本体' in file):
            sentiment_lexicon_path = os.path.join(desktop, file)
            print(f"找到可能的情感词汇本体文件: {file}")
            break

# 尝试读取Excel文件
try:
    import openpyxl
    print(f"尝试加载情感词汇本体: {sentiment_lexicon_path}")
    if os.path.exists(sentiment_lexicon_path):
        wb = openpyxl.load_workbook(sentiment_lexicon_path, read_only=True)

        # 遍历所有工作表
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            print(f"处理工作表: {sheet_name}")

            # 获取列标题
            header_row = next(sheet.rows)
            headers = [cell.value for cell in header_row]

            # 查找词语和情感极性的列
            word_col = None
            pos_col = None  # 词性列
            emotion_col = None  # 情感分类列
            intensity_col = None  # 强度列
            polarity_col = None  # 极性列

            for i, header in enumerate(headers):
                if header and isinstance(header, str):
                    header_lower = header.lower()
                    if '词语' in header_lower:
                        word_col = i
                    elif '词性' in header_lower:
                        pos_col = i
                    elif '情感分类' in header_lower:
                        emotion_col = i
                    elif '强度' in header_lower:
                        intensity_col = i
                    elif '极性' in header_lower:
                        polarity_col = i

            if word_col is not None:
                # 读取数据行
                for row in list(sheet.rows)[1:]:  # 跳过标题行
                    cells = [cell.value for cell in row]
                    if len(cells) <= max(filter(None, [word_col, emotion_col, intensity_col, polarity_col])):
                        continue

                    word = cells[word_col]
                    if not word:
                        continue

                    # 基于您提供的数据格式进行解析
                    # 假设极性为2是负面，1是正面
                    if polarity_col is not None and cells[polarity_col]:
                        polarity = str(cells[polarity_col]).strip()

                        if polarity == '1':
                            sentiment_words['positive'].add(word)
                        elif polarity == '2':
                            sentiment_words['negative'].add(word)

                    # 处理强度信息
                    if intensity_col is not None and cells[intensity_col]:
                        intensity = str(cells[intensity_col]).strip()
                        if intensity.isdigit():
                            intensity_value = float(intensity) / 9.0  # 归一化强度（假设最大是9）
                            if word in sentiment_words['positive'] or word in sentiment_words['negative']:
                                sentiment_words['degrees'][word] = intensity_value
    else:
        print(f"情感词汇本体文件 {sentiment_lexicon_path} 不存在")
except Exception as e:
    print(f"加载Excel情感词汇本体时出错: {e}")

# 如果没有从Excel加载数据，则解析您提供的样例数据
if len(sentiment_words['positive']) == 0 and len(sentiment_words['negative']) == 0:
    print("使用样例数据...")

    # 从您提供的样例数据中解析
    sample_data = """
    词语 词性种类 词义数 词义序号 情感分类 强度 极性 辅助情感分类 强度 极性
    脏乱 adj 1 1 NN 7 2
    糟报 adj 1 1 NN 5 2
    早衰 adj 1 1 NE 5 2
    责备 verb 1 1 NN 5 2
    贼眼 noun 1 1 NN 5 2
    战祸 noun 1 1 ND 5 2 NC 5 2
    招灾 adj 1 1 NN 5 2
    折辱 noun 1 1 NE 5 2 NN 5 2
    中山狼 noun 1 1 NN 5 2
    """

    lines = sample_data.strip().split('\n')
    headers = lines[0].split()

    # 查找列索引
    word_index = headers.index('词语')
    intensity_index = headers.index('强度')
    polarity_index = headers.index('极性')

    # 解析行
    for line in lines[1:]:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) <= polarity_index:
            continue

        word = parts[word_index]
        try:
            intensity = parts[intensity_index]
            polarity = parts[polarity_index]

            # 极性为1是正面，为2是负面
            if polarity == '1':
                sentiment_words['positive'].add(word)
            elif polarity == '2':
                sentiment_words['negative'].add(word)

            # 处理强度
            intensity_value = float(intensity) / 9.0  # 归一化强度
            sentiment_words['degrees'][word] = intensity_value
        except (IndexError, ValueError) as e:
            # 跳过无法解析的行
            continue
# 添加样例数据
print("添加基本情感词汇...")
extra_positive = [
    '优秀', '良好', '满意', '高效', '创新', '舒适', '成功', '美好', '快乐', '幸福',
    '积极', '优化', '完美', '卓越', '精彩', '健康', '便捷', '专业', '热情', '诚信',
    '喜爱', '赞赏', '优质', '满足', '进步', '准确', '实用', '合理', '方便', '真实',
    '丰富', '顺利', '高质量', '优惠', '贴心', '安全', '创意', '和谐', '尊重', '信任',
    '效率', '突出', '惊喜', '愉快', '值得', '希望', '机会', '感谢', '认可', '好评'
]

extra_negative = [
    '糟糕', '失败', '问题', '缺陷', '劣质', '低效', '缺乏', '不足', '错误', '不满',
    '差劲', '担忧', '忧虑', '难过', '悲伤', '痛苦', '失望', '遗憾', '焦虑', '愤怒',
    '恼火', '抱怨', '指责', '批评', '责难', '不良', '消极', '拖延', '混乱', '危险',
    '困难', '弱点', '威胁', '损失', '风险', '挫折', '崩溃', '破坏', '打扰', '麻烦',
    '无法', '无助', '无奈', '无效', '无用', '不当', '不利', '不佳', '不好', '傻逼'
]

# 添加到词典
sentiment_words['positive'].update(extra_positive)
sentiment_words['negative'].update(extra_negative)

print(f"加载了 {len(sentiment_words['positive'])} 个正面词, {len(sentiment_words['negative'])} 个负面词")

# 执行中文分词
text_column = None
for col in df.columns:
    if df[col].dtype == object:  # 找第一个文本列
        text_column = col
        break

if not text_column:
    print("找不到适合分析的文本列")
    exit(1)

print(f"使用列 '{text_column}' 进行分析")
df['cut'] = df[text_column].apply(lambda x: jieba.lcut(str(x)))

# 增强的情感分析函数 - 结合SnowNLP和情感词汇本体
print("准备情感分析...")

def enhanced_sentiment_analysis(text):
    """
    结合SnowNLP和情感词汇本体的增强情感分析函数
    返回0-1之间的得分，0表示极度负面，1表示极度正面
    """
    try:
        # 基本SnowNLP情感分析
        snow_score = SnowNLP(str(text)).sentiments

        # 使用情感词汇本体进行自定义分析
        text = str(text)
        words = jieba.lcut(text)

        # 初始化
        sentiment_score = 0
        negation_count = 0  # 否定词计数
        degree_multiplier = 1.0  # 程度副词乘数
        word_count = 0  # 计算有情感的词汇数量

        for i, word in enumerate(words):
            # 检查是否为程度副词
            if word in sentiment_words['degrees']:
                degree_multiplier = sentiment_words['degrees'].get(word, 1.0)
                continue

            # 检查是否为否定词
            if word in negation_words:
                negation_count += 1
                continue

            # 检查词语情感并使用强度信息
            word_score = 0.5
            intensity = 1  # 默认强度

            if word in sentiment_words['positive']:
                # 使用词汇特定的强度，如果有的话
                if word in sentiment_words['degrees']:
                    intensity = sentiment_words['degrees'].get(word, 1.0)
                word_score = 1 * intensity
                word_count += 1
            elif word in sentiment_words['negative']:
                if word in sentiment_words['degrees']:
                    intensity = sentiment_words['degrees'].get(word, 1.0)
                word_score = -1 * intensity
                word_count += 1

            # 应用程度乘数（用于前一个词为程度副词的情况）
            word_score *= degree_multiplier
            degree_multiplier = 1.0  # 重置乘数

            # 应用否定词效果（奇数个否定词会反转情感）
            if negation_count % 2 == 1:
                word_score *= -1

            negation_count = 0  # 重置否定词计数
            sentiment_score += word_score

        # 归一化得分到0-1范围
        if word_count > 0:
            lexicon_score = (sentiment_score / (word_count * 2) + 0.5)
        elif words:
            lexicon_score = (sentiment_score / (len(words) + 1) + 1) / 2


        lexicon_score = max(0, min(1, lexicon_score))  # 确保在0-1范围内

        # 融合两种得分
        snow_weight = 0.9
        lexicon_weight = 0.1
        final_score = snow_weight * snow_score + lexicon_weight * lexicon_score

        return final_score
    except Exception as e:
        print(f"情感分析出错: {e}")
        return 0.5  # 出错时返回中性情感





print("执行情感分析...")
df['sentiment_score'] = df[text_column].apply(enhanced_sentiment_analysis)

# 添加索引列作为X轴值
df['index'] = range(1, len(df) + 1)

# 创建散点图 - 使用从红到绿的颜色映射，恢复原有的格式
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df['index'],
    df['sentiment_score'],
    alpha=1,
    c=df['sentiment_score'],
    cmap='RdYlGn',
    s=50,
    edgecolors='k',
    linewidths=1
)

plt.xlabel('Document Index', fontsize=12)
plt.ylabel('Sentiment Score (0-1)', fontsize=12)
plt.title('Document Sentiment Score Distribution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Sentiment Score')
plt.tight_layout()
plt.savefig(os.path.join(desktop, 'sentiment_score_scatter.png'), dpi=300)
plt.close()
print(f"情感得分散点图已保存至: {os.path.join(desktop, 'sentiment_score_scatter.png')}")
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=50)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Score Distribution')
plt.savefig(os.path.join(desktop, 'score_distribution.png'))
# 创建包含评分的新CSV文件
print("生成包含情感评分的CSV文件...")
output_df = df.copy()

# 仅保留情感得分
output_df['情感得分'] = output_df['sentiment_score']

# 保留有用的列，删除中间处理列
columns_to_keep = [col for col in output_df.columns if
                   col not in ['cut', 'sentiment_score', 'index']]
result_df = output_df[columns_to_keep]

# 保存结果到新的CSV文件
output_csv_path = os.path.join(desktop, '情感分析结果.csv')
result_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)
print(f"包含情感分析结果的CSV文件已保存至: {output_csv_path}")

print("中文文本情感分析已完成!")