import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import re
import jieba
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
import random
from collections import Counter, defaultdict
import threading
import time


# 内置简化版Word2Vec实现，不依赖gensim
class SimpleWord2Vec:
    def __init__(self, vector_size=100, window=5, min_count=5, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.word_vectors = {}
        self.key_to_index = {}
        self.vocab = {}

    def train(self, sentences):
        # 构建词汇表
        word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1

        # 过滤低频词
        self.vocab = {word: count for word, count in word_count.items() if count >= self.min_count}

        # 构建索引
        self.key_to_index = {word: i for i, word in enumerate(self.vocab.keys())}

        # 随机初始化词向量
        for word in self.vocab:
            self.word_vectors[word] = np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size,
                                                        self.vector_size)
            # 归一化
            norm = np.linalg.norm(self.word_vectors[word])
            if norm > 0:
                self.word_vectors[word] /= norm

        # 训练词向量
        for epoch in range(self.epochs):
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    if word not in self.vocab:
                        continue

                    # 定义上下文窗口
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)

                    # 收集上下文词
                    context_words = [sentence[j] for j in range(start, end) if j != i and sentence[j] in self.vocab]

                    # 简单更新词向量 (只是示例，实际Word2Vec更复杂)
                    for context_word in context_words:
                        # 更新词向量
                        self._update_vectors(word, context_word)

    def _update_vectors(self, word, context_word):
        # 简化版的词向量更新逻辑
        # 在真正的Word2Vec中，这里应该是基于负采样或层次softmax的更复杂更新
        lr = 0.01  # 学习率

        # 获取词向量
        word_vec = self.word_vectors[word]
        context_vec = self.word_vectors[context_word]

        # 计算相似度
        similarity = np.dot(word_vec, context_vec)

        # 简化梯度更新
        gradient = (1.0 - similarity) * lr

        # 更新两个词向量
        word_update = gradient * context_vec
        context_update = gradient * word_vec

        # 应用更新
        self.word_vectors[word] += word_update
        self.word_vectors[context_word] += context_update

        # 重新归一化
        for vec_name in [word, context_word]:
            vec = self.word_vectors[vec_name]
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.word_vectors[vec_name] = vec / norm

    def most_similar(self, word, topn=10):
        """找到与给定词最相似的词"""
        if word not in self.word_vectors:
            return []

        word_vec = self.word_vectors[word]
        similarities = []

        for other_word, other_vec in self.word_vectors.items():
            if other_word != word:
                similarity = np.dot(word_vec, other_vec)
                similarities.append((other_word, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


class Word2VecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Word2Vec 模型训练与应用")
        self.root.geometry("800x600")
        self.model = None
        self.data = None
        self.text_column = None

        # 创建UI组件
        self.create_widgets()

    def create_widgets(self):
        # 顶部菜单
        menubar = tk.Menu(self.root)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="加载CSV文件", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)

        # 模型菜单
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="训练模型", command=self.train_model)
        menubar.add_cascade(label="模型", menu=model_menu)

        # 功能菜单
        function_menu = tk.Menu(menubar, tearoff=0)
        function_menu.add_command(label="查找相似词", command=self.find_similar_words)
        function_menu.add_command(label="词向量可视化", command=self.visualize_vectors)
        menubar.add_cascade(label="功能", menu=function_menu)

        self.root.config(menu=menubar)

        # 主框架分为左右两部分
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧为控制面板
        self.control_frame = tk.LabelFrame(self.main_frame, text="控制面板")
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        # 状态标签
        self.status_label = tk.Label(self.control_frame, text="状态: 未加载数据", anchor="w")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

        # 模型参数框架
        self.params_frame = tk.LabelFrame(self.control_frame, text="模型参数")
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)

        # 词向量维度
        tk.Label(self.params_frame, text="词向量维度:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.vector_size_var = tk.StringVar(value="100")
        tk.Entry(self.params_frame, textvariable=self.vector_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # 窗口大小
        tk.Label(self.params_frame, text="窗口大小:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.window_var = tk.StringVar(value="5")
        tk.Entry(self.params_frame, textvariable=self.window_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # 最小计数
        tk.Label(self.params_frame, text="最小词频:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.min_count_var = tk.StringVar(value="5")
        tk.Entry(self.params_frame, textvariable=self.min_count_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # 训练轮数
        tk.Label(self.params_frame, text="训练轮数:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="5")
        tk.Entry(self.params_frame, textvariable=self.epochs_var, width=10).grid(row=3, column=1, padx=5, pady=2)

        # 右侧为结果显示区
        self.result_frame = tk.LabelFrame(self.main_frame, text="结果显示")
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 文本框用于显示结果
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加滚动条
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)

        # 显示欢迎信息
        self.result_text.insert(tk.END, "欢迎使用Word2Vec模型训练与应用程序!\n\n请从'文件'菜单加载CSV文件开始。")

    def load_csv(self):
        # 打开文件对话框，选择桌面上的CSV文件
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path = filedialog.askopenfilename(
            initialdir=desktop_path,
            title="选择CSV文件",
            filetypes=(("CSV文件", "*.csv"), ("所有文件", "*.*"))
        )

        if file_path:
            try:
                # 读取CSV文件
                self.data = pd.read_csv(file_path)
                self.status_label.config(text=f"状态: 已加载数据 ({file_path.split('/')[-1]})")

                # 显示数据预览
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "数据预览:\n")
                self.result_text.insert(tk.END, f"行数: {len(self.data)}\n")
                self.result_text.insert(tk.END, f"列数: {len(self.data.columns)}\n\n")
                self.result_text.insert(tk.END, "前5行数据:\n")
                self.result_text.insert(tk.END, str(self.data.head()))

                # 询问用户哪一列包含文本数据
                column_names = list(self.data.columns)
                self.text_column = simpledialog.askstring(
                    "选择文本列",
                    f"请输入包含文本数据的列名:\n可用列: {', '.join(column_names)}",
                    parent=self.root
                )

                if self.text_column not in column_names:
                    messagebox.showerror("错误", f"列名 '{self.text_column}' 不存在!")
                    return

                self.result_text.insert(tk.END, f"\n\n已选择文本列: {self.text_column}\n")

            except Exception as e:
                messagebox.showerror("错误", f"加载CSV文件时出错:\n{str(e)}")

    def preprocess_text(self, texts):
        # 对文本进行预处理
        processed_texts = []
        for text in texts:
            if isinstance(text, str):
                # 去除特殊字符和数字
                text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
                # 使用jieba分词
                words = jieba.lcut(text)
                # 过滤空字符
                words = [word for word in words if word.strip()]
                processed_texts.append(words)

        return processed_texts

    def train_model(self):
        if self.data is None or self.text_column is None:
            messagebox.showerror("错误", "请先加载数据并选择文本列!")
            return

        try:
            # 获取文本数据
            texts = self.data[self.text_column].fillna("").tolist()

            # 预处理文本
            processed_texts = self.preprocess_text(texts)

            # 获取模型参数
            vector_size = int(self.vector_size_var.get())
            window = int(self.window_var.get())
            min_count = int(self.min_count_var.get())
            epochs = int(self.epochs_var.get())

            # 训练Word2Vec模型
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "开始训练模型...\n")
            self.root.update()

            # 使用我们自己的简化版Word2Vec实现
            self.model = SimpleWord2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                epochs=epochs
            )

            # 开始训练
            self.model.train(processed_texts)

            # 显示训练结果
            self.result_text.insert(tk.END, "模型训练完成!\n")
            self.result_text.insert(tk.END, f"词汇量: {len(self.model.vocab)}\n")
            self.result_text.insert(tk.END, f"词向量维度: {self.model.vector_size}\n\n")

            # 显示一些常见词的向量示例
            self.result_text.insert(tk.END, "部分词汇示例:\n")
            for i, word in enumerate(list(self.model.vocab.keys())[:10]):
                self.result_text.insert(tk.END, f"{i + 1}. {word}\n")

            self.status_label.config(text="状态: 模型训练完成")

        except Exception as e:
            messagebox.showerror("错误", f"训练模型时出错:\n{str(e)}")

    def find_similar_words(self):
        if self.model is None:
            messagebox.showerror("错误", "请先训练模型!")
            return

        # 询问用户输入词语
        word = simpledialog.askstring(
            "查找相似词",
            "请输入要查找的词语:",
            parent=self.root
        )

        if not word:
            return

        try:
            # 检查词语是否在词汇表中
            if word not in self.model.vocab:
                messagebox.showerror("错误", f"词语 '{word}' 不在词汇表中!")
                return

            # 询问用户需要返回多少个相似词
            num_similar = simpledialog.askinteger(
                "相似词数量",
                "请输入需要返回的相似词数量:",
                parent=self.root,
                minvalue=1,
                maxvalue=100
            )

            if not num_similar:
                return

            # 获取相似词
            similar_words = self.model.most_similar(word, topn=num_similar)

            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"与 '{word}' 最相似的 {num_similar} 个词:\n\n")

            for i, (similar_word, similarity) in enumerate(similar_words):
                self.result_text.insert(tk.END, f"{i + 1}. {similar_word} (相似度: {similarity:.4f})\n")

        except Exception as e:
            messagebox.showerror("错误", f"查找相似词时出错:\n{str(e)}")

    def visualize_vectors(self):
        if self.model is None:
            messagebox.showerror("错误", "请先训练模型!")
            return

        # 询问用户需要可视化多少个词
        num_words = simpledialog.askinteger(
            "词向量可视化",
            "请输入需要可视化的词语数量 (建议 10-30):",
            parent=self.root,
            minvalue=5,
            maxvalue=100
        )

        if not num_words:
            return

        try:
            # 获取词频最高的n个词
            words = list(self.model.vocab.keys())[:num_words]

            # 获取这些词的向量
            word_vectors = np.array([self.model.word_vectors[word] for word in words])

            # 使用PCA降维到2维
            pca = PCA(n_components=2)
            result = pca.fit_transform(word_vectors)

            # 创建可视化窗口
            viz_window = tk.Toplevel(self.root)
            viz_window.title("词向量可视化")
            viz_window.geometry("800x600")

            # 创建matplotlib图形
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(result[:, 0], result[:, 1], alpha=0.7)

            # 添加词语标注
            for i, word in enumerate(words):
                ax.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=9)

            ax.set_title(f"Word2Vec词向量的二维PCA投影 (前{num_words}个词)")
            ax.set_xlabel("主成分1")
            ax.set_ylabel("主成分2")
            ax.grid(True, linestyle='--', alpha=0.7)

            # 将matplotlib图形嵌入到tkinter窗口
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("错误", f"可视化词向量时出错:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = Word2VecApp(root)
    root.mainloop()