import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from scipy.linalg import sqrtm
import os
import time
import csv

# --- 配置类 ---
class Config:
    """全局配置，管理数据路径、模型参数等"""
    DATA_DIR = './lcqmc/'  # 数据目录
    MODEL_SAVE_PATH = './finetuned_bert_chinese'  # 微调模型保存路径
    USE_SYNTHETIC_DATA = False  # 是否使用合成数据
    DATA_SAMPLE_SIZE = None  # 数据采样大小（None 表示全量）
    FINETUNE_EPOCHS = 5  # 微调轮数
    FEATURE_BATCH_SIZE = 64  # 特征提取批大小
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
    MAX_LENGTH = 128  # 最大序列长度
    LOG_DIR = './logs'  # 日志目录

# --- 自定义数据集 ---
class SentencePairDataset(Dataset):
    """处理句对数据集，适配不同列名，生成 BERT 输入"""
    def __init__(self, data, tokenizer, max_length=Config.MAX_LENGTH):
        self.data = data.reset_index(drop=True)  # 重置索引
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 动态确定列名（假设前两列为句子，最后一列为标签）
        if len(data.columns) != 3:
            raise ValueError(f"数据集需包含 3 列（两个句子和标签），当前有 {len(data.columns)} 列")
        self.sent1_col = data.columns[0]
        self.sent2_col = data.columns[1]
        self.label_col = data.columns[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent1 = str(self.data.iloc[idx][self.sent1_col])
        sent2 = str(self.data.iloc[idx][self.sent2_col])
        encoding = self.tokenizer(
            sent1, sent2,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),  # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0)  # [max_length]
        }
        if self.label_col in self.data.columns:
            item['labels'] = torch.tensor(int(self.data.iloc[idx][self.label_col]), dtype=torch.long)
        return item

# --- 数据准备 ---
def prepare_data(config=Config):
    """加载或生成数据集，处理 CSV 格式，支持动态分隔符和列名"""
    print("正在准备数据...")
    raw_data_file = os.path.join(config.DATA_DIR, 'LCQMC_train.csv')
    train_file = os.path.join(config.DATA_DIR, 'processed_source_train.csv')
    val_file = os.path.join(config.DATA_DIR, 'processed_source_val.csv')

    # 检查预处理数据
    if (os.path.exists(train_file) and os.path.exists(val_file) and
        not config.USE_SYNTHETIC_DATA and not config.DATA_SAMPLE_SIZE):
        print(f"加载预处理数据：{train_file}, {val_file}")
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        return train_data, val_data

    # 加载原始数据
    print(f"尝试加载原始数据：{raw_data_file}")
    try:
        if config.USE_SYNTHETIC_DATA:
            raise FileNotFoundError("强制使用合成数据")
        if not os.path.exists(raw_data_file):
            raise FileNotFoundError(f"未找到 {raw_data_file}")

        # 检测分隔符
        with open(raw_data_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            for sep in [',', '\t', ';']:
                if len(first_line.split(sep)) == 3:
                    break
            else:
                raise ValueError("无法确定 CSV 分隔符，需为逗号、制表符或分号，且包含 3 列")

        # 尝试加载（支持标题或无标题）
        try:
            data = pd.read_csv(raw_data_file, sep=sep, header=0)
            if len(data.columns) != 3:
                raise ValueError("CSV 需包含 3 列")
        except:
            data = pd.read_csv(raw_data_file, sep=sep, header=None)
            data.columns = ['sentence1', 'sentence2', 'label']

        print(f"成功加载数据，形状：{data.shape}")
        print(f"列名：{data.columns.tolist()}")
        print(f"前两行：\n{data.head(2)}")

    except Exception as e:
        print(f"加载失败：{e}，生成合成数据...")
        data = pd.DataFrame({
            'sentence1': ['花呗如何还款', '支付宝怎么用', '我爱这个新手机', '今天天气真好', '明天会下雨吗'] * 40,
            'sentence2': ['花呗怎么还款', '花呗如何还款', '这个手机很棒', '今天天气不错', '明天下雨概率大吗'] * 40,
            'label': ([1, 0, 1, 1, 1]) * 40
        })

    # 数据清洗
    data = data.dropna()
    data['label'] = pd.to_numeric(data['label'], errors='coerce').astype(int)
    data = data.dropna()
    print(f"清洗后数据形状：{data.shape}")
    print(f"标签分布：{np.bincount(data['label'])}")

    # 采样
    if config.DATA_SAMPLE_SIZE and len(data) > config.DATA_SAMPLE_SIZE:
        data = data.sample(n=config.DATA_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    # 拆分数据集
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['label']
    )

    # 保存预处理数据
    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    print(f"保存预处理数据：{train_file}, {val_file}")

    return train_data, val_data

# --- 微调 BERT ---
def finetune_bert(train_data, val_data, config=Config):
    """微调 BERT 模型，保存到指定路径"""
    model_path = config.MODEL_SAVE_PATH
    if os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        print(f"已找到微调模型：{model_path}，跳过微调")
        return

    print(f"开始微调 BERT（中文），轮数：{config.FINETUNE_EPOCHS}")
    try:
        tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        model = BertForSequenceClassification.from_pretrained(
            './bert-base-chinese',
            num_labels=2
        )
    except OSError as e:
        print(f"加载失败：{e}，请确保 ./bert-base-chinese 包含模型文件")
        raise

    train_dataset = SentencePairDataset(train_data, tokenizer)
    val_dataset = SentencePairDataset(val_data, tokenizer)

    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=config.FINETUNE_EPOCHS,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir=config.LOG_DIR,
        logging_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available()
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"微调模型已保存：{model_path}")

# --- 特征提取 ---
def extract_features_batch(data, model, tokenizer, config=Config):
    """批量提取句对的 BERT CLS 特征"""
    print(f"提取 {len(data)} 条数据的特征...")
    model.to(config.DEVICE)
    model.eval()

    dataset = SentencePairDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.FEATURE_BATCH_SIZE)

    features = []
    labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(config.DEVICE),
                'attention_mask': batch['attention_mask'].to(config.DEVICE)
            }
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
            features.append(cls_embedding.cpu().numpy())
            if 'labels' in batch:
                labels.append(batch['labels'].cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0) if labels else None
    print(f"特征提取耗时：{time.time() - start_time:.2f} 秒")
    return features, labels

# --- EasyTL 实现 ---
class EasyTL:
    """EasyTL 算法，包括 CORAL 对齐和相似性预测（LP 和 Softmax 版本）"""
    @staticmethod
    def coral_alignment(source_features, target_features):
        """CORAL 对齐，减少源域和目标域特征分布差异"""
        feature_dim = source_features.shape[1]
        cov_s = np.cov(source_features.T) + np.eye(feature_dim) * 1e-5
        cov_t = np.cov(target_features.T) + np.eye(feature_dim) * 1e-5
        try:
            C_s_inv_sqrt = np.linalg.inv(sqrtm(cov_s))
            C_t_sqrt = sqrtm(cov_t)
            A = C_s_inv_sqrt @ C_t_sqrt
            z_s = source_features @ A
            z_t = target_features @ A
        except np.linalg.LinAlgError:
            print("CORAL 对齐失败，使用原始特征")
            z_s = source_features
            z_t = target_features
        return z_s, z_t

    @staticmethod
    def softmax_predict(z_s, z_t, source_labels, beta=1.0, num_classes=2):
        """Softmax 版本预测，输出连续相似性概率"""
        h_c = [np.mean(z_s[source_labels == c], axis=0) for c in range(num_classes)]
        h_c = np.array(h_c)
        distances = np.sum((z_t[:, np.newaxis, :] - h_c[np.newaxis, :, :]) ** 2, axis=2)
        neg_beta_distances = -beta * distances
        exp_scores = np.exp(neg_beta_distances - np.max(neg_beta_distances, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs[:, 1], (probs[:, 1] > 0.5).astype(int)

    @staticmethod
    def lp_predict(source_features, target_features, source_labels, num_classes=2):
        """线性规划版本预测，优化相似性权重"""
        N_t = target_features.shape[0]
        z_s, z_t = EasyTL.coral_alignment(source_features, target_features)
        h_c = [np.mean(z_s[source_labels == c], axis=0) for c in range(num_classes)]
        h_c = np.array(h_c)

        D = np.zeros((N_t, num_classes))
        for j in range(N_t):
            for c in range(num_classes):
                D[j, c] = np.sum((z_t[j] - h_c[c]) ** 2)

        prob = LpProblem("EasyTL_LP", LpMinimize)
        M_vars = [[LpVariable(f"M_{j}_{c}", 0, 1) for c in range(num_classes)] for j in range(N_t)]
        prob += lpSum(D[j][c] * M_vars[j][c] for j in range(N_t) for c in range(num_classes))
        for j in range(N_t):
            prob += lpSum(M_vars[j][c] for c in range(num_classes)) == 1
        if N_t >= num_classes:
            for c in range(num_classes):
                prob += lpSum(M_vars[j][c] for j in range(N_t)) >= 1

        solver = PULP_CBC_CMD(msg=False, timeLimit=300)
        prob.solve(solver)

        M_values = np.zeros((N_t, num_classes))
        if prob.status == 1:
            for j in range(N_t):
                for c in range(num_classes):
                    M_values[j, c] = M_vars[j][c].varValue
        else:
            print("LP 优化失败，使用最近中心点分配")
            closest = np.argmin(D, axis=1)
            for j in range(N_t):
                M_values[j, closest[j]] = 1.0

        # 添加正则化，软化概率
        probs = M_values + 1e-5
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1], np.argmax(probs, axis=1)

# --- 批量评估 ---
def evaluate_on_test_set(source_features, source_labels, test_data, model, tokenizer, config=Config):
    """在测试集上评估 EasyTL 性能"""
    print(f"\n评估测试集（{len(test_data)} 条数据）...")
    target_features, true_labels = extract_features_batch(test_data, model, tokenizer, config)

    if true_labels is None:
        print("错误：测试集无标签，无法评估")
        return {}

    # 使用 Softmax 版本（默认）
    probs, preds = EasyTL.softmax_predict(source_features, target_features, source_labels)
    # 可选：使用 LP 版本
    # probs, preds = EasyTL.lp_predict(source_features, target_features, source_labels)

    metrics = {
        'accuracy': accuracy_score(true_labels, preds),
        'f1_score': f1_score(true_labels, preds, zero_division=0),
        'precision': precision_score(true_labels, preds, zero_division=0),
        'recall': recall_score(true_labels, preds, zero_division=0),
        'auc': roc_auc_score(true_labels, probs) if len(np.unique(true_labels)) > 1 else -1.0
    }

    print("\n--- 测试集评估结果 ---")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    print("-----------------------\n")
    return metrics

# --- 交互式预测 ---
def predict_interactive(model, tokenizer, source_features, source_labels, config=Config):
    """交互式预测句对相似性"""
    model.to(config.DEVICE)
    model.eval()
    print("\n进入交互模式（输入 'quit' 退出）")
    while True:
        sent1 = input("句子 1: ")
        if sent1.lower() == 'quit':
            break
        sent2 = input("句子 2: ")

        encoding = tokenizer(
            sent1, sent2,
            return_tensors='pt',
            max_length=config.MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(config.DEVICE) for k in encoding}
        with torch.no_grad():
            outputs = model(**inputs)
            target_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        probs, _ = EasyTL.softmax_predict(source_features, target_features, source_labels)
        prob = probs[0]
        explanation = '高度相似' if prob > 0.7 else '有些相似' if prob > 0.3 else '不相似'
        print(f"\n相似性概率: {prob:.4f}")
        print(f"解释: {explanation}")

# --- 主函数 ---
def main():
    """主流程：数据准备、微调、特征提取、评估、交互预测"""
    start_time = time.time()
    config = Config()
    print(f"使用设备：{config.DEVICE}")

    # 步骤 1：数据准备
    print("\n=== 步骤 1：数据准备 ===")
    train_data, val_data = prepare_data(config)
    print(f"训练集形状：{train_data.shape}, 验证集形状：{val_data.shape}")

    # 步骤 2：微调 BERT
    print("\n=== 步骤 2：微调 BERT ===")
    finetune_bert(train_data, val_data, config)

    # 步骤 3：加载特征提取模型
    print("\n=== 步骤 3：加载模型 ===")
    try:
        model = BertModel.from_pretrained(config.MODEL_SAVE_PATH)
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
    except:
        print("加载微调模型失败，使用 bert-base-chinese")
        model = BertModel.from_pretrained('bert-base-chinese')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 步骤 4：提取源域特征
    print("\n=== 步骤 4：特征提取 ===")
    phi_s_file = os.path.join(config.DATA_DIR, 'Phi_s.npy')
    labels_file = os.path.join(config.DATA_DIR, 'source_labels.npy')
    if (os.path.exists(phi_s_file) and os.path.exists(labels_file) and
        not config.DATA_SAMPLE_SIZE):
        source_features = np.load(phi_s_file)
        source_labels = np.load(labels_file)
    else:
        source_features, source_labels = extract_features_batch(train_data, model, tokenizer, config)
        if not config.DATA_SAMPLE_SIZE:
            np.save(phi_s_file, source_features)
            np.save(labels_file, source_labels)
            print(f"保存源域特征：{phi_s_file}, {labels_file}")
    print(f"源域特征形状：{source_features.shape}, 标签形状：{source_labels.shape}")

    # 步骤 5：评估验证集
    print("\n=== 步骤 5：验证集评估 ===")
    evaluate_on_test_set(source_features, source_labels, val_data, model, tokenizer, config)

    # 步骤 6：交互式预测
   # print("\n=== 步骤 6：交互式预测 ===")
   # predict_interactive(model, tokenizer, source_features, source_labels, config)

    print(f"\n总耗时：{(time.time() - start_time) / 60:.2f} 分钟")

if __name__ == '__main__':
    main()