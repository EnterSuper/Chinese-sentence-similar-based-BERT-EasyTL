import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.linalg import sqrtm
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import os
import time
import argparse

# --- 配置类 ---
class Config:
    """全局配置，管理数据路径、模型参数等"""
    DATA_DIR = './lcqmc/'  # 数据目录
    MODEL_SAVE_PATH = './finetuned_bert_chinese'  # 微调模型保存路径
    FEATURE_BATCH_SIZE = 64  # 特征提取批大小
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
    MAX_LENGTH = 128  # 最大序列长度

# --- 自定义数据集 ---
class SentencePairDataset(Dataset):
    """处理句对数据集，适配不同列名，生成 BERT 输入"""
    def __init__(self, data, tokenizer, max_length=Config.MAX_LENGTH):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        if self.label_col in self.data.columns:
            item['labels'] = torch.tensor(int(self.data.iloc[idx][self.label_col]), dtype=torch.long)
        return item

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
        # z_s, z_t = EasyTL.coral_alignment(source_features, target_features)
        z_s, z_t = [source_features, target_features]
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

        probs = M_values + 1e-5
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1], np.argmax(probs, axis=1)

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
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            features.append(cls_embedding.cpu().numpy())
            if 'labels' in batch:
                labels.append(batch['labels'].cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0) if labels else None
    print(f"特征提取耗时：{time.time() - start_time:.2f} 秒")
    return features, labels

# --- 直接 BERT 预测 ---
def bert_predict(data, model, tokenizer, config=Config):
    """使用微调的 BERT 模型直接预测"""
    print(f"直接 BERT 预测 {len(data)} 条数据...")
    model.to(config.DEVICE)
    model.eval()

    dataset = SentencePairDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.FEATURE_BATCH_SIZE)

    probs = []
    preds = []
    labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(config.DEVICE),
                'attention_mask': batch['attention_mask'].to(config.DEVICE)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            probs.append(batch_probs)
            preds.append(batch_preds)
            if 'labels' in batch:
                labels.append(batch['labels'].cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0) if labels else None
    print(f"直接 BERT 预测耗时：{time.time() - start_time:.2f} 秒")
    return probs, preds, labels

# --- 测试集评估 ---
def evaluate_test_set(source_features, source_labels, test_data, model_easytl, model_bert, tokenizer, config=Config, method='softmax'):
    """评估测试集，支持 EasyTL Softmax、LP 和直接 BERT 预测"""
    print(f"\n使用 {method} 方法评估测试集（{len(test_data)} 条数据）...")
    
    if method in ['softmax', 'lp']:
        target_features, true_labels = extract_features_batch(test_data, model_easytl, tokenizer, config)
        if true_labels is None:
            print("错误：测试集无标签，无法评估")
            return {}
        
        if method == 'softmax':
            probs, preds = EasyTL.softmax_predict(source_features, target_features, source_labels)
        else:
            probs, preds = EasyTL.lp_predict(source_features, target_features, source_labels)
    else:  # direct_bert
        probs, preds, true_labels = bert_predict(test_data, model_bert, tokenizer, config)
        if true_labels is None:
            print("错误：测试集无标签，无法评估")
            return {}

    metrics = {
        'accuracy': accuracy_score(true_labels, preds),
        'f1_score': f1_score(true_labels, preds, zero_division=0),
        'precision': precision_score(true_labels, preds, zero_division=0),
        'recall': recall_score(true_labels, preds, zero_division=0),
        'auc': roc_auc_score(true_labels, probs) if len(np.unique(true_labels)) > 1 else -1.0
    }

    print(f"\n--- {method} 测试集评估结果 ---")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    print("-----------------------\n")
    return metrics

# --- 加载测试数据 ---
def load_test_data(test_file, config=Config):
    """加载测试集，支持动态分隔符和列名"""
    print(f"加载测试数据：{test_file}")
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            for sep in [',', '\t', ';']:
                if len(first_line.split(sep)) == 3:
                    break
            else:
                raise ValueError("无法确定 CSV 分隔符，需为逗号、制表符或分号，且包含 3 列")

        try:
            data = pd.read_csv(test_file, sep=sep, header=0)
            if len(data.columns) != 3:
                raise ValueError("CSV 需包含 3 列")
        except:
            data = pd.read_csv(test_file, sep=sep, header=None)
            data.columns = ['sentence1', 'sentence2', 'label']

        print(f"成功加载测试数据，形状：{data.shape}")
        print(f"列名：{data.columns.tolist()}")
        print(f"前两行：\n{data.head(2)}")

        data = data.dropna()
        data['label'] = pd.to_numeric(data['label'], errors='coerce').astype(int)
        data = data.dropna()
        print(f"清洗后测试数据形状：{data.shape}")
        print(f"标签分布：{np.bincount(data['label'])}")
        return data

    except Exception as e:
        print(f"加载测试数据失败：{e}")
        raise

# --- 主函数 ---
def main():
    """测试脚本：加载模型和数据，评估测试集"""
    parser = argparse.ArgumentParser(description="测试 BERT 和 EasyTL 模型")
    parser.add_argument('--test_file', type=str, default='./lcqmc/ant_commercial.csv',
                        help='测试集 CSV 文件路径')
    parser.add_argument('--method', type=str, default='lp',
                        choices=['softmax', 'lp', 'direct_bert'],
                        help='预测方法：softmax, lp, direct_bert')
    args = parser.parse_args()

    config = Config()
    print(f"使用设备：{config.DEVICE}")
    start_time = time.time()

    # 加载测试数据
    test_data = load_test_data(args.test_file, config)

    # 加载微调模型和分词器
    print("\n加载微调模型...")
    try:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
        model_easytl = BertModel.from_pretrained(config.MODEL_SAVE_PATH)
        model_bert = BertForSequenceClassification.from_pretrained(config.MODEL_SAVE_PATH, num_labels=2)
    except Exception as e:
        print(f"加载微调模型失败：{e}")
        raise

    # 加载源域特征
    print("\n加载源域特征...")
    phi_s_file = os.path.join(config.DATA_DIR, 'Phi_s.npy')
    labels_file = os.path.join(config.DATA_DIR, 'source_labels.npy')
    if not (os.path.exists(phi_s_file) and os.path.exists(labels_file)):
        print(f"源域特征文件缺失：{phi_s_file}, {labels_file}")
        raise FileNotFoundError("请先运行 main.py 生成特征文件")
    source_features = np.load(phi_s_file)
    source_labels = np.load(labels_file)
    print(f"源域特征形状：{source_features.shape}, 标签形状：{source_labels.shape}")

    # 评估测试集
    print("\n评估测试集...")
    evaluate_test_set(
        source_features, source_labels, test_data,
        model_easytl, model_bert, tokenizer, config,
        method=args.method
    )

    print(f"\n总耗时：{(time.time() - start_time) / 60:.2f} 分钟")

if __name__ == '__main__':
    main()