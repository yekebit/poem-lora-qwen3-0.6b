import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)

class SparseMoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features, top_k=2):
        super(SparseMoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 每个专家是一个独立的线性层（你可以替换成更复杂的网络）
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        # 门控网络：输入 -> 专家 logits
        self.gate = nn.Linear(in_features, num_experts)

    def forward(self, x):
        import pdb;pdb.set_trace()
        batch_size, seq_len, _ = x.shape if x.dim() == 3 else (x.shape[0], 1, x.shape[1])
        x_flat = x.view(-1, x.size(-1))  # [B*seq, in_features]，支持序列输入

        # 1. 门控打分（logits）
        gate_logits = self.gate(x_flat)  # [B*seq, num_experts]

        # 2. 选 top-k 专家
        topk_vals, topk_idxs = torch.topk(gate_logits, self.top_k, dim=-1)  # [B*seq, k]
        # topk_idxs.shape = [B*seq,k]
        # topk_vals.shape = [B*seq,k]
        topk_weights = F.softmax(topk_vals, dim=-1)  # [B*seq, k]

        # 3. 初始化输出
        output_flat = torch.zeros_like(x_flat[:, :self.experts[0].fc.out_features])  # [B*seq, out_features]

        # 4. 对每个专家，累加它被选中时的贡献
        for i, expert in enumerate(self.experts):
            # 找出哪些位置选择了当前专家 i
            selected = (topk_idxs == i)  # [B*seq, k], bool
            if selected.any():
                # 获取这些位置对应的权重（在 top-k 中的位置）
                weights = torch.where(selected, topk_weights, 0.0).sum(dim=-1)  # [B*seq]
                expert_out = expert(x_flat)  # [B*seq, out_features] 此时的expert是一个专家了
                output_flat += weights.unsqueeze(-1) * expert_out

        # 5. 恢复原始形状
        output = output_flat.view(*x.shape[:-1], -1)  # [B, seq, out] 或 [B, out]
        return output


# 参数
input_size = 5
output_size = 3
num_experts = 4
batch_size = 10
top_k = 2  # 每个 token 只激活 2 个专家

# 创建模型
model = SparseMoELayer(num_experts, input_size, output_size, top_k=top_k)

# 测试单样本或序列
# demo = torch.randn(batch_size, input_size)          # [10, 5]
demo = torch.randn(batch_size, 7, input_size)    # 也支持 [10, 7, 5]（序列）

output = model(demo)
print("Output shape:", output.shape)  # [10, 3] 或 [10, 7, 3]