import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)
    
class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        # x.size() = [batch_size,input_size]
        import pdb;pdb.set_trace()
        gate_score = F.softmax(self.gate(x), dim=-1)
        # gate.size() = [batch_size,num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outputs.size() = [batch_size,num_experts,output_feature]
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)

        """
        gate_score.unsqueeze(1).size() = [batch_size, 1, expert_num])
        output.size() = [batch_size,output_size]
        """
        return output

input_size = 5
output_size = 3
num_experts = 4
batch_size = 10

model = MoELayer(num_experts, input_size, output_size)


demo = torch.randn(batch_size, input_size)

output = model(demo)

print(output.shape)  # 输出: torch.Size([10, 3])