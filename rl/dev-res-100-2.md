res-100-4-seed*.log
client_nums = 100
participants = 10
history_dim = 8
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024 * 2
        num_blocks = 32 + 16
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
以res-100-4为基准，实验reward变成 lossi - loss_(i-1)
criterion为
cost = torch.sum(torch.prod(pred * one_hots + (1 - pred) * (1 - one_hots), dim=1).unsqueeze(dim=-1) * rewards)
cost /= pred.shape[0]