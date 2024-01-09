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
以dev-res-100-2为基准，reward采用 lossi - loss_(i-1)
改变criterion为
cost = torch.sum(torch.prod(pred * one_hots + (1 - pred) * (1 - one_hots), dim=1).unsqueeze(dim=-1) * rewards * combination(100, 10))
cost /= pred.shape[0]

100和10是硬编码的