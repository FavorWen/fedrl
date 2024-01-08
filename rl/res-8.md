res5-seed*.log
实验环境：
        p_hidden_size = 1024 * 2
        l_hidden_size = 512 * 2
        hidden_size = 1024 * 2
        num_blocks = 32 * 4
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
以res-5.md为基准，看加深网络深度、加宽网络宽度的效果