res5-seed*.log
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024
        num_blocks = 32 * 2
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
以res-4.md为基准，看加深网络深度的效果