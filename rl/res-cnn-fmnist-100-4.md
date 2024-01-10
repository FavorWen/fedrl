res5-seed*.log
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024 * 4
        num_blocks = 8
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
        history_dim = 8
实验配置与res-8-100一致，除数据集外