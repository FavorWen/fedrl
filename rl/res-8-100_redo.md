res4-seed*.log
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024 * 4
        num_blocks = 8
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
和相同情况下的CNN 100(dev-res-5-100)作对比, 增加residual block宽度，减少深度
criterion采用multiprob
reward = loss_i - loss_(i-1)
optimizer = sgd + clip

participant_nums = 10