import numpy as np

# 示例宾果卡，1代表“是”，0代表“否”
bingo_card_1 = np.array([
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1]
])

bingo_card_2 = np.array([
    [0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1]
])

def calculate_weighted_similarity(card1, card2, lambda_param=2, miu_4=0.6, miu_3=0.25, num_lines=12):
    assert card1.shape == card2.shape, "The two bingo cards must have the same dimensions"
    
    # 基本相似性
    basic_similarity = np.sum(card1 == card2) / card1.size
    
    def line_similarity(line1, line2):
        count = np.sum(line1 == line2)
        if count == 5:
            return 1
        elif count == 4:
            return miu_4
        elif count == 3:
            return miu_3
        return 0
    
    # 加权相似性
    weighted_similarity = 0
    for i in range(5):
        weighted_similarity += line_similarity(card1[i, :], card2[i, :])  # 横线
        weighted_similarity += line_similarity(card1[:, i], card2[:, i])  # 竖线

    # 对角线
    weighted_similarity += line_similarity(np.diag(card1), np.diag(card2))
    weighted_similarity += line_similarity(np.diag(np.fliplr(card1)), np.diag(np.fliplr(card2)))
    weighted_similarity /= num_lines
    
    # 总相似性
    total_similarity = (basic_similarity + lambda_param * weighted_similarity) / (1+lambda_param)
    
    return total_similarity

# 计算两张宾果卡的相似性
similarity = calculate_weighted_similarity(bingo_card_1, bingo_card_2)
print(f"The similarity between the two bingo cards is: {similarity:.2%}")
