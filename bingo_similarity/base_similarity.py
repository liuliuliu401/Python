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

def calculate_base_similarity(card1, card2):
    # 确保两张卡的尺寸相同
    assert card1.shape == card2.shape, "The two bingo cards must have the same dimensions"
    
    # 计算相同格子的数量
    same_count = np.sum(card1 == card2)
    
    # 计算相似性
    similarity = same_count / card1.size
    
    return similarity

# 计算两张宾果卡的相似性
similarity = calculate_base_similarity(bingo_card_1, bingo_card_2)
print(f"The similarity between the two bingo cards is: {similarity:.2%}")
