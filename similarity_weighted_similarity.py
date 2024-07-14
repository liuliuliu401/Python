import numpy as np
from cosine_sim import cosine_sim

# 示例宾果卡文本
bingo_text = [
    ["男的", "在大专/末流985211上大学", "加入了学校的动漫社", "经常在游戏/动画/二次元同好群里活跃", "聊天栏基本上都是二次元群"],
    ["玩过五款以上二次元手游", "每个月至少看一部当月番剧", "阅片无数自认资深动画迷", "steam游戏时长合计500h以上", "每天和舍友以外的人说话不超过三句"],
    ["熟练运用各种二次元梗", "热衷购买与搜集手办/模型/周边", "大学目前没有校花", "能熟练运用互联网找到自己想找的任何资源", "周末和假期基本不出门"],
    ["单身并憧憬纯洁的爱情", "二次元美少女头像", "知道酒肆及其名人名言", "刷b站/贴吧/论坛", "喜欢男娘/百合"],
    ["玩gal/看轻小说", "歌单里基本没有中文歌", "相册里二次元图片占一半以上", "经常在网上发表自己的意见", "打音游"]
]

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


def calculate_line_logical_similarity(text_line):
    """
        获得行/列/对角线的相似度评分
    """
    cosine_matrix = cosine_sim(text_line)
    # 提取上三角部分（不包括对角线）的索引
    upper_tri_indices = np.triu_indices(len(cosine_matrix), k=1)
    # 提取上三角部分的值
    upper_tri_values = cosine_matrix[upper_tri_indices]
    # 计算平均相似度
    avg_similarity = np.mean(upper_tri_values)
    return avg_similarity

def calculate_line_similarity(line1, line2, miu_4=0.6, miu_3=0.25):
    count = np.sum(line1 == line2)
    if count == 5:
        return 1
    elif count == 4:
        return miu_4
    elif count == 3:
        return miu_3
    return 0

def calculate_similarity(card1, card2, text_grid, lambda_param=2, miu_4=0.6, miu_3=0.25):
    assert card1.shape == card2.shape, "The two bingo cards must have the same dimensions"
    assert card1.shape[0] == card1.shape[1], "The two bingo cards must be squares"
    
    # 基本相似性
    basic_similarity = np.sum(card1 == card2) / card1.size

    # 计算逻辑相似性
    row_logical_similarities = np.zeros(card1.shape[0])
    col_logical_similarities = np.zeros(card1.shape[0])
    np_text_grid = np.array(text_grid)

    for i in range(card1.shape[0]):
        row_logical_similarities[i] = calculate_line_logical_similarity(np_text_grid[i, :])
        col_logical_similarities[i] = calculate_line_logical_similarity(np_text_grid[:, i])
    
    diag1_logical_similarity = calculate_line_logical_similarity([text_grid[i][i] for i in range(len(text_grid))])
    diag2_logical_similarity = calculate_line_logical_similarity([text_grid[i][len(text_grid) - 1 - i] for i in range(len(text_grid))])

    # for this example, result is:
    # row_logical_similarities = array([0.26199278, 0.25492245, 0.24864471, 0.25135895, 0.23563964])
    # col_logical_similarities = array([0.2504028 , 0.17183741, 0.19847852, 0.30093073, 0.25710556])
    # diag1_logical_similarity = 0.2114661159748125
    # diag2_logical_similarity = 0.22413438949947112
    
    def apply_text_weight(line_similarity, text_similarity):
        return line_similarity * text_similarity

    # 加权相似性
    weighted_similarity = 0
    logical_similarity_sum = 0

    for i in range(5):
        row_similarity = calculate_line_similarity(card1[i, :], card2[i, :], miu_4=miu_4, miu_3=miu_3)
        col_similarity = calculate_line_similarity(card1[:, i], card2[:, i], miu_4=miu_4, miu_3=miu_3)
        
        row_weight = apply_text_weight(row_similarity, row_logical_similarities[i])
        col_weight = apply_text_weight(col_similarity, col_logical_similarities[i])
        
        weighted_similarity += row_weight
        weighted_similarity += col_weight
        
        logical_similarity_sum += row_logical_similarities[i]
        logical_similarity_sum += col_logical_similarities[i]

    diag1_similarity = calculate_line_similarity(np.diag(card1), np.diag(card2))
    diag2_similarity = calculate_line_similarity(np.diag(np.fliplr(card1)), np.diag(np.fliplr(card2)))
    
    diag1_weight = apply_text_weight(diag1_similarity, diag1_logical_similarity)
    diag2_weight = apply_text_weight(diag2_similarity, diag2_logical_similarity)
    
    weighted_similarity += diag1_weight
    weighted_similarity += diag2_weight
    
    logical_similarity_sum += diag1_logical_similarity
    logical_similarity_sum += diag2_logical_similarity

    weighted_similarity /= logical_similarity_sum

    total_similarity = (basic_similarity + lambda_param * weighted_similarity) / (1 + lambda_param)
    
    return total_similarity


# 计算两张宾果卡的相似性
similarity = calculate_similarity(bingo_card_1, bingo_card_2, bingo_text)
print(f"The similarity between the two bingo cards is: {similarity:.2%}")
# The similarity between the two bingo cards is: 85.84%


