"""检查 test 相关子集是否存在同名文件"""
import os

SOURCE_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
subsets = ['ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt']

all_names = []
for s in subsets:
    files = os.listdir(os.path.join(SOURCE_DIR, s))
    files = [f for f in files if f.endswith('.jpg')]
    print(f"{s}: {len(files)} 张")
    all_names.extend(files)

total = len(all_names)
unique = len(set(all_names))
print(f"\n总计: {total} 张, 唯一文件名: {unique}, 重复(被覆盖): {total - unique}")
