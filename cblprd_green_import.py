"""
从 CBLPRD-330k_v1 中提取新能源绿牌，复制到 re_LPR_DATA_PERSP/green/
train.txt → green/train/
val.txt   → green/train/（并入训练，不单独留 val）
"""

import os
import shutil

CBLPRD_ROOT = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CBLPRD-330k_v1"
OUTPUT_ROOT = r"C:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\re_LPR_DATA_PERSP\green"

GREEN_TYPES = {"新能源小型车", "新能源大型车"}

CHARS_VALID = set(
    '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
    '0123456789'
    'ABCDEFGHJKLMNPQRSTUVWXYZ'
)


def is_valid_plate(plate):
    """8位新能源车牌，第3位必须是D或F"""
    if len(plate) != 8:
        return False
    for c in plate:
        if c not in CHARS_VALID:
            return False
    return True


def process_split(txt_path, out_dir, tag, src_root):
    os.makedirs(out_dir, exist_ok=True)

    ok = 0
    skip_type = 0
    skip_invalid = 0
    skip_missing = 0

    with open(txt_path, encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split(' ')
        if len(parts) < 3:
            continue

        rel_path = parts[0]
        plate    = parts[1]
        ptype    = parts[2]

        if ptype not in GREEN_TYPES:
            skip_type += 1
            continue

        if not is_valid_plate(plate):
            skip_invalid += 1
            continue

        src = os.path.join(src_root, rel_path)
        if not os.path.exists(src):
            skip_missing += 1
            continue

        dst_name = '{}_{}_{:06d}.jpg'.format(plate, tag, ok)
        dst = os.path.join(out_dir, dst_name)
        shutil.copy2(src, dst)
        ok += 1

        if ok % 5000 == 0:
            print('  {} 张已复制...'.format(ok))

    print('  完成: ok={}, 跳过(非绿牌)={}, 跳过(格式异常)={}, 跳过(文件缺失)={}'.format(
        ok, skip_type, skip_invalid, skip_missing))
    return ok


def main():
    train_txt = os.path.join(CBLPRD_ROOT, 'train.txt')
    val_txt   = os.path.join(CBLPRD_ROOT, 'val.txt')
    src_root  = CBLPRD_ROOT

    out_train = os.path.join(OUTPUT_ROOT, 'train')

    print('[1/2] 处理 train.txt → green/train/')
    n1 = process_split(train_txt, out_train, 'cbltrain', src_root)

    print('[2/2] 处理 val.txt → green/train/（并入训练）')
    n2 = process_split(val_txt, out_train, 'cblval', src_root)

    print('\n全部完成！新增绿牌: {} 张'.format(n1 + n2))
    print('输出目录: {}'.format(out_train))

    # 统计当前 green/train 总数
    total = len([f for f in os.listdir(out_train) if f.endswith('.jpg')])
    print('green/train 现有总数: {} 张'.format(total))


if __name__ == '__main__':
    main()
