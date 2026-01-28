import qlib
from qlib.data import D
from qlib.constant import REG_CN
import pandas as pd
import os

# 1. 这里的路径必须和你的数据下载脚本 (fetch_baostock_data.py) 中的 qlib_export_path 一致
provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_2024h1")

print(f"Checking data in: {provider_uri}")

try:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
except Exception as e:
    print(f"❌ Qlib init failed: {e}")
    exit(1)

print("-" * 30)

# 2. 测试获取一只常见的股票数据 (茅台)
# 注意：Baostock 的代码格式可能是 sh.600519 或 SH600519，我们都试一下
test_instruments = ['SH600519', 'sh.600519', '600519.SH']
found_data = False

for inst in test_instruments:
    print(f"🔍 尝试读取股票: {inst} ...")
    try:
        # 尝试读取 2020 年的数据
        df = D.features([inst], ['$close', '$open', '$volume'], start_time='2020-01-01', end_time='2020-01-10')
        if not df.empty:
            print(f"✅ 成功！读到数据了：\n{df}")
            found_data = True
            break
        else:
            print("⚠️  数据为空 (Empty DataFrame)")
    except Exception as e:
        print(f"❌ 读取报错: {e}")

if not found_data:
    print("\n🚨 严重问题：无法读取任何股票数据！")
    print("可能原因：")
    print("1. 股票代码格式不对 (Instruments list mismatch)")
    print("2. 字段名不对 (e.g. $Close vs $close)")
    print("3. 二进制 .bin 文件未生成")
else:
    print("\n✅ 单只股票读取正常。接下来检查 csi300 成分股...")
    
    # 3. 检查成分股是否能对上
    try:
        instruments = D.instruments('csi300')
        inst_list = D.list_instruments(instruments=instruments, start_time='2020-01-01', end_time='2020-01-01')
        print(f"📊 csi300 在 2020-01-01 包含股票数量: {len(inst_list)}")
        if len(inst_list) == 0:
            print("🚨 致命错误：csi300 列表为空，或者列表里的股票代码在你的数据库里找不到！")
    except Exception as e:
        print(f"❌ 获取 csi300 列表失败: {e}")