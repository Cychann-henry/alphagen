import qlib
from qlib.data import D
from qlib.constant import REG_CN
import pandas as pd
import os

def check_latest_dates():
    # 数据路径
    provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data_2024h1")
    print(f"初始化 Qlib，数据路径: {provider_uri}")
    
    try:
        qlib.init(provider_uri=provider_uri, region=REG_CN)
    except Exception as e:
        print(f"❌ Qlib 初始化失败: {e}")
        return

    print("-" * 40)

    # 1. 检查总体交易日历
    cal = D.calendar(start_time='1990-01-01', end_time='2099-12-31')
    if len(cal) > 0:
        latest_cal_date = cal[-1]
        print(f"🗓️  Qlib 日历记录的最后一个交易日: {latest_cal_date.strftime('%Y-%m-%d')}")
    else:
        print("❌ 日历为空！")

    print("-" * 40)

    # 2. 检查 csi300 成分股配置的最新有效日期
    print("🔍 正在检查 csi300 成分股最新日期...")
    try:
        # 获取极宽时间范围内的所有 csi300 记录返回字典格式 {stock: [[start_date, end_date], ...]}
        inst_dict = D.list_instruments(D.instruments('csi300'), start_time='2000-01-01', end_time='2099-12-31', as_list=False)
        
        max_end_date = pd.Timestamp('1970-01-01')
        for inst, spans in inst_dict.items():
            for span in spans:
                # span[1] 是该股票在 csi300 中的结束日期
                span_end = pd.Timestamp(span[1])
                if span_end > max_end_date:
                    max_end_date = span_end

        print(f"✅ csi300 配置的最晚有效日期为: {max_end_date.strftime('%Y-%m-%d')}")
        if max_end_date.year < 2022:
            print("🚨 警告：你的 csi300 成分股数据已经严重过期，没有包含 2022 年及以后的数据！")
    except Exception as e:
        print(f"❌ 检查 csi300 失败: {e}")

    print("-" * 40)

    # 3. 检查实际 K 线数据的最新日期 (以 SH600519 为例)
    test_stock = 'SH600519'
    print(f"🔍 正在检查 {test_stock} (贵州茅台) K线数据的最新日期...")
    try:
        # 尝试读取直到一个极远的未来日期
        df = D.features([test_stock], ['$close'], start_time='2020-01-01', end_time='2099-12-31')
        if not df.empty:
            # 必须 dropna，因为如果没有数据，qlib 可能会返回填满 NaN 的 DataFrame
            df_valid = df.dropna()
            if not df_valid.empty:
                latest_kline_date = df_valid.index.get_level_values('datetime').max()
                print(f"✅ {test_stock} 的最新真实 K 线日期为: {latest_kline_date.strftime('%Y-%m-%d')}")
            else:
                print(f"⚠️ {test_stock} 在这期间的数据全部为缺失值(NaN)！")
        else:
            print(f"⚠️ 无法读取到 {test_stock} 的任何数据 (DataFrame 为空)。")
    except Exception as e:
        print(f"❌ 检查 K 线特征失败: {e}")

if __name__ == "__main__":
    check_latest_dates()