import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# ================= 配置区域 =================
# 你的 Qlib 数据目录 (请确保与之前一致)
QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data_2024h1")
INSTRUMENTS_DIR = os.path.join(QLIB_DATA_DIR, "instruments")
START_DATE = "2010-01-01"  # 抓取开始时间
END_DATE = datetime.now().strftime("%Y-%m-%d") # 抓取到今天

def get_trading_dates(start, end):
    """获取所有交易日"""
    rs = bs.query_history_k_data_plus("sh.000001", "date", start_date=start, end_date=end, frequency="d")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    return [x[0] for x in data_list]

def format_code(code):
    """将 sh.600000 转换为 SH600000 (Qlib格式)"""
    return code.replace(".", "").upper()

def update_csi300():
    print("🚀 正在登录 Baostock...")
    bs.login()

    if not os.path.exists(INSTRUMENTS_DIR):
        os.makedirs(INSTRUMENTS_DIR)

    # 为了效率，我们按月采样更新成分股（成分股变动通常不频繁）
    # 获取每个月的最后一个交易日作为采样点
    print("📅 获取交易日历...")
    all_dates = get_trading_dates(START_DATE, END_DATE)
    
    # 转换成 datetime 对象方便处理
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
    
    # 筛选每月的最后一天 (近似) 和现在的最后一天
    sample_dates = []
    current_month = None
    for d in date_objs:
        if current_month != d.month:
            if sample_dates:
                # 把上个月的最后一天存下来
                pass 
            current_month = d.month
        # 简单策略：每隔 15 天查一次，或者每个月查一次
        # Baostock 的 query_hs300_stocks 接口只支持查询具体某一天
    
    # 更稳妥的策略：按月遍历，查询当月第一天
    print(f"🔍 开始抓取 CSI300 成分股 ({START_DATE} ~ {END_DATE})...")
    
    # 存储结构: { 'SH600000': [[start, end], [start, end]], ... }
    stock_intervals = {}
    
    # 生成按月的日期列表
    check_dates = []
    curr = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    while curr <= end:
        # 找离这个日期最近的交易日（向前找）
        date_str = curr.strftime("%Y-%m-%d")
        # 这里简化处理，直接查这一天，如果是非交易日 baostock 会返回空或前一值
        # 实际上 baostock query_hs300 需要指定 date，我们用每月的 15 号查
        check_dates.append(date_str)
        # 下个月
        curr += timedelta(days=30)
    
    # 确保最后一天也被包含
    if END_DATE not in check_dates:
        check_dates.append(END_DATE)

    # 开始查询
    all_cons = []
    
    for date in tqdm(check_dates):
        rs = bs.query_hs300_stocks(date=date)
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            # row format: updateDate, code, code_name
            code = format_code(row[1])
            all_cons.append({'date': date, 'code': code})
            
    df = pd.DataFrame(all_cons)
    if df.empty:
        print("❌ 未获取到任何成分股数据，请检查网络或日期范围。")
        return

    print("🧩 正在整理时间区间 (这可能需要一点时间)...")
    
    # 转换为 Qlib 格式：Symbol, Start_Date, End_Date
    # 逻辑：如果一只股票在 1月出现，2月出现，3月出现，那它就是 1月~3月
    final_list = []
    
    # 获取所有出现过的股票
    unique_stocks = df['code'].unique()
    
    for stock in tqdm(unique_stocks):
        dates = sorted(df[df['code'] == stock]['date'].tolist())
        if not dates:
            continue
            
        # 合并连续的时间段
        start = dates[0]
        last = dates[0]
        
        for i in range(1, len(dates)):
            curr = dates[i]
            # 计算两个日期相差天数
            d1 = datetime.strptime(last, "%Y-%m-%d")
            d2 = datetime.strptime(curr, "%Y-%m-%d")
            
            # 如果间隔超过 40 天（因为我们是约30天采一次样），说明中间断了
            if (d2 - d1).days > 45:
                # 结算上一段
                final_list.append(f"{stock}\t{start}\t{last}")
                start = curr
            
            last = curr
        
        # 结算最后一段 (把结束时间延后到未来，确保现在能用)
        # 如果 last 是最近的日期，我们假设它直到未来都有效
        final_list.append(f"{stock}\t{start}\t2099-12-31")

    # 写入文件
    out_path = os.path.join(INSTRUMENTS_DIR, "csi300.txt")
    print(f"💾 正在写入文件: {out_path}")
    
    with open(out_path, "w") as f:
        for line in final_list:
            f.write(line + "\n")
            
    # 顺便生成一个 all.txt (全市场)
    print("💾 顺便生成 all.txt (包含所有出现过的股票)...")
    all_path = os.path.join(INSTRUMENTS_DIR, "all.txt")
    with open(all_path, "w") as f:
        for stock in unique_stocks:
            f.write(f"{stock}\t{START_DATE}\t2099-12-31\n")

    bs.logout()
    print("✅ 完成！现在你的 csi300 数据是最新的了。")

if __name__ == "__main__":
    update_csi300()
