import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO

# --- 1. 數據對齊與讀取功能 ---

def read_and_resample(uploaded_files):
    """讀取多個檔案並對齊到統一的時間軸"""
    all_dfs = []
    file_names = []
    
    # 讀取原始數據
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file).iloc[1:]
        else:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=None, engine='python').iloc[1:]
        
        df.columns = ["Time_s", "Displacement_mm", "Force_kN"]
        df = df.astype(float).sort_values("Time_s")
        all_dfs.append(df)
        file_names.append(uploaded_file.name)

    # 定義統一的時間軸 (取所有檔案中最小的最大時間)
    max_time = min([df["Time_s"].max() for df in all_dfs])
    min_time = max([df["Time_s"].min() for df in all_dfs])
    common_time = np.linspace(min_time, max_time, 500) # 統一取 500 個點

    # 對每個檔案進行插值對齊
    aligned_data = {"Time_s": common_time}
    individual_processed = []

    for i, df in enumerate(all_dfs):
        interp_disp = np.interp(common_time, df["Time_s"], df["Displacement_mm"])
        interp_force = np.interp(common_time, df["Time_s"], df["Force_kN"])
        
        aligned_data[f"Disp_{i}"] = interp_disp
        aligned_data[f"Force_{i}"] = interp_force
        
        # 存回個別的 DF 方便後續計算
        temp_df = pd.DataFrame({
            "Time_s": common_time,
            "Displacement_mm": interp_disp,
            "Force_kN": interp_force,
            "Source": file_names[i]
        })
        individual_processed.append(temp_df)

    merged_df = pd.DataFrame(aligned_data)
    return individual_processed, merged_df

# --- 2. 統計與指標計算功能 ---

def calculate_detailed_metrics(df, label):
    """計算 AUC, Max Y, X at Max Y, AUC to Peak"""
    x = df["Displacement_mm"].values
    y = df["Force_kN"].values
    
    # 全域面積
    total_auc = np.trapz(y, x)
    
    # 最大值與對應 X
    idx_max = np.argmax(y)
    max_y = y[idx_max]
    x_at_max_y = x[idx_max]
    
    # 到最大值的面積
    auc_to_peak = np.trapz(y[:idx_max+1], x[:idx_max+1])
    
    return {
        "來源": label,
        "總曲線下面積 (kN-mm)": round(total_auc, 4),
        "最大力 (kN)": round(max_y, 4),
        "最大力對應位移 (mm)": round(x_at_max_y, 4),
        "峰值前曲線下面積 (kN-mm)": round(auc_to_peak, 4)
    }

# --- 3. 介面與繪圖 ---

st.set_page_config(page_title="材料測試數據整合分析", layout="wide")
st.title("🚀 進階數據整合與特性分析系統")

uploaded_files = st.file_uploader("上傳測試數據 (CSV/XLSX)", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    # A. 數據處理與對齊
    individual_dfs, merged_df = read_and_resample(uploaded_files)
    
    # 計算平均數據 (Consolidated)
    disp_cols = [c for c in merged_df.columns if "Disp_" in c]
    force_cols = [c for c in merged_df.columns if "Force_" in c]
    
    consolidated_df = pd.DataFrame({
        "Time_s": merged_df["Time_s"],
        "Displacement_mm": merged_df[disp_cols].mean(axis=1),
        "Force_kN": merged_df[force_cols].mean(axis=1),
        "Displacement_std": merged_df[disp_cols].std(axis=1),
        "Force_std": merged_df[force_cols].std(axis=1)
    })

    # B. 建立分析指標表
    summary_list = []
    for df in individual_dfs:
        summary_list.append(calculate_detailed_metrics(df, df["Source"].iloc[0]))
    
    # 加入最後一列 Consolidated 數據
    summary_list.append(calculate_detailed_metrics(consolidated_df, "★ Consolidated_Average"))
    summary_table = pd.DataFrame(summary_list)

    # C. 網頁呈現與 Tabs
    tab1, tab2, tab3 = st.tabs(["📊 曲線對比圖", "📋 統計特徵摘要", "💾 數據編輯與下載"])

    with tab1:
        st.subheader("所有輸入檔案與平均曲線對照")
        fig_all = go.Figure()
        
        # 繪製各個 input 曲線
        for df in individual_dfs:
            fig_all.add_trace(go.Scatter(x=df["Displacement_mm"], y=df["Force_kN"], 
                                         mode='lines', name=df["Source"].iloc[0], 
                                         line=dict(width=1), opacity=0.5))
        
        # 繪製平均曲線
        fig_all.add_trace(go.Scatter(x=consolidated_df["Displacement_mm"], y=consolidated_df["Force_kN"], 
                                     mode='lines', name="Average Result", 
                                     line=dict(color='black', width=3, dash='dash')))
        
        fig_all.update_layout(xaxis_title="Displacement (mm)", yaxis_title="Force (kN)", template="plotly_white")
        st.plotly_chart(fig_all, use_container_width=True)

    with tab2:
        st.subheader("數據特徵統計表")
        st.dataframe(summary_table, use_container_width=True)
        
        # 下載摘要表按鈕
        csv_summary = summary_table.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下載統計摘要表 (CSV)", data=csv_summary, 
                           file_name="summary_statistics.csv", mime="text/csv")

    with tab3:
        st.subheader("合併後的數據明細")
        edited_df = st.data_editor(consolidated_df, use_container_width=True)
        
        # 下載合併後的完整數據 (使用 Excel 方案避免錯位)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Result')
        
        st.download_button("下載整合數據 (Excel)", data=buffer.getvalue(), 
                           file_name="consolidated_data.xlsx", 
                           mime="application/vnd.ms-excel")

else:
    st.info("請上傳至少兩個檔案以進行對齊與統計分析。")