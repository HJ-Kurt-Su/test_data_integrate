import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce
import numpy as np
from scipy.integrate import simpson
import io

# --- Functions ---

def read_single_file(uploaded_file):
    """Reads and formats a single file, adding a 'Source' column for identification."""
    if uploaded_file.name.endswith(".xlsx"):
        file_data = pd.ExcelFile(uploaded_file).parse(sheet_name=0).iloc[1:]
    elif uploaded_file.name.endswith(".csv"):
        try:
            file_data = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python').iloc[1:]
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            file_data = pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python').iloc[1:]
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return None
    
    file_data.columns = ["Time_s", "Displacement_mm", "Force_kN"]
    df = file_data.astype(float)
    
    # 功能 0: 數據對齊 (歸零處理)
    df["Displacement_mm"] = df["Displacement_mm"] - df["Displacement_mm"].iloc[0]
    df["Force_kN"] = df["Force_kN"] - df["Force_kN"].iloc[0]
    
    df["Source"] = uploaded_file.name
    return df

def analyze_curve(df, x_col, y_col, label):
    """功能 2-4: 計算曲線下面積與最大值特徵"""
    # 確保數據按 X 排序以利積分
    df_sorted = df.sort_values(by=x_col)
    x = df_sorted[x_col].values
    y = df_sorted[y_col].values
    
    # 2. 全曲線面積 (AUC)
    total_auc = simpson(y=y, x=x)
    
    # 3. Y軸最大值及其對應 X
    idx_max = np.argmax(y)
    y_max = y[idx_max]
    x_at_y_max = x[idx_max]
    
    # 4. 到達 Y 最大值前的面積
    auc_to_max = simpson(y=y[:idx_max+1], x=x[:idx_max+1])
    
    return {
        "Dataset": label,
        "Total_AUC": total_auc,
        "Max_Force_kN": y_max,
        "Disp_at_Max_Force_mm": x_at_y_max,
        "AUC_to_Max": auc_to_max
    }

def merge_data_frames(data_frames):
    dfs_to_merge = []
    for i, df in enumerate(data_frames):
        # 僅保留數值欄位進行合併，並加上索引
        temp_df = df[["Time_s", "Displacement_mm", "Force_kN"]].rename(columns={
            "Displacement_mm": f"Displacement_mm_{i}",
            "Force_kN": f"Force_kN_{i}"
        })
        dfs_to_merge.append(temp_df)
    return reduce(lambda left, right: pd.merge(left, right, on='Time_s', how='inner'), dfs_to_merge)

def compute_statistics(merged_data):
    disp_cols = merged_data.filter(like="Displacement_mm")
    force_cols = merged_data.filter(like="Force_kN")

    merged_data['Displacement_avg'] = disp_cols.mean(axis=1)
    merged_data['Displacement_std'] = disp_cols.std(axis=1)
    merged_data['Force_avg'] = force_cols.mean(axis=1)
    merged_data['Force_std'] = force_cols.std(axis=1)
    
    # 用於後續分析的欄位
    return merged_data

def process_uploaded_files(files):
    data_frames = []
    analysis_results = []

    for uploaded_file in files:
        file_data = read_single_file(uploaded_file)
        if file_data is not None:
            data_frames.append(file_data)
            # 功能 2-4: 個別檔案分析
            analysis_results.append(analyze_curve(file_data, "Displacement_mm", "Force_kN", uploaded_file.name))

    if len(data_frames) > 1:
        merged_data = merge_data_frames(data_frames)
        consolidated_data = compute_statistics(merged_data)
        
        # 功能 2-4: 綜合數據分析
        avg_analysis = analyze_curve(consolidated_data, "Displacement_avg", "Force_avg", "Consolidated_Avg")
        analysis_results.append(avg_analysis)
        
        # 功能 5: 轉化為結果 DataFrame
        summary_df = pd.DataFrame(analysis_results)
        
        return consolidated_data, data_frames, summary_df
    else:
        st.warning("Please upload at least two files for merging!")
        return None, None, None

def plot_all_curves(data_frames, consolidated_data):
    """功能 1: 繪製所有原始曲線與平均曲線於同一圖表"""
    fig = go.Figure()
    
    # 繪製各個 input 曲線
    for df in data_frames:
        fig.add_trace(go.Scatter(
            x=df["Displacement_mm"], y=df["Force_kN"],
            mode='lines', name=df["Source"].iloc[0],
            line=dict(width=1), opacity=0.5
        ))
    
    # 繪製平均曲線
    fig.add_trace(go.Scatter(
        x=consolidated_data["Displacement_avg"], 
        y=consolidated_data["Force_avg"],
        mode='lines', name="Consolidated Average",
        line=dict(color='black', width=3, dash='dash')
    ))
    
    fig.update_layout(title="All Input Curves vs Consolidated Average", 
                      xaxis_title="Displacement (mm)", yaxis_title="Force (kN)",
                      template="plotly_white")
    return fig

# Example: Generating figure with final_data
def plot_variability(data):
    fig = go.Figure()

    # Add scatter plot with error bars for displacement and force variability
    fig.add_trace(go.Scatter(
        x=data['Displacement_avg'],
        y=data['Force_avg'],
        error_x=dict(
            type='data',
            array=data['Displacement_std'],
            visible=True
        ),
        error_y=dict(
            type='data',
            array=data['Force_std'],
            visible=True
        ),
        mode='markers',
        name="Average with Variability",
        marker=dict(color='orange', size=8)
    ))

    # Customize the layout
    fig.update_layout(
        title="Displacement vs Force with Variability",
        xaxis_title="Displacement (mm)",
        yaxis_title="Force (kN)",
        legend_title="Legend",
        template="plotly_white"
    )
    return fig


def get_plotly_download_link(fig, file_name):
    """將 Plotly 圖表轉換為可下載的 HTML 位元組流"""
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()
    
    return html_bytes
# --- Streamlit UI ---
st.title("Advanced Displacement vs Force Analysis")

uploaded_files = st.file_uploader("Upload files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    consolidated_data, individual_dfs, summary_df = process_uploaded_files(uploaded_files)
    
    if consolidated_data is not None:
        st.success("Analysis Complete!")

        # 功能 1: 呈現所有曲線圖
        st.subheader("1. Combined Curves Visualization")
        fig_all = plot_all_curves(individual_dfs, consolidated_data)
        # st.plotly_chart(plot_all_curves(individual_dfs, consolidated_data))
        st.plotly_chart(fig_all)
    
    # 新增下載按鈕 (Combined Curves)
        st.download_button(
            label="Download Combined Curves Plot (HTML)",
            data=get_plotly_download_link(fig_all, "combined_curves.html"),
            file_name="combined_curves.html",
            mime="text/html",
            key="download_all_curves"
        )

        # 功能 5: 呈現分析結果表格與下載按鈕
        st.subheader("2. Key Performance Indicators (KPIs) Summary")
        st.dataframe(summary_df)
        
        summary_csv = summary_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Summary Report (CSV)",
            data=summary_csv,
            file_name="curve_analysis_summary.csv",
            mime="text/csv"
        )

        # 保留原有的下載與統計圖表
        st.subheader("3. Statistical Data")
        csv_full = consolidated_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Download Full Consolidated Data", csv_full, "full_data.csv")
        
        # 顯示原本的 Variability Plot
        # (這裡需確保 plot_variability 傳入的是包含平均值的 DataFrame)
        # st.plotly_chart(plot_variability(consolidated_data))
        st.subheader("3. Statistical Data & Variability")
        fig_var = plot_variability(consolidated_data)
        st.plotly_chart(fig_var)
        
        # 新增下載按鈕 (Variability Plot)
        st.download_button(
            label="Download Variability Plot (HTML)",
            data=get_plotly_download_link(fig_var, "variability_plot.html"),
            file_name="variability_plot.html",
            mime="text/html",
            key="download_variability"
    )