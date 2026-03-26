import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce

def read_single_file(uploaded_file):
    """Reads a single uploaded file and returns a formatted DataFrame."""
    if uploaded_file.name.endswith(".xlsx"):
        file_data = pd.ExcelFile(uploaded_file).parse(sheet_name=0).iloc[1:]
        file_data.columns = ["Time_s", "Displacement_mm", "Force_kN"]
    elif uploaded_file.name.endswith(".csv"):
        try:
            file_data = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python').iloc[1:]
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            file_data = pd.read_csv(uploaded_file, encoding='latin1', sep=None, engine='python').iloc[1:]
        file_data.columns = ["Time_s", "Displacement_mm", "Force_kN"]
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return None
    
    return file_data.astype(float)

def merge_data_frames(data_frames):
    """Renames columns to prevent duplicates and merges all DataFrames on 'Time_s'."""
    dfs_to_merge = []
    for i, df in enumerate(data_frames):
        renamed_df = df.rename(columns={
            "Displacement_mm": f"Displacement_mm_{i}",
            "Force_kN": f"Force_kN_{i}"
        })
        dfs_to_merge.append(renamed_df)

    return reduce(lambda left, right: pd.merge(left, right, on='Time_s', how='inner'), dfs_to_merge)

def compute_statistics(merged_data):
    """Computes statistics like average, std, min, and max for Displacement and Force."""
    # Select columns for displacement and force
    disp_cols = merged_data.filter(like="Displacement_mm")
    force_cols = merged_data.filter(like="Force_kN")

    # Compute statistics
    merged_data['Displacement_avg'] = disp_cols.mean(axis=1)
    merged_data['Displacement_std'] = disp_cols.std(axis=1)
    merged_data['Displacement_max'] = disp_cols.max(axis=1)
    merged_data['Displacement_min'] = disp_cols.min(axis=1)

    merged_data['Force_avg'] = force_cols.mean(axis=1)
    merged_data['Force_std'] = force_cols.std(axis=1)
    merged_data['Force_max'] = force_cols.max(axis=1)
    merged_data['Force_min'] = force_cols.min(axis=1)

    return merged_data[['Displacement_avg', 'Force_avg', 'Displacement_std', 'Force_std', 'Displacement_max', 'Displacement_min', 'Force_max', 'Force_min']]

# Main coordinator function
def process_uploaded_files(files):
    data_frames = []

    for uploaded_file in files:
        file_data = read_single_file(uploaded_file)
        if file_data is not None:
            data_frames.append(file_data)

    if len(data_frames) > 1:
        merged_data = merge_data_frames(data_frames)
        final_data = compute_statistics(merged_data)
        return final_data
    else:
        st.warning("Please upload at least two files for merging!")
        return None


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
# Streamlit interface setup
st.title("Displacement vs Force Data Consolidation and Visualization")

# File upload widgets
uploaded_files = st.file_uploader("Upload multiple files (xlsx or csv)", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    # Data processing
    consolidated_data = process_uploaded_files(uploaded_files)
    
    if consolidated_data is not None:
        st.success("Data consolidated successfully!")

        # Downloadable CSV file
        csv = consolidated_data.to_csv(index=False)
        st.download_button(label="Download Consolidated CSV File", data=csv, file_name="integrated_displacement_force_results.csv", mime="text/csv")

        # Plot 1: Displacement_avg vs Force_avg
        fig1 = px.scatter(consolidated_data, x="Displacement_avg", y="Force_avg", title="Displacement_avg vs Force_avg Curve", labels={"Displacement_avg": "Displacement (mm)", "Force_avg": "Force (kN)"})
        st.plotly_chart(fig1)

        # Plot 2: Displacement variability vs Force variability
        fig2 = plot_variability(consolidated_data)
        st.plotly_chart(fig2)

        # # Plot 3
        # fig2 = px.scatter(consolidated_data, x="Displacement_std", y="Force_std", title="Displacement vs Force Variability", labels={"Displacement_std": "Displacement STD (mm)", "Force_std": "Force STD (kN)"})
        # st.plotly_chart(fig2)

        # Display processed data
        st.write("Processed Data Table:")
        st.dataframe(consolidated_data)
    else:
        st.error("Failed to consolidate data. Ensure files have matching structure.")
