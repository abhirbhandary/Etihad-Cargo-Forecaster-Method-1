import streamlit as st
import pandas as pd
import numpy as np

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Etihad Cargo Model - 1", layout="wide")
st.title("✈️ Etihad Cargo: LD3 Mathematical Forecaster")

# Initialize Session State
if 'res_df' not in st.session_state:
    st.session_state['res_df'] = None

# Custom Rounding Function
def threshold_ceil(value, threshold=0.20):
    if value == 0 or pd.isna(value): return 0
    fraction = value % 1
    return np.floor(value) if (0 < fraction <= threshold) else np.ceil(value)

# --- 2. DATA LOADING & PREP ---
@st.cache_data
def get_processed_data():
    # Make sure your CSV file is in the same folder!
    try:
        df = pd.read_csv('etihad_data.csv')
        df['Leg Dep Date'] = pd.to_datetime(df['Leg Dep Date'])
        df['DOW'] = df['Leg Dep Date'].dt.dayofweek + 1
        
        # Calculate base metrics
        # (Handling division by zero with replace)
        df['ShowUp_Rate_FJ'] = (df['PAX Count Actl F/B'] / (df['F'] + df['J']).replace(0, np.nan)).fillna(1)
        df['ShowUp_Rate_Y'] = (df['PAX Count Actl Y'] / df['Y'].replace(0, np.nan)).fillna(1)
        
        mask_bags = df['Bag Count L3E Actl Ttl'] > 0
        df.loc[mask_bags, 'PPB_FJ_actual'] = df['PAX Count Actl F/B'] / df['Bag Count L3E Actl F/B'].replace(0, np.nan)
        df.loc[mask_bags, 'PPB_Y_actual'] = df['PAX Count Actl Y'] / df['Bag Count L3E Actl Y'].replace(0, np.nan)
        
        # Ensure MAX_LD3 exists, default to 30 if missing (Safety check)
        if 'MAX_LD3' not in df.columns:
            df['MAX_LD3'] = 30 
            
        return df
    except FileNotFoundError:
        st.error("File 'etihad_data.csv' not found. Please drag and drop it into the sidebar file explorer.")
        return pd.DataFrame()

df = get_processed_data()

# --- 3. SIDEBAR CONTROLS ---
if not df.empty:
    st.sidebar.header("Model Parameters")
    
    # Date Picker
    min_date = df['Leg Dep Date'].min()
    max_date = df['Leg Dep Date'].max()
    
    test_range = st.sidebar.date_input("Evaluation Period", 
                                       [max_date - pd.Timedelta(days=30), max_date])
    
    lookback = st.sidebar.slider("Lookback Window (Days)", 30, 180, 90)
    user_threshold = st.sidebar.slider("Squeeze Threshold", 0.0, 0.5, 0.2, 0.05)

    # --- 4. THE MATHEMATICAL ENGINE ---
    def run_model(df_input, start_date, end_date, lookback_days, threshold):
        results = []
        # Core Model Loop
        target_dates = pd.date_range(start_date, end_date)
        progress_bar = st.progress(0)
        
        for i, current_date in enumerate(target_dates):
            # Update Progress Bar
            progress_bar.progress((i + 1) / len(target_dates))
            
            # Create rolling history
            history = df_input[(df_input['Leg Dep Date'] >= current_date - pd.Timedelta(days=lookback_days)) & 
                               (df_input['Leg Dep Date'] < current_date)].copy()
            
            if history.empty: continue

            # 1. DOW Multipliers (Fleet Level)
            dow_means = history.groupby('DOW')[['PPB_FJ_actual', 'PPB_Y_actual']].median()
            overall_median = history[['PPB_FJ_actual', 'PPB_Y_actual']].median()
            dow_multiplier = (dow_means / overall_median).fillna(1.0)

            # 2. Hierarchical Tables (L1 and L2)
            l1 = history.groupby(['Orig', 'Dest', 'Equip Actl'])[['PPB_FJ_actual', 'PPB_Y_actual', 'ShowUp_Rate_FJ', 'ShowUp_Rate_Y']].median().reset_index()
            l2 = history.groupby(['Equip Actl'])[['PPB_FJ_actual', 'PPB_Y_actual', 'ShowUp_Rate_FJ', 'ShowUp_Rate_Y']].median().reset_index()

            # 3. Predict for today's flights
            todays_flights = df_input[df_input['Leg Dep Date'] == current_date]
            
            for _, row in todays_flights.iterrows():
                # Skip invalid flights
                total_pax = row.get('F',0) + row.get('J',0) + row.get('Y',0)
                if total_pax < 10 or row['Bag Count L3E Actl Ttl'] == 0: continue
                
                # Lookup Logic
                match = l1[(l1['Orig'] == row['Orig']) & (l1['Dest'] == row['Dest']) & (l1['Equip Actl'] == row['Equip Actl'])]
                
                if not match.empty:
                    ppb_fj, ppb_y = match['PPB_FJ_actual'].values[0], match['PPB_Y_actual'].values[0]
                    su_fj, su_y = match['ShowUp_Rate_FJ'].values[0], match['ShowUp_Rate_Y'].values[0]
                else:
                    match_l2 = l2[l2['Equip Actl'] == row['Equip Actl']]
                    if not match_l2.empty:
                        ppb_fj, ppb_y = match_l2['PPB_FJ_actual'].values[0], match_l2['PPB_Y_actual'].values[0]
                        su_fj, su_y = match_l2['ShowUp_Rate_FJ'].values[0], match_l2['ShowUp_Rate_Y'].values[0]
                    else:
                        ppb_fj, ppb_y, su_fj, su_y = 12.0, 25.0, 0.95, 0.92

                # Apply DOW & Threshold
                if (row['DOW']) in dow_multiplier.index:
                    ppb_fj *= dow_multiplier.loc[row['DOW'], 'PPB_FJ_actual']
                    ppb_y *= dow_multiplier.loc[row['DOW'], 'PPB_Y_actual']

                # Calculate Predicted Containers
                p_prem = threshold_ceil(((row.get('F',0)+row.get('J',0))*su_fj)/ppb_fj, threshold)
                p_y = threshold_ceil((row.get('Y',0)*su_y)/ppb_y, threshold)
                
                # Squeeze Logic
                final_prediction = min(p_prem + p_y, row['MAX_LD3'])
                
                results.append({
                    'Date': current_date, 
                    'Flight': row['FLTNO'], 
                    'Orig': row['Orig'], 
                    'Dest': row['Dest'],
                    'Actual': row['Bag Count L3E Actl Ttl'], 
                    'Predicted': final_prediction
                })
                
        return pd.DataFrame(results)

    # --- THE MISSING "START" BUTTON ---
    if st.sidebar.button("Run Forecast Model"):
        if len(test_range) == 2:
            with st.spinner("Calculating Density & Optimizing Containers..."):
                # Run the math and save to session state
                st.session_state['res_df'] = run_model(df, test_range[0], test_range[1], lookback, user_threshold)
            st.success("Optimization Complete!")
        else:
            st.warning("Please select a valid Start and End date.")

# --- 5. UI & VISUALIZATION ---
# This part only runs if the button above has been clicked at least once
if st.session_state.res_df is not None and not st.session_state.res_df.empty:
    res_df = st.session_state.res_df
    
    st.divider()
    st.subheader("Interactive Flight Matrix")
    
    col_a, col_b = st.columns(2)
    
    # 1. Dynamic Origin Filter
    origins = sorted(res_df['Orig'].unique())
    sel_orig = col_a.selectbox("Filter Origin", origins)
    
    # 2. Dynamic Destination Filter
    available_dests = sorted(res_df[res_df['Orig'] == sel_orig]['Dest'].unique())
    sel_dest = col_b.selectbox("Filter Destination", available_dests)

    # 3. Filter Data
    view_df = res_df[(res_df['Orig'] == sel_orig) & (res_df['Dest'] == sel_dest)].copy()
    
    if not view_df.empty:
        # Metrics
        mae = (view_df['Actual'] - view_df['Predicted']).abs().mean()
        #bins_saved = (view_df['Predicted'] < view_df['Actual']).sum()
        
        st.header(f"Route: {sel_orig} ➔ {sel_dest}")
        c1, c2 = st.columns(2)
        c1.metric("MAE Accuracy", f"{round(mae, 2)}")
        c2.metric("Flights Analyzed", len(view_df))
        #c3.metric("AKE Slots Saved", int(bins_saved))

        # Pivot Table
        view_df['Display'] = view_df['Actual'].astype(int).astype(str) + " / " + view_df['Predicted'].astype(int).astype(str)
        pivot = view_df.pivot(index='Flight', columns='Date', values='Display').fillna("-")
        pivot.columns = [d.strftime('%b-%d') for d in pivot.columns]
        
        
        st.caption("Legend: First number = Actual LD3 / Second number = Predicted LD3")
        st.dataframe(pivot, use_container_width=True)
        
        # Chart
        st.line_chart(view_df.groupby('Date')[['Actual', 'Predicted']].sum())
    else:
        st.warning("No flights found for this route selection.")
        
elif st.session_state.res_df is None:
    st.info("👈 Please adjust settings in the Sidebar and click 'Run Forecast Model' to begin.")