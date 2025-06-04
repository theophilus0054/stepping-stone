import streamlit as st

st.set_page_config(page_title="Transportation Optimizer", layout="wide")

st.title("🚚 Transportation Problem Solver")
st.write("This app helps solve the Transportation Problem using the Least Cost Method and optimize it with the Stepping Stone Method.")

# Tabs for logical steps
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 Input", 
    "📉 Least Cost", 
    "🔁 Stepping Stone", 
    "✅ Final"
])

with tab1:
    st.header("Step 1: Input Problem")

    rows = st.number_input("Number of supply sources (rows)", min_value=1, value=3, key="rows")
    cols = st.number_input("Number of demand destinations (columns)", min_value=1, value=3, key="cols")

    st.subheader("Enter Cost Matrix")
    cost_matrix = []
    for i in range(rows):
        row = []
        cols_input = st.columns(cols)
        for j in range(cols):
            with cols_input[j]:
                value = st.number_input(f"Cost[{i},{j}]", value=0, key=f"cost_{i}_{j}")
                row.append(value)
        cost_matrix.append(row)

    st.subheader("Enter Supply and Demand")
    supply = []
    demand = []

    supply_cols = st.columns(rows)
    for i in range(rows):
        with supply_cols[i]:
            s = st.number_input(f"Supply[{i}]", value=0, key=f"supply_{i}")
            supply.append(s)

    demand_cols = st.columns(cols)
    for j in range(cols):
        with demand_cols[j]:
            d = st.number_input(f"Demand[{j}]", value=0, key=f"demand_{j}")
            demand.append(d)

    st.write("Cost Matrix:")
    st.table(cost_matrix)
    st.write("Supply:", supply)
    st.write("Demand:", demand)

with tab2:
    st.header("Step 2: Initial Feasible Solution (Least Cost Method)")
    st.info("🚧 This section will show the initial allocation matrix after applying the Least Cost Method. Work in progress.")

with tab3:
    st.header("Step 3: Optimize Using Stepping Stone Method")
    st.info("🚧 This section will show each iteration of optimization using the Stepping Stone Method. Work in progress.")

with tab4:
    st.header("Step 4: Final Optimized Transportation Table")
    st.info("🚧 Final result will be displayed here after optimization is complete. Work in progress.")
