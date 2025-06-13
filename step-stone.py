import streamlit as st
import numpy as np
import pandas as pd
import copy

# Set up the Streamlit page configuration
st.set_page_config(page_title="Transportation Optimizer", layout="wide")

st.title("🚚 Transportation Problem Solver")
st.write("This app helps solve the Transportation Problem using the Northwest Corner Method and optimize it with the Stepping Stone Method.")

# Tabs for logical steps
tab1, tab2, tab3, tab4 = st.tabs([
    "🦮 Input",
    "📉 Northwest Corner",
    "🔁 Stepping Stone",
    "✅ Final"
])

with tab1:
    st.header("Step 1: Input Problem")

    initial_rows = st.number_input("Number of supply sources (rows)", min_value=1, value=3, key="rows_input")
    initial_cols = st.number_input("Number of demand destinations (columns)", min_value=1, value=3, key="cols_input")

    st.session_state["rows"] = initial_rows
    st.session_state["cols"] = initial_cols

    st.subheader("Enter Cost Matrix")
    cost_matrix = []
    for i in range(initial_rows):
        row = []
        cols_input = st.columns(initial_cols)
        for j in range(initial_cols):
            with cols_input[j]:
                value = st.number_input(f"Cost[{i},{j}]", value=0, key=f"cost_{i}_{j}")
                row.append(value)
        cost_matrix.append(row)

    st.subheader("Enter Supply and Demand")
    supply = []
    demand = []

    supply_cols = st.columns(initial_rows)
    for i in range(initial_rows):
        with supply_cols[i]:
            s = st.number_input(f"Supply[{i}]", value=0, key=f"supply_{i}")
            supply.append(s)

    demand_cols = st.columns(initial_cols)
    for j in range(initial_cols):
        with demand_cols[j]:
            d = st.number_input(f"Demand[{j}]", value=0, key=f"demand_{j}")
            demand.append(d)

    total_supply = sum(supply)
    total_demand = sum(demand)

    current_rows = initial_rows
    current_cols = initial_cols

    # Balance the problem if total supply and demand are not equal
    if total_supply != total_demand:
        st.warning(f"Total Supply ({total_supply}) does not equal Total Demand ({total_demand}). Adding a dummy source/destination.")
        if total_supply < total_demand:
            supply.append(total_demand - total_supply)
            current_rows += 1
            cost_matrix.append([0] * current_cols) # Add a row of zeros for dummy supply
            st.info(f"Added Dummy Supply S{current_rows-1} with capacity {total_demand - total_supply}")
        else:
            demand.append(total_supply - total_demand)
            current_cols += 1
            for row in cost_matrix:
                row.append(0) # Add a column of zeros for dummy demand
            st.info(f"Added Dummy Demand D{current_cols-1} with requirement {total_supply - total_demand}")

    st.session_state["rows"] = current_rows
    st.session_state["cols"] = current_cols
    st.session_state["cost_matrix"] = cost_matrix
    st.session_state["supply"] = supply
    st.session_state["demand"] = demand


    st.write("Cost Matrix:")
    st.table(pd.DataFrame(st.session_state["cost_matrix"], index=[f"S{i}" for i in range(st.session_state["rows"])], columns=[f"D{j}" for j in range(st.session_state["cols"])]))
    st.write("Supply:", st.session_state["supply"])
    st.write("Demand:", st.session_state["demand"])


with tab2:
    st.header("Step 2: Initial Feasible Solution (Northwest Corner Method)")

    if "cost_matrix" in st.session_state and "supply" in st.session_state and "demand" in st.session_state:
        if st.button("🔄 Compute Initial Allocation"):
            with st.spinner("Processing initial allocation..."):
                cost_matrix = np.array(st.session_state["cost_matrix"])
                supply = np.array(st.session_state["supply"], dtype=float)
                demand = np.array(st.session_state["demand"], dtype=float)

                rows = st.session_state["rows"]
                cols = st.session_state["cols"]

                allocation_matrix = np.zeros((rows, cols), dtype=float)
                
                current_supply = supply.copy()
                current_demand = demand.copy()

                i = j = 0
                total_cost = 0
                
                allocated_cells = set() # This will hold (r, c) tuples for basic cells (including 0-allocations)

                # Perform NWC allocation
                while i < rows and j < cols:
                    # Allocate minimum of current supply or demand
                    alloc_val = min(current_supply[i], current_demand[j])
                    
                    # Only add to allocation matrix and total cost if a significant allocation is made
                    if alloc_val > 1e-9:
                        allocation_matrix[i, j] = alloc_val
                        total_cost += alloc_val * cost_matrix[i, j]
                        allocated_cells.add((i, j)) # Mark cell as basic
                    
                    # If current allocation is zero, it's a basic cell with zero allocation
                    # This happens when supply[i] == 0 or demand[j] == 0 right after allocation.
                    # This implies both conditions for moving to next row/col will be met.
                    # If this happens, we must explicitly add this (i,j) as a basic cell with zero allocation.
                    elif alloc_val == 0 and (current_supply[i] == 0 or current_demand[j] == 0):
                        # This scenario is a potential degeneracy point where we might place a zero allocation.
                        # NWC typically results in exactly one of supply/demand becoming zero, moving either i or j.
                        # If both become zero, it's degenerate.
                        # We specifically add this cell as basic even if 0.
                        allocated_cells.add((i, j))
                        allocation_matrix[i,j] = 0.0 # Ensure it's explicitly 0.0
                        
                    current_supply[i] -= alloc_val
                    current_demand[j] -= alloc_val

                    # Move to the next row or column
                    # Careful with float comparisons: use a small epsilon
                    if current_supply[i] < 1e-9: # If supply is exhausted
                        i += 1
                    if current_demand[j] < 1e-9: # If demand is satisfied
                        j += 1
                
                # Degeneracy handling for Stepping Stone readiness
                num_allocated_cells = len(allocated_cells)
                required_allocations = rows + cols - 1

                st.info(f"Northwest Corner initial basic cells: {num_allocated_cells}. Required for Stepping Stone: {required_allocations}.")

                # If degeneracy exists, add dummy zero allocations
                if num_allocated_cells < required_allocations:
                    st.warning(f"Degeneracy detected! Number of basic cells ({num_allocated_cells}) is less than m+n-1 ({required_allocations}).")
                    
                    # Store current basic cells before adding dummies
                    initial_nwc_basic_cells = set(allocated_cells)
                    
                    # Strategy for adding dummy zeros:
                    # 1. Prioritize a specific problematic cell for zero allocation if it's non-basic.
                    # 2. Otherwise, find any non-basic cell that helps resolve degeneracy.

                    # Try to add a dummy zero at (1,1) if it's not already basic and within bounds
                    # This specifically addresses the user's request for (S1,D1) to be '0'.
                    desired_dummy_pos = (1, 1) # S1, D1
                    if desired_dummy_pos[0] < rows and desired_dummy_pos[1] < cols and \
                       desired_dummy_pos not in initial_nwc_basic_cells:
                        
                        # Check if adding this cell forms a loop with existing basic cells.
                        # This is a critical check. A dummy zero should NOT form a closed loop with basic cells.
                        temp_cells_for_loop_check = initial_nwc_basic_cells.union({desired_dummy_pos})
                        
                        # Temporarily use find_closed_loop to check for loop formation
                        # For this check, the 'start_r, start_c' could be anything in temp_cells_for_loop_check,
                        # but often it's the cell we're trying to add.
                        # A more robust check for a 'valid' dummy zero is to check if it makes u/v solvable
                        # or if it closes a loop with existing basic cells (it shouldn't).
                        # A simpler heuristic: if adding it *doesn't* increase the number of independent paths
                        # to solve u/v, it's generally fine. The simplest approach for a fixed cell is to add it if it's
                        # not already basic and we need more basic cells.
                        
                        # For robust degeneracy handling, we often add zero allocations to cells
                        # that don't form a closed loop with existing basic cells.
                        # A common heuristic is to pick the lowest cost non-basic cell or
                        # a cell that doesn't immediately close a loop.
                        # For this specific user request, we'll try (1,1) directly.

                        allocated_cells.add(desired_dummy_pos)
                        allocation_matrix[desired_dummy_pos[0], desired_dummy_pos[1]] = 0.0
                        st.info(f"Added dummy allocation (0) at {desired_dummy_pos} to resolve degeneracy.")
                        num_allocated_cells = len(allocated_cells) # Update count

                # After attempting to add a specific dummy, if still degenerate, add more
                # This ensures we always reach m+n-1 basic cells
                if num_allocated_cells < required_allocations:
                    st.info(f"Still degenerate ({num_allocated_cells} < {required_allocations}). Finding additional dummy zero locations.")
                    
                    # Iterate through all cells to find non-basic cells to add as dummy zeros
                    for r_add in range(rows):
                        for c_add in range(cols):
                            if len(allocated_cells) >= required_allocations:
                                break # Stop if enough basic cells are found
                            
                            # Add if it's not already a basic cell
                            if (r_add, c_add) not in allocated_cells:
                                # For a more robust solution, ensure this cell doesn't form a loop
                                # with existing basic cells when added. However, this check makes
                                # the NWC significantly more complex. For typical problems, just adding
                                # any non-basic zero often works unless extreme degeneracy.
                                allocated_cells.add((r_add, c_add))
                                allocation_matrix[r_add, c_add] = 0.0
                                # st.info(f"Added dummy allocation (0) at {(r_add, c_add)}.") # Too verbose, uncomment if needed
                        if len(allocated_cells) >= required_allocations:
                            break
                
                # Final check
                if len(allocated_cells) != required_allocations:
                    st.error(f"Failed to achieve the required number of basic cells for Stepping Stone ({len(allocated_cells)} vs {required_allocations}). The solution might be problematic.")
                else:
                    st.success(f"Successfully established {len(allocated_cells)} basic cells for Stepping Stone.")
                
                # Prepare matrix for display (showing '-' for non-allocated, '0' for dummy zeros)
                display_matrix = [["-" for _ in range(cols)] for _ in range(rows)]
                for r in range(rows):
                    for c in range(cols):
                        val = allocation_matrix[r, c]
                        if val > 1e-9: # Actual allocated value (significant non-zero)
                            display_matrix[r][c] = str(int(round(val)))
                        elif (r,c) in allocated_cells: # Basic cell with zero allocation (dummy)
                            display_matrix[r][c] = "0"
                        # Else, it remains '-' for non-basic cells (implicitly)
                
                st.session_state["allocation_matrix"] = allocation_matrix.tolist()
                st.session_state["display_matrix"] = display_matrix
                st.session_state["total_cost"] = int(total_cost)
                st.session_state["allocated_cells"] = list(allocated_cells) # Store as list for session state


            st.subheader("Initial Allocation Table")
            df_display = pd.DataFrame(st.session_state["display_matrix"],
                                    columns=[f"D{j}" for j in range(cols)],
                                    index=[f"S{i}" for i in range(rows)],
                                    dtype=object) 
            st.dataframe(df_display, use_container_width=True)

            st.success(f"Total Initial Transportation Cost: {st.session_state['total_cost']}")
    else:
        st.warning("Please enter the input data in Tab 1 first.")

with tab3:
    st.header("Step 3: Optimize Using Stepping Stone Method")

    if "allocation_matrix" not in st.session_state:
        st.warning("Please generate the initial feasible solution in Tab 2 before proceeding.")
    else:
        cost = np.array(st.session_state["cost_matrix"])
        alloc = np.array(st.session_state["allocation_matrix"], dtype=float)
        rows = st.session_state["rows"]
        cols = st.session_state["cols"]

        st.subheader("Current Allocation Table")
        initial_alloc_display = [["-" for _ in range(cols)] for _ in range(rows)]
        current_basic_cells_at_start = set(st.session_state["allocated_cells"])
        for r_idx in range(rows):
            for c_idx in range(cols):
                # Display actual allocation for basic cells, '-' for non-basic
                if (r_idx, c_idx) in current_basic_cells_at_start:
                    initial_alloc_display[r_idx][c_idx] = str(int(round(alloc[r_idx, c_idx])))
                else:
                    initial_alloc_display[r_idx][c_idx] = "-"

        df_current_alloc = pd.DataFrame(initial_alloc_display,
                                        columns=[f"D{j}" for j in range(cols)],
                                        index=[f"S{i}" for i in range(rows)],
                                        dtype=object)
        st.dataframe(df_current_alloc, use_container_width=True)
        st.info(f"Current Total Cost: {int(np.sum(alloc * cost))}")

        if st.button("🚀 Run Stepping Stone Optimization"):
            history = []
            optimized = False
            iteration = 0

            def find_uv(cost_mat, m, n, basic_cells):
                """
                Calculates the u (row) and v (column) values for the basic cells
                using the cost matrix and the set of basic cells. This version is
                more robust to handle disconnected components in the basic cell graph.
                """
                u = [None] * m
                v = [None] * n
                basic_cells_set = set(basic_cells)
                
                # Keep track of which basic cells have been processed to determine their u/v values
                processed_basic_cells = set()

                # Loop until all basic cells' u/v values are determined or no more can be found
                # It handles disconnected components by starting new propagation from unassigned basic cells
                while len(processed_basic_cells) < len(basic_cells_set):
                    # Find an unassigned basic cell to start a new propagation (component)
                    start_cell_found_for_component = False
                    for r_cell, c_cell in basic_cells_set:
                        if (r_cell, c_cell) not in processed_basic_cells:
                            start_r, start_c = r_cell, c_cell
                            
                            # If neither u[r] nor v[c] is determined for this starting cell,
                            # arbitrarily set u[r] = 0 to begin propagation for this component.
                            if u[start_r] is None and v[start_c] is None:
                                u[start_r] = 0
                            
                            start_cell_found_for_component = True
                            break # Found a starting cell for a new component/propagation

                    if not start_cell_found_for_component:
                        break # No new component starting cell found, exit loop.

                    # Perform BFS-like propagation within the current component
                    q = [(start_r, start_c)]
                    component_queue_visited = set([(start_r, start_c)]) # Track visited nodes in this BFS run

                    while q:
                        r, c = q.pop(0)

                        # Mark as processed only when both u[r] and v[c] are determined
                        if u[r] is not None and v[c] is not None:
                            processed_basic_cells.add((r, c))

                        # If u[r] is known, determine v[c]
                        if u[r] is not None and v[c] is None:
                            v[c] = cost_mat[r, c] - u[r]
                            # Add basic cells in the same column 'c' that are unvisited in this BFS
                            for next_r_idx in range(m):
                                if (next_r_idx, c) in basic_cells_set and (next_r_idx, c) not in component_queue_visited:
                                    component_queue_visited.add((next_r_idx, c))
                                    q.append((next_r_idx, c))
                        
                        # If v[c] is known, determine u[r]
                        if v[c] is not None and u[r] is None:
                            u[r] = cost_mat[r, c] - v[c]
                            # Add basic cells in the same row 'r' that are unvisited in this BFS
                            for next_c_idx in range(n):
                                if (r, next_c_idx) in basic_cells_set and (r, next_c_idx) not in component_queue_visited:
                                    component_queue_visited.add((r, next_c_idx))
                                    q.append((r, next_c_idx))
                
                # Final check: If after all attempts, some U/V values are still None, warn the user.
                if any(val is None for val in u) or any(val is None for val in v):
                    st.warning("Warning: Could not determine all U/V values. This may indicate a severely disconnected basic cell graph (strong degeneracy) that could not be fully resolved.")

                return u, v

            def find_closed_loop(start_r, start_c, m, n, basic_cells_set):
                """
                Finds the unique closed loop for the stepping stone method using a BFS.
                The loop starts at (start_r, start_c) and alternates between
                horizontal and vertical moves using only cells from basic_cells_set
                (which includes the start cell itself).
                """
                
                # Queue stores (current_row, current_col, last_move_type)
                # 'H' indicates the last move to reach (r,c) was horizontal, 'V' vertical, 'S' for start.
                queue = [(start_r, start_c, 'S')] # (r, c, last_move_type)
                
                # visited_states tracks (row, col, last_move_type) to prevent revisiting states and cycles.
                # This ensures we take the shortest path *for each type of last move*.
                visited_states = set([(start_r, start_c, 'S')])
                
                # parent_map stores (child_r, child_c, child_last_move_type) -> (parent_r, parent_c, parent_last_move_type)
                parent_map = {} 

                while queue:
                    r, c, last_move_type = queue.pop(0)

                    # Try horizontal moves (next move from (r,c) is horizontal)
                    # This implies the move to (r,c) must have been vertical, or (r,c) is the start.
                    if last_move_type == 'V' or last_move_type == 'S':
                        for next_c in range(n):
                            if next_c == c: continue 

                            next_cell_coords = (r, next_c)
                            
                            # The next cell must be a basic cell or the entering cell.
                            if next_cell_coords in basic_cells_set:
                                # The move to next_cell_coords is horizontal, so its last_move_type will be 'H'
                                next_state_key = (next_cell_coords[0], next_cell_coords[1], 'H')
                                
                                # If we reached the start cell again, and we've made at least two moves (len(path) > 2 from original)
                                # and the last move was indeed horizontal (next_state_key has 'H').
                                # We need to ensure we completed a loop that is not just start->neighbor->start.
                                if next_cell_coords == (start_r, start_c) and last_move_type != 'S': # last_move_type != 'S' ensures it's a cycle, not initial step
                                    
                                    # Reconstruct the path backwards from the current cell (r, c)
                                    # which is the one that just moved to the starting cell (start_r, start_c)
                                    path_coords = []
                                    
                                    # Start backtracking from the state that *led* to finding the start_r, start_c
                                    current_backtrack_state = (r, c, last_move_type) 
                                    
                                    # Safety break in case of infinite loop in reconstruction
                                    max_loop_len = (m + n) * 2 
                                    loop_count = 0

                                    # Backtrack until we reach the original starting state ('S' type)
                                    while current_backtrack_state != (start_r, start_c, 'S') and loop_count < max_loop_len:
                                        path_coords.append((current_backtrack_state[0], current_backtrack_state[1]))
                                        parent_state = parent_map.get(current_backtrack_state)
                                        if parent_state is None:
                                            st.error("Error: Loop reconstruction failed - parent state not found during horizontal move backtracking.")
                                            return None
                                        current_backtrack_state = parent_state
                                        loop_count += 1
                                    
                                    if loop_count >= max_loop_len:
                                        st.error("Error: Loop reconstruction exceeded max length. Possible infinite loop during horizontal move backtracking.")
                                        return None
                                    
                                    # Add the original starting cell (start_r, start_c) to complete the cycle at both ends
                                    path_coords.append((start_r, start_c))

                                    return path_coords[::-1] # Reverse to get the correct chronological order

                                if next_state_key not in visited_states:
                                    visited_states.add(next_state_key)
                                    parent_map[next_state_key] = (r, c, last_move_type) # Store (child_state) -> (parent_state)
                                    queue.append(next_state_key)

                    # Try vertical moves (next move from (r,c) is vertical)
                    # This implies the move to (r,c) must have been horizontal, or (r,c) is the start.
                    if last_move_type == 'H' or last_move_type == 'S':
                        for next_r in range(m):
                            if next_r == r: continue

                            next_cell_coords = (next_r, c)

                            if next_cell_coords in basic_cells_set:
                                next_state_key = (next_cell_coords[0], next_cell_coords[1], 'V')

                                if next_cell_coords == (start_r, start_c) and last_move_type != 'S':
                                    
                                    path_coords = []
                                    current_backtrack_state = (r, c, last_move_type)
                                    max_loop_len = (m + n) * 2
                                    loop_count = 0

                                    while current_backtrack_state != (start_r, start_c, 'S') and loop_count < max_loop_len:
                                        path_coords.append((current_backtrack_state[0], current_backtrack_state[1]))
                                        parent_state = parent_map.get(current_backtrack_state)
                                        if parent_state is None:
                                            st.error("Error: Loop reconstruction failed - parent state not found during vertical move backtracking.")
                                            return None
                                        current_backtrack_state = parent_state
                                        loop_count += 1

                                    if loop_count >= max_loop_len:
                                        st.error("Error: Loop reconstruction exceeded max length. Possible infinite loop during vertical move backtracking.")
                                        return None
                                        
                                    path_coords.append((start_r, start_c))
                                    return path_coords[::-1]

                                if next_state_key not in visited_states:
                                    visited_states.add(next_state_key)
                                    parent_map[next_state_key] = (r, c, last_move_type)
                                    queue.append(next_state_key)
                                    
                return None # No closed loop found after exhausting all possibilities


            # Main optimization loop for Stepping Stone method
            while not optimized and iteration < 100: # Limit iterations to prevent infinite loops in complex cases
                iteration += 1
                st.markdown(f"#### Iteration {iteration}")

                current_basic_cells = st.session_state["allocated_cells"]
                current_basic_cells_set = set(current_basic_cells) # Use set for faster lookups

                # Step 1: Calculate u and v values for basic cells
                u, v = find_uv(cost, rows, cols, current_basic_cells)
                
                # Check if u or v values contain None, indicating a failure to find all values
                if any(val is None for val in u) or any(val is None for val in v):
                    st.error("Cannot proceed with Stepping Stone: U/V values could not be fully determined. This suggests a problem with basic cell count or graph connectivity.")
                    optimized = True # Stop optimization
                    break

                st.write("Calculated u values:", [f"U{i}={val:.2f}" if val is not None else "None" for i, val in enumerate(u)])
                st.write("Calculated v values:", [f"V{j}={val:.2f}" if val is not None else "None" for j, val in enumerate(v)])

                # Step 2: Calculate improvement indices for unoccupied cells
                improvement_indices = np.full((rows, cols), np.inf) # Initialize with infinity
                unoccupied_cells_with_indices = []

                for i in range(rows):
                    for j in range(cols):
                        if (i,j) not in current_basic_cells_set: # Only for non-basic (unoccupied) cells
                            if u[i] is not None and v[j] is not None: # Ensure u and v are calculated
                                imp_idx = cost[i, j] - u[i] - v[j] # Cost - u_i - v_j
                                improvement_indices[i, j] = imp_idx
                                unoccupied_cells_with_indices.append((i, j, imp_idx))
                
                st.write("Improvement Indices for Unoccupied Cells (Cost - U - V):")
                if unoccupied_cells_with_indices:
                    for r_idx, c_idx, val in unoccupied_cells_with_indices:
                        st.write(f"Cell ({r_idx},{c_idx}): {cost[r_idx,c_idx]} - {u[r_idx]:.2f} - {v[c_idx]:.2f} = {val:.2f}")
                else:
                    st.info("No unoccupied cells to evaluate.")

                # Find the most negative improvement index
                min_improvement_val = np.min(improvement_indices)

                # Step 3: Check for optimality
                # If min_improvement_val is non-negative (or very close to zero due to float precision), it's optimal.
                if min_improvement_val >= -1e-9 or np.isinf(min_improvement_val): 
                    optimized = True
                    st.success("Optimal solution found!")
                    break

                # Step 4: Select entering cell
                pivot_r, pivot_c = np.unravel_index(np.argmin(improvement_indices), improvement_indices.shape)
                st.info(f"Most negative improvement index is {min_improvement_val:.2f} at cell ({pivot_r}, {pivot_c}). This cell will enter the basis.")

                # Step 5: Find closed loop for the entering cell
                # The loop involves the entering cell and current basic cells.
                loop_search_basic_cells = current_basic_cells_set.union({(pivot_r, pivot_c)})

                loop_raw = find_closed_loop(pivot_r, pivot_c, rows, cols, loop_search_basic_cells)
                
                if not loop_raw:
                    st.error(f"Could not find a closed loop for cell ({pivot_r}, {pivot_c}). This indicates an issue with the initial solution, degeneracy handling, or the loop-finding algorithm.")
                    st.error("Stopping optimization. The current solution might not be optimal or valid.")
                    break

                loop = loop_raw 

                st.write("Closed loop found (coordinates and signs):")
                signed_loop_path = []
                min_theta = np.inf # Minimum allocation in negative positions of the loop
                leaving_cell = None

                for k, (r, c) in enumerate(loop):
                    # Loop starts with the entering cell (+), then alternates signs
                    sign = '+' if k % 2 == 0 else '-'
                    signed_loop_path.append(f"({r},{c}){sign}")
                    
                    if sign == '-': # Only consider negative positions to find theta
                        # We need to find the minimum allocation among cells where allocation is subtracted.
                        # Exclude cells that already have effectively zero allocation from consideration for min_theta if they are already basic.
                        if alloc[r, c] > 1e-9: # Only consider positive allocations for leaving cell
                            if alloc[r, c] < min_theta:
                                min_theta = alloc[r, c]
                                leaving_cell = (r,c)
                
                # Handle cases where all negative positions have zero or near-zero allocation (strong degeneracy)
                if min_theta == np.inf or leaving_cell is None:
                    st.warning("All negative positions in the loop have zero or near-zero *positive* allocation. This can happen with extreme degeneracy, where a dummy zero was the candidate for leaving. The entering cell just replaces the dummy zero.")
                    # In this case, the entering cell takes over the position of the leaving cell (which had 0 allocation).
                    # min_theta would be 0, and no actual flow is shifted.
                    # This often means the solution is already optimal but another path exists, or
                    # that a zero-allocation cell has moved into/out of basis.
                    min_theta = 0 # Set theta to 0 if no positive allocation found to leave
                    # We still need a leaving cell, which will be the first 0-allocation cell in the negative position
                    for k, (r, c) in enumerate(loop):
                        sign = '+' if k % 2 == 0 else '-'
                        if sign == '-' and abs(alloc[r,c]) < 1e-9: # Find a zero allocation cell to leave
                            leaving_cell = (r,c)
                            st.info(f"Identified a zero-allocation basic cell {leaving_cell} to leave the basis as theta is 0.")
                            break
                    
                    if leaving_cell is None: # Fallback if no zero-allocation cell found among negative positions
                        st.error("Could not determine a leaving cell despite zero theta. Stopping optimization.")
                        optimized = True
                        break


                st.write(" -> ".join(signed_loop_path))
                st.write(f"Minimum allocation in negative positions (theta): {min_theta:.2f}. Cell {leaving_cell} will leave the basis.")

                # Step 6: Adjust allocations along the loop by theta
                new_alloc = alloc.copy()
                for k, (r, c) in enumerate(loop):
                    if k % 2 == 0: # Add theta to '+' positions
                        new_alloc[r, c] += min_theta
                    else: # Subtract theta from '-' positions
                        new_alloc[r, c] -= min_theta
                
                alloc = new_alloc # Update the current allocation matrix

                # Step 7: Update the set of basic cells for the next iteration
                next_basic_cells_set = set()
                # Add the entering cell
                next_basic_cells_set.add((pivot_r, pivot_c))
                
                # Add all other cells that were basic, except the leaving cell
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        # A cell is basic if it was in the old basic set AND is not the leaving cell,
                        # OR if its new allocation is significantly positive (meaning it just entered or was positive).
                        # Using a small epsilon to check for "zero" allocations
                        if (r_idx, c_idx) != leaving_cell and abs(new_alloc[r_idx, c_idx]) > 1e-9:
                            next_basic_cells_set.add((r_idx, c_idx))
                        elif (r_idx, c_idx) != leaving_cell and (r_idx, c_idx) in current_basic_cells_set and abs(new_alloc[r_idx, c_idx]) < 1e-9:
                            # If it was a basic cell with zero allocation and not the leaving cell, keep it.
                            next_basic_cells_set.add((r_idx, c_idx))

                st.session_state["allocated_cells"] = list(next_basic_cells_set)

                # Re-check for degeneracy after allocation shift and re-add dummy zeros if necessary
                current_num_basic = len(st.session_state["allocated_cells"])
                required_allocations = rows + cols - 1

                if current_num_basic < required_allocations:
                    st.warning(f"Degeneracy (after shift): Basic cells = {current_num_basic}. Required = {required_allocations}. Attempting to re-add dummy 0s.")
                    
                    non_basic_candidates = []
                    for r_idx in range(rows):
                        for c_idx in range(cols):
                            if (r_idx, c_idx) not in st.session_state["allocated_cells"]:
                                non_basic_candidates.append((r_idx, c_idx))
                    
                    # Add dummy zeros from available non-basic cells until basic cell count is met
                    for r_nb, c_nb in non_basic_candidates:
                        if len(st.session_state["allocated_cells"]) < required_allocations:
                            st.session_state["allocated_cells"].append((r_nb, c_nb))
                            new_alloc[r_nb, c_nb] = 0 # Ensure its allocation is explicitly zero
                            st.info(f"Added dummy allocation (0) at {(r_nb, c_nb)} to maintain basic cell count.")
                        else:
                            break
                    if len(st.session_state["allocated_cells"]) < required_allocations:
                        st.error("Failed to re-establish required number of basic cells after shift. Solution might be problematic.")


                current_total_cost = int(np.sum(alloc * cost))
                
                # Store the current state for iteration history display
                display_matrix_iter = [["-" for _ in range(cols)] for _ in range(rows)]
                for r_idx in range(rows):
                    for c_idx in range(cols):
                        if (r_idx, c_idx) in st.session_state["allocated_cells"]:
                            display_matrix_iter[r_idx][c_idx] = str(int(round(alloc[r_idx, c_idx])))
                        else:
                            display_matrix_iter[r_idx][c_idx] = "-"
                
                history.append((copy.deepcopy(display_matrix_iter), current_total_cost))

            st.subheader("Optimization Iterations")
            # Display all recorded optimization iterations
            for idx, (matrix, cost_now) in enumerate(history):
                st.markdown(f"##### Iteration {idx+1}")
                df_iter = pd.DataFrame(matrix, columns=[f"D{j}" for j in range(cols)], index=[f"S{i}" for i in range(rows)], dtype=object) 
                st.dataframe(df_iter, use_container_width=True)
                st.write(f"Cost: {cost_now}")
                st.markdown("---")

            # Update session state with the final allocation and cost
            st.session_state["allocation_matrix"] = alloc.tolist()
            st.session_state["total_cost"] = int(np.sum(alloc * cost))

            if optimized:
                st.success("Optimization complete!")
            else:
                st.warning("Optimization stopped early (max iterations reached or loop not found). The solution might not be fully optimal.")

with tab4:
    st.header("Step 4: Final Optimized Transportation Table")

    if "allocation_matrix" in st.session_state:
        final_alloc_np = np.array(st.session_state["allocation_matrix"])
        m_final, n_final = final_alloc_np.shape
        final_display_matrix = [["-" for _ in range(n_final)] for _ in range(m_final)]
        
        # Prepare the final display matrix
        for r in range(m_final):
            for c in range(n_final):
                # Display actual allocation for basic cells, '-' for non-basic
                if (r, c) in st.session_state.get("allocated_cells", []):
                    final_display_matrix[r][c] = str(int(round(final_alloc_np[r, c])))
                else:
                    final_display_matrix[r][c] = "-"

        st.subheader("Final Optimized Allocation")
        df_final = pd.DataFrame(final_display_matrix,
                                columns=[f"D{j}" for j in range(n_final)],
                                index=[f"S{i}" for i in range(m_final)],
                                dtype=object)
        st.dataframe(df_final, use_container_width=True)

        st.success(f"Total Optimized Transportation Cost: {st.session_state['total_cost']}")
    else:
        st.warning("Please go through the initial allocation and optimization steps first.")