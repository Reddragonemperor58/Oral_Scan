# --- START OF FILE data_processing.py ---
import numpy as np
import logging
import pandas as pd
# No matplotlib import needed here unless for standalone testing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, data):
        self.data = data 
        self.cleaned_data = None
        self.force_matrix = None 
        self.timestamps = None
        self.tooth_ids = None 
        self.num_sensor_points_per_tooth_map = {} 
        self.ordered_tooth_sensor_pairs = [] 
        self.max_force_overall = 100.0
        self.cof_trajectory = [] # New: To store (timestamp, cof_x, cof_y)

    def clean_data(self):
        # ... (same as before) ...
        if not isinstance(self.data, pd.DataFrame): logging.error("Input data is not a Pandas DataFrame."); self.cleaned_data = pd.DataFrame(); return self.cleaned_data
        required_cols = ['timestamp', 'tooth_id', 'sensor_point_id', 'force', 'contact_time']
        if not all(col in self.data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data.columns]; logging.error(f"Data missing: {missing}"); self.cleaned_data = pd.DataFrame(); return self.cleaned_data
        self.cleaned_data = self.data.dropna(subset=required_cols).copy()
        try:
            self.cleaned_data['timestamp'] = pd.to_numeric(self.cleaned_data['timestamp'])
            self.cleaned_data['tooth_id'] = pd.to_numeric(self.cleaned_data['tooth_id']).astype(int)
            self.cleaned_data['sensor_point_id'] = pd.to_numeric(self.cleaned_data['sensor_point_id']).astype(int)
            self.cleaned_data['force'] = pd.to_numeric(self.cleaned_data['force'], errors='coerce').astype(float)
            self.cleaned_data['contact_time'] = pd.to_numeric(self.cleaned_data['contact_time'], errors='coerce').astype(float)
        except Exception as e: logging.error(f"Type conversion error: {e}"); self.cleaned_data = pd.DataFrame(); return self.cleaned_data
        self.cleaned_data.dropna(subset=['force', 'contact_time'], inplace=True)
        self.cleaned_data = self.cleaned_data[(self.cleaned_data['force'] >= 0) & (self.cleaned_data['contact_time'] >= 0)]
        self.cleaned_data = self.cleaned_data.drop_duplicates(subset=['timestamp', 'tooth_id', 'sensor_point_id'], keep='last')
        self.tooth_ids = sorted(self.cleaned_data['tooth_id'].unique())
        self.ordered_tooth_sensor_pairs = []
        if not self.cleaned_data.empty:
            self.num_sensor_points_per_tooth_map = self.cleaned_data.groupby('tooth_id')['sensor_point_id'].nunique().to_dict()
            for tid in self.tooth_ids:
                sensor_points_for_this_tooth = sorted(self.cleaned_data[self.cleaned_data['tooth_id'] == tid]['sensor_point_id'].unique())
                for sp_id in sensor_points_for_this_tooth: self.ordered_tooth_sensor_pairs.append((tid, sp_id))
            if 'force' in self.cleaned_data.columns and not self.cleaned_data['force'].empty:
                 valid_forces = self.cleaned_data['force'][self.cleaned_data['force'] > 0]
                 self.max_force_overall = valid_forces.max() if not valid_forces.empty else 100.0
        else: self.num_sensor_points_per_tooth_map = {}
        logging.info("Data cleaned: %d rows, %d unique teeth. Pairs: %d. Max force: %.1f",
                     len(self.cleaned_data), len(self.tooth_ids), len(self.ordered_tooth_sensor_pairs), self.max_force_overall)
        return self.cleaned_data

    def create_force_matrix(self):
        # ... (same as before) ...
        if self.cleaned_data is None or self.cleaned_data.empty: self.clean_data()
        if self.cleaned_data.empty: logging.error("Cleaned data empty."); self.force_matrix=np.array([]); self.timestamps=[]; return self.force_matrix, self.timestamps
        self.timestamps = sorted(self.cleaned_data['timestamp'].unique())
        if not self.ordered_tooth_sensor_pairs or not self.timestamps: logging.warning("No pairs/timestamps."); self.force_matrix=np.array([]); return self.force_matrix, self.timestamps
        self.force_matrix = np.full((len(self.timestamps), len(self.ordered_tooth_sensor_pairs)), np.nan, dtype=float)
        try:
            pivot_data = self.cleaned_data[['timestamp', 'tooth_id', 'sensor_point_id', 'force']].copy()
            pivot_data['force'] = pd.to_numeric(pivot_data['force'], errors='coerce').astype(float)
            pivot_df = pivot_data.pivot_table(index='timestamp', columns=['tooth_id', 'sensor_point_id'], values='force')
            pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
            temp_matrix_df = pd.DataFrame(index=self.timestamps, columns=pd.MultiIndex.from_tuples(self.ordered_tooth_sensor_pairs), dtype=float)
            temp_matrix_df.update(pivot_df) 
            self.force_matrix = temp_matrix_df.to_numpy(dtype=float)
        except Exception as e:
            logging.error(f"Pivot error: {e}. Fallback."); pair_to_col_idx = {p:i for i,p in enumerate(self.ordered_tooth_sensor_pairs)}
            for r,t in enumerate(self.timestamps):
                td = self.cleaned_data[self.cleaned_data['timestamp']==t]
                for _,dp in td.iterrows():
                    p=(int(dp['tooth_id']), int(dp['sensor_point_id'])); self.force_matrix[r,pair_to_col_idx[p]]=float(dp['force'])
        logging.info("Force matrix: shape=%s, dtype=%s", self.force_matrix.shape, self.force_matrix.dtype)
        return self.force_matrix, self.timestamps

    def get_average_force_for_tooth(self, tooth_id):
        # ... (same as before) ...
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size == 0 or tooth_id not in self.tooth_ids: return self.timestamps or [], np.array([], dtype=float)
        indices = [i for i, (tid, _) in enumerate(self.ordered_tooth_sensor_pairs) if tid == tooth_id]
        if not indices: return self.timestamps or [], np.array([], dtype=float)
        fm_float = self.force_matrix.astype(float) if self.force_matrix.dtype!=float else self.force_matrix
        avg_forces = np.nanmean(fm_float[:, indices], axis=1)
        return self.timestamps, np.nan_to_num(avg_forces, nan=0.0).astype(float)
        
    def get_all_forces_at_time(self, timestamp):
        # ... (same as before) ...
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size == 0 or not self.timestamps: return self.ordered_tooth_sensor_pairs, np.array([], dtype=float)
        ts_array = np.array(self.timestamps); time_idx = np.argmin(np.abs(ts_array - timestamp))
        forces = self.force_matrix[time_idx, :]
        return self.ordered_tooth_sensor_pairs, np.nan_to_num(forces, nan=0.0).astype(float)

    # --- NEW METHOD for COF ---
    def calculate_cof_trajectory(self, tooth_cell_definitions, num_sensor_points_per_tooth_layout=4):
        """
        Calculates the Center of Force (COF) trajectory.
        Needs layout information to know the (x,y) of each sensor point.
        Args:
            tooth_cell_definitions (dict): From DentalArchVisualizer, mapping layout_idx to 
                                           {'center':(x,y), 'width':w, 'height':h, 'actual_id': id}
            num_sensor_points_per_tooth_layout (int): How sensor points are arranged (e.g., 4 for 2x2).
        """
        if self.force_matrix is None or self.force_matrix.size == 0:
            logging.warning("Force matrix not available for COF calculation.")
            self.cof_trajectory = []
            return

        self.cof_trajectory = []
        grid_dim = int(np.sqrt(num_sensor_points_per_tooth_layout))
        if grid_dim == 0 : grid_dim = 1 # Avoid division by zero if layout is 1 point

        for ts_idx, timestamp in enumerate(self.timestamps):
            sum_force_x = 0
            sum_force_y = 0
            total_force_this_step = 0
            
            # Get forces for all sensor points at this timestamp
            current_pairs, current_forces = self.get_all_forces_at_time(timestamp)
            force_map_this_ts = {pair: force for pair, force in zip(current_pairs, current_forces)}

            for layout_idx, cell_prop in tooth_cell_definitions.items():
                tooth_id = cell_prop['actual_id']
                cell_center_x, cell_center_y = cell_prop['center']
                cell_w, cell_h = cell_prop['width'], cell_prop['height']
                
                # Get sensor point IDs for this tooth from data
                actual_sp_ids_for_tooth = sorted([spid for tid, spid in self.ordered_tooth_sensor_pairs if tid == tooth_id])

                sub_cell_w = cell_w / grid_dim
                sub_cell_h = cell_h / grid_dim

                for r in range(grid_dim): # row in sub-grid within cell
                    for c in range(grid_dim): # col in sub-grid
                        sp_layout_idx = r * grid_dim + c # 0, 1, 2, 3 for 2x2

                        if sp_layout_idx < len(actual_sp_ids_for_tooth):
                            sensor_point_id = actual_sp_ids_for_tooth[sp_layout_idx]
                            force = force_map_this_ts.get((tooth_id, sensor_point_id), 0.0)

                            if force > 1e-3: # Consider only significant forces
                                # Calculate (x,y) of this sensor point's sub-cell center
                                # (r=0 is top row of sub-cells)
                                sp_x = cell_center_x - cell_w/2 + sub_cell_w/2 + c * sub_cell_w
                                sp_y = cell_center_y + cell_h/2 - sub_cell_h/2 - r * sub_cell_h
                                
                                sum_force_x += force * sp_x
                                sum_force_y += force * sp_y
                                total_force_this_step += force
            
            if total_force_this_step > 1e-3:
                cof_x = sum_force_x / total_force_this_step
                cof_y = sum_force_y / total_force_this_step
                self.cof_trajectory.append((timestamp, cof_x, cof_y))
            # else:
                # No significant force, COF is undefined or could be last known, or (0,0)
                # self.cof_trajectory.append((timestamp, np.nan, np.nan)) # Or skip
        
        logging.info(f"Calculated COF trajectory with {len(self.cof_trajectory)} points.")

    def get_cof_up_to_timestamp(self, current_timestamp):
        """Returns COF points up to (and including) the given timestamp."""
        if not self.cof_trajectory:
            return []
        
        # Find all COF points whose timestamp is <= current_timestamp
        valid_cof_points = []
        for ts, x, y in self.cof_trajectory:
            if ts <= current_timestamp + 1e-6: # Add small epsilon for float comparison
                valid_cof_points.append((x,y)) # Only x,y for drawing line
            else:
                break # Assuming trajectory is sorted by timestamp
        return valid_cof_points

if __name__ == '__main__':
    # ... (Updated __main__ to test COF if desired) ...
    reader = SensorDataReader()
    data = reader.simulate_data(duration=1, num_teeth=4, num_sensor_points_per_tooth=4)
    processor = DataProcessor(data.copy())
    processor.create_force_matrix()

    # For COF, DataProcessor needs the layout. This is usually done by the Visualizer.
    # For a standalone test here, we'd need to mock `tooth_cell_definitions`.
    # Example:
    mock_layout = {
        0: {'center': (-2, 2), 'width': 1, 'height': 1, 'actual_id': 1},
        1: {'center': ( 2, 2), 'width': 1, 'height': 1, 'actual_id': 2},
        2: {'center': (-2,-2), 'width': 1, 'height': 1, 'actual_id': 3},
        3: {'center': ( 2,-2), 'width': 1, 'height': 1, 'actual_id': 4},
    }
    # Filter mock_layout to only include teeth present in processor.tooth_ids
    # and map layout index to actual_id correctly.
    # This is complex for a simple test. COF is best tested via the visualizer.
    # processor.calculate_cof_trajectory(mock_layout_filtered_and_mapped) 
    # print("COF Trajectory:", processor.cof_trajectory)
    print("DataProcessor standalone test completed.")

# --- END OF FILE data_processing.py ---