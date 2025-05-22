# --- START OF FILE data_processing.py ---
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, data):
        self.data = data # This should be a Pandas DataFrame
        self.cleaned_data = None
        self.force_matrix = None 
        self.timestamps = None
        self.tooth_ids = None # Unique tooth IDs [1, 2, ...]
        self.num_sensor_points_per_tooth_map = {} # tooth_id -> num_sensor_points for that tooth
        self.ordered_tooth_sensor_pairs = [] # Canonical list of (tooth_id, sensor_point_id) defining matrix columns

    def clean_data(self):
        if not isinstance(self.data, pd.DataFrame):
            logging.error("Input data is not a Pandas DataFrame.")
            # Attempt to convert if it's a list of dicts, or handle error
            if isinstance(self.data, list) and self.data and isinstance(self.data[0], dict):
                self.data = pd.DataFrame(self.data)
            else:
                self.cleaned_data = pd.DataFrame() # Empty DataFrame
                return self.cleaned_data

        # Ensure 'sensor_point_id' exists
        if 'sensor_point_id' not in self.data.columns:
            logging.warning("'sensor_point_id' not in data, adding default value 1 for all rows.")
            self.data['sensor_point_id'] = 1
        
        required_cols = ['timestamp', 'tooth_id', 'sensor_point_id', 'force', 'contact_time']
        if not all(col in self.data.columns for col in required_cols):
            logging.error(f"Data is missing one or more required columns: {required_cols}")
            self.cleaned_data = pd.DataFrame()
            return self.cleaned_data

        self.cleaned_data = self.data.dropna(subset=required_cols)
        
        # Convert types robustly
        try:
            self.cleaned_data['timestamp'] = pd.to_numeric(self.cleaned_data['timestamp'])
            self.cleaned_data['tooth_id'] = pd.to_numeric(self.cleaned_data['tooth_id']).astype(int)
            self.cleaned_data['sensor_point_id'] = pd.to_numeric(self.cleaned_data['sensor_point_id']).astype(int)
            self.cleaned_data['force'] = pd.to_numeric(self.cleaned_data['force'])
            self.cleaned_data['contact_time'] = pd.to_numeric(self.cleaned_data['contact_time'])
        except Exception as e:
            logging.error(f"Error converting data types: {e}")
            self.cleaned_data = pd.DataFrame()
            return self.cleaned_data

        self.cleaned_data = self.cleaned_data[
            (self.cleaned_data['force'] >= 0) &
            (self.cleaned_data['contact_time'] >= 0)
        ]
        
        self.cleaned_data = self.cleaned_data.drop_duplicates(
            subset=['timestamp', 'tooth_id', 'sensor_point_id'], keep='last'
        )
        
        self.tooth_ids = sorted(self.cleaned_data['tooth_id'].unique())
        
        self.ordered_tooth_sensor_pairs = []
        if not self.cleaned_data.empty:
            # Determine number of sensor points for each tooth_id
            self.num_sensor_points_per_tooth_map = self.cleaned_data.groupby('tooth_id')['sensor_point_id'].nunique().to_dict()
            
            # Create the ordered list of (tooth_id, sensor_point_id) pairs
            # This ensures a consistent column order in the force_matrix
            for tid in self.tooth_ids: # Iterate through sorted unique tooth_ids
                # Get sensor points for this specific tooth_id, sorted
                sensor_points_for_this_tooth = sorted(
                    self.cleaned_data[self.cleaned_data['tooth_id'] == tid]['sensor_point_id'].unique()
                )
                for sp_id in sensor_points_for_this_tooth:
                     self.ordered_tooth_sensor_pairs.append((tid, sp_id))
        else:
            self.num_sensor_points_per_tooth_map = {}


        num_unique_pairs = len(self.ordered_tooth_sensor_pairs)
        logging.info("Data cleaned: %d rows, %d unique teeth. Total unique (tooth, sensor_point) pairs: %d",
                     len(self.cleaned_data), len(self.tooth_ids), num_unique_pairs)
        return self.cleaned_data

    def create_force_matrix(self):
        if self.cleaned_data is None or self.cleaned_data.empty: # Ensure clean_data ran and produced data
            self.clean_data()
            if self.cleaned_data.empty:
                logging.error("Cleaned data is empty. Cannot create force matrix.")
                self.force_matrix = np.array([])
                self.timestamps = []
                return self.force_matrix, self.timestamps
        
        self.timestamps = sorted(self.cleaned_data['timestamp'].unique())
        
        if not self.ordered_tooth_sensor_pairs or not self.timestamps:
            logging.warning("No ordered_tooth_sensor_pairs or timestamps. Force matrix will be empty.")
            self.force_matrix = np.array([])
            return self.force_matrix, self.timestamps
            
        # Initialize force matrix with NaNs or zeros. NaNs can help identify missing data.
        self.force_matrix = np.full((len(self.timestamps), len(self.ordered_tooth_sensor_pairs)), np.nan)
        
        pair_to_col_idx = {pair: i for i, pair in enumerate(self.ordered_tooth_sensor_pairs)}

        # More efficient way to fill matrix using pandas:
        # Pivot table: index=timestamp, columns=(tooth_id, sensor_point_id), values=force
        try:
            pivot_df = self.cleaned_data.pivot_table(
                index='timestamp',
                columns=['tooth_id', 'sensor_point_id'],
                values='force'
            )
            # Reindex pivot_df to match self.timestamps and self.ordered_tooth_sensor_pairs for consistency
            # Important: columns in pivot_df are MultiIndex. We need to map them to self.ordered_tooth_sensor_pairs.
            
            # Ensure pivot_df columns are tuples (tooth_id, sensor_point_id)
            pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)

            # Reindex rows to match self.timestamps, filling missing timestamps with NaN rows
            pivot_df = pivot_df.reindex(self.timestamps)
            
            # Create a new DataFrame with columns in the exact order of self.ordered_tooth_sensor_pairs
            # Fill missing (tooth_id, sensor_point_id) columns for some timestamps with NaN
            temp_matrix_df = pd.DataFrame(index=self.timestamps, columns=pd.MultiIndex.from_tuples(self.ordered_tooth_sensor_pairs))
            
            # Fill temp_matrix_df with values from pivot_df
            # This handles cases where pivot_df might not have all pairs for all timestamps
            for pair in self.ordered_tooth_sensor_pairs:
                if pair in pivot_df.columns:
                    temp_matrix_df[pair] = pivot_df[pair]
            
            self.force_matrix = temp_matrix_df.to_numpy()
            # Replace any remaining NaNs if desired (e.g., with 0)
            # self.force_matrix = np.nan_to_num(self.force_matrix, nan=0.0) 
            # For visualization, keeping NaNs can be useful to show "no reading"

        except Exception as e:
            logging.error(f"Error creating force matrix with pivot_table: {e}. Falling back to slower loop.")
            # Fallback to iterative filling (slower)
            for row_idx, t in enumerate(self.timestamps):
                time_data_for_ts = self.cleaned_data[self.cleaned_data['timestamp'] == t]
                for _, row_data_point in time_data_for_ts.iterrows():
                    pair = (int(row_data_point['tooth_id']), int(row_data_point['sensor_point_id']))
                    if pair in pair_to_col_idx:
                        col_idx = pair_to_col_idx[pair]
                        self.force_matrix[row_idx, col_idx] = row_data_point['force']
            # self.force_matrix = np.nan_to_num(self.force_matrix, nan=0.0) # if using np.nan for init

        logging.info("Force matrix created: shape=%s (timestamps x (tooth_id, sensor_point_id) pairs)", self.force_matrix.shape)
        return self.force_matrix, self.timestamps

    def get_forces_for_sensor_point(self, tooth_id, sensor_point_id):
        if self.force_matrix is None or self.force_matrix.size == 0:
            self.create_force_matrix()
        if self.force_matrix.size == 0: return [], []

        target_pair = (tooth_id, sensor_point_id)
        try:
            col_idx = self.ordered_tooth_sensor_pairs.index(target_pair)
            forces = self.force_matrix[:, col_idx]
            return self.timestamps, np.nan_to_num(forces, nan=0.0) # Replace NaN with 0 for plotting
        except ValueError:
            logging.warning(f"Sensor point ({tooth_id}, {sensor_point_id}) not found in ordered pairs.")
            return self.timestamps, np.zeros(len(self.timestamps) if self.timestamps else 0)

    def get_average_force_for_tooth(self, tooth_id):
        if self.force_matrix is None or self.force_matrix.size == 0:
            self.create_force_matrix()
        if self.force_matrix.size == 0 or tooth_id not in self.tooth_ids: return [], []

        relevant_cols_indices = [
            i for i, (tid, _) in enumerate(self.ordered_tooth_sensor_pairs) if tid == tooth_id
        ]
        if not relevant_cols_indices:
            return self.timestamps, np.zeros(len(self.timestamps) if self.timestamps else 0)

        tooth_forces_subset = self.force_matrix[:, relevant_cols_indices]
        # Calculate mean ignoring NaNs, then convert resulting NaNs (if all inputs were NaN) to 0
        avg_forces = np.nanmean(tooth_forces_subset, axis=1)
        return self.timestamps, np.nan_to_num(avg_forces, nan=0.0)
        
    def get_all_forces_at_time(self, timestamp):
        """Returns a list of forces corresponding to self.ordered_tooth_sensor_pairs."""
        if self.force_matrix is None or self.force_matrix.size == 0:
            self.create_force_matrix()
        if self.force_matrix.size == 0 or not self.timestamps:
             return self.ordered_tooth_sensor_pairs, [] 

        timestamps_array = np.array(self.timestamps)
        if timestamps_array.size == 0:
            return self.ordered_tooth_sensor_pairs, np.zeros(len(self.ordered_tooth_sensor_pairs))

        time_idx = np.argmin(np.abs(timestamps_array - timestamp))
        forces = self.force_matrix[time_idx, :]
        # It's good practice to replace NaNs for visualization if they mean "no force"
        return self.ordered_tooth_sensor_pairs, np.nan_to_num(forces, nan=0.0)


if __name__ == '__main__':
    from data_acquisition import SensorDataReader
    reader = SensorDataReader()
    data = reader.simulate_data(duration=1, num_teeth=2, num_sensor_points_per_tooth=3)
    
    processor = DataProcessor(data.copy()) # Pass a copy to avoid modifying original in tests
    cleaned = processor.clean_data()
    print("Cleaned Data Sample:\n", cleaned.head())
    print("Unique Tooth IDs:", processor.tooth_ids)
    print("Sensor points per tooth map:", processor.num_sensor_points_per_tooth_map)
    print("Ordered tooth-sensor pairs (first 10):", processor.ordered_tooth_sensor_pairs[:10])

    fm, ts = processor.create_force_matrix()
    if fm.size > 0:
        print("Force Matrix Shape:", fm.shape)
        print("Timestamps (first 5):", ts[:5])

        if processor.tooth_ids:
            avg_t, avg_f = processor.get_average_force_for_tooth(processor.tooth_ids[0])
            print(f"Avg force for tooth {processor.tooth_ids[0]} (first 5): {avg_f[:5]}")

            if processor.ordered_tooth_sensor_pairs:
                 p_tid, p_spid = processor.ordered_tooth_sensor_pairs[0]
                 sp_t, sp_f = processor.get_forces_for_sensor_point(p_tid, p_spid)
                 print(f"Force for sensor point ({p_tid},{p_spid}) (first 5): {sp_f[:5]}")
        
        if ts:
            pairs_at_t0, forces_at_t0 = processor.get_all_forces_at_time(ts[0])
            print(f"\nAll forces at time {ts[0]} (first 5 pairs):")
            for p, f in zip(pairs_at_t0[:5], forces_at_t0[:5]):
                print(f"  Pair {p}: Force {f:.2f}")
    else:
        print("Force matrix is empty or could not be created.")

# --- END OF FILE data_processing.py ---