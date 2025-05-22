# --- START OF FILE data_processing.py ---
import numpy as np
import logging
import matplotlib.pyplot as plt 
import pandas as pd
from data_acquisition import SensorDataReader

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

    def clean_data(self):
        if not isinstance(self.data, pd.DataFrame):
            logging.error("Input data is not a Pandas DataFrame.")
            self.cleaned_data = pd.DataFrame()
            return self.cleaned_data
        
        required_cols = ['timestamp', 'tooth_id', 'sensor_point_id', 'force', 'contact_time']
        if not all(col in self.data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.data.columns]
            logging.error(f"Data is missing required columns: {missing}. Expecting {required_cols}")
            self.cleaned_data = pd.DataFrame()
            return self.cleaned_data

        self.cleaned_data = self.data.dropna(subset=required_cols).copy() # Use .copy() to avoid SettingWithCopyWarning
        try:
            self.cleaned_data['timestamp'] = pd.to_numeric(self.cleaned_data['timestamp'])
            self.cleaned_data['tooth_id'] = pd.to_numeric(self.cleaned_data['tooth_id']).astype(int)
            self.cleaned_data['sensor_point_id'] = pd.to_numeric(self.cleaned_data['sensor_point_id']).astype(int)
            # Ensure 'force' is float from the start
            self.cleaned_data['force'] = pd.to_numeric(self.cleaned_data['force'], errors='coerce').astype(float)
            self.cleaned_data['contact_time'] = pd.to_numeric(self.cleaned_data['contact_time'], errors='coerce').astype(float)
        except Exception as e:
            logging.error(f"Error converting data types: {e}"); self.cleaned_data = pd.DataFrame(); return self.cleaned_data

        self.cleaned_data.dropna(subset=['force', 'contact_time'], inplace=True) # Drop rows where conversion to float failed for force/contact_time

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
                 self.max_force_overall = self.cleaned_data['force'].max() if self.cleaned_data['force'].max() > 0 else 100.0
        else: self.num_sensor_points_per_tooth_map = {}
        
        logging.info("Data cleaned: %d rows, %d unique teeth. Total (tooth, sensor_point) pairs: %d. Max force: %.1f",
                     len(self.cleaned_data), len(self.tooth_ids), len(self.ordered_tooth_sensor_pairs), self.max_force_overall)
        return self.cleaned_data

    def create_force_matrix(self):
        if self.cleaned_data is None or self.cleaned_data.empty: self.clean_data()
        if self.cleaned_data.empty:
            logging.error("Cleaned data empty. Cannot create force matrix."); self.force_matrix = np.array([]); self.timestamps = []; return self.force_matrix, self.timestamps
        
        self.timestamps = sorted(self.cleaned_data['timestamp'].unique())
        if not self.ordered_tooth_sensor_pairs or not self.timestamps:
            logging.warning("No sensor pairs or timestamps. Force matrix empty."); self.force_matrix = np.array([]); return self.force_matrix, self.timestamps
            
        # Initialize with float dtype explicitly
        self.force_matrix = np.full((len(self.timestamps), len(self.ordered_tooth_sensor_pairs)), np.nan, dtype=float)
        try:
            # Ensure 'force' column is float before pivot
            pivot_data = self.cleaned_data[['timestamp', 'tooth_id', 'sensor_point_id', 'force']].copy()
            pivot_data['force'] = pd.to_numeric(pivot_data['force'], errors='coerce').astype(float)

            pivot_df = pivot_data.pivot_table(index='timestamp', columns=['tooth_id', 'sensor_point_id'], values='force')
            pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)
            
            temp_matrix_df = pd.DataFrame(index=self.timestamps, columns=pd.MultiIndex.from_tuples(self.ordered_tooth_sensor_pairs), dtype=float) # Ensure dtype
            temp_matrix_df.update(pivot_df) 
            self.force_matrix = temp_matrix_df.to_numpy(dtype=float) # Ensure final numpy array is float
        except Exception as e:
            logging.error(f"Pivot_table error for force matrix: {e}. Using loop fallback.")
            pair_to_col_idx = {pair: i for i, pair in enumerate(self.ordered_tooth_sensor_pairs)}
            for row_idx, t in enumerate(self.timestamps):
                time_data = self.cleaned_data[self.cleaned_data['timestamp'] == t]
                for _, data_point in time_data.iterrows():
                    pair = (int(data_point['tooth_id']), int(data_point['sensor_point_id']))
                    if pair in pair_to_col_idx: 
                        self.force_matrix[row_idx, pair_to_col_idx[pair]] = float(data_point['force']) # Ensure float assignment
        
        logging.info("Force matrix: shape=%s (timestamps x (tooth,sensor) pairs), dtype=%s", 
                     self.force_matrix.shape, self.force_matrix.dtype)
        return self.force_matrix, self.timestamps

    def get_average_force_for_tooth(self, tooth_id):
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size == 0 or tooth_id not in self.tooth_ids: 
            return self.timestamps if self.timestamps else [], np.array([], dtype=float) # Return empty float array
        
        indices = [i for i, (tid, _) in enumerate(self.ordered_tooth_sensor_pairs) if tid == tooth_id]
        if not indices: 
            return self.timestamps if self.timestamps else [], np.array([], dtype=float)
        
        # Ensure force_matrix is float before nanmean
        if self.force_matrix.dtype != float and self.force_matrix.dtype != np.float64 and self.force_matrix.dtype != np.float32 :
            logging.warning(f"Force matrix dtype is {self.force_matrix.dtype}, attempting to cast to float for nanmean.")
            try:
                fm_float = self.force_matrix.astype(float)
            except ValueError:
                logging.error("Could not cast force_matrix to float. Returning zeros for average.")
                return self.timestamps if self.timestamps else [], np.zeros(len(self.timestamps or []), dtype=float)
        else:
            fm_float = self.force_matrix
            
        avg_forces = np.nanmean(fm_float[:, indices], axis=1)
        return self.timestamps, np.nan_to_num(avg_forces, nan=0.0).astype(float) # Ensure float output
        
    def get_all_forces_at_time(self, timestamp):
        if self.force_matrix is None: self.create_force_matrix()
        if self.force_matrix.size == 0 or not self.timestamps: 
            return self.ordered_tooth_sensor_pairs, np.array([], dtype=float)
        
        ts_array = np.array(self.timestamps)
        if ts_array.size == 0: 
            return self.ordered_tooth_sensor_pairs, np.zeros(len(self.ordered_tooth_sensor_pairs), dtype=float)
        
        time_idx = np.argmin(np.abs(ts_array - timestamp))
        forces = self.force_matrix[time_idx, :]
        return self.ordered_tooth_sensor_pairs, np.nan_to_num(forces, nan=0.0).astype(float)

if __name__ == '__main__':
    reader = SensorDataReader()
    data = reader.simulate_data(duration=0.2, num_teeth=2, num_sensor_points_per_tooth=2)
    processor = DataProcessor(data.copy())
    processor.create_force_matrix() # This will call clean_data
    print("Force matrix dtype:", processor.force_matrix.dtype)
    if processor.timestamps and processor.tooth_ids:
        print("Max force overall:", processor.max_force_overall)
        t0 = processor.timestamps[0]
        pairs, forces = processor.get_all_forces_at_time(t0)
        print(f"Forces at t={t0} (dtype: {forces.dtype}):")
        for p, f in zip(pairs, forces): print(f"  {p}: {f:.2f}")
        avg_t, avg_f = processor.get_average_force_for_tooth(processor.tooth_ids[0])
        print(f"Avg force tooth {processor.tooth_ids[0]} (dtype: {avg_f.dtype}): {avg_f}")
# --- END OF FILE data_processing.py ---