# --- START OF FILE data_acquisition.py ---

import serial
import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SensorDataReader:
    def __init__(self, port='COM4', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        # Add 'sensor_point_id' to columns
        self.data = pd.DataFrame(columns=['timestamp', 'tooth_id', 'sensor_point_id', 'force', 'contact_time'])
        self.is_connected = False

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.is_connected = True
            logging.info(f"Connected to sensor on {self.port}")
        except serial.SerialException as e:
            logging.error(f"Failed to connect to sensor: {e}")
            self.is_connected = False

    def read_data(self, duration=5):
        if not self.is_connected:
            logging.warning("No sensor connected. Using simulated data.")
            # Pass num_sensor_points_per_tooth to simulate_data if needed, or use a default
            return self.simulate_data(duration=duration, num_teeth=16, num_sensor_points_per_tooth=4) 

        start_time = time.time()
        data_list = []
        while time.time() - start_time < duration:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if line:
                    try:
                        # Assuming real sensor data now includes sensor_point_id
                        # timestamp, tooth_id, sensor_point_id, force, contact_time
                        parts = list(map(float, line.split(',')))
                        if len(parts) == 5:
                            timestamp, tooth_id, sensor_point_id, force, contact_time = parts
                            data_list.append({
                                'timestamp': timestamp,
                                'tooth_id': int(tooth_id),
                                'sensor_point_id': int(sensor_point_id), # New
                                'force': force,
                                'contact_time': contact_time
                            })
                        else:
                             logging.warning(f"Invalid data line (expected 5 parts): {line}")
                    except ValueError as e:
                        logging.warning(f"Invalid data format: {line}, error: {e}")
                time.sleep(0.01)
            except serial.SerialException as e:
                logging.error(f"Serial read error: {e}")
                break
        if data_list:
            new_data = pd.DataFrame(data_list)
            self.data = pd.concat([self.data, new_data], ignore_index=True) if not self.data.empty else new_data
        return self.data

    # In class SensorDataReader, update simulate_data method:
    def simulate_data(self, duration=5, num_teeth=16, num_sensor_points_per_tooth=4):
        timestamps = np.arange(0, duration, 0.1) # Keep 0.1s interval for enough data points
        data_list = []
        for t in timestamps:
            for tooth_id in range(1, num_teeth + 1):
                # Base force for the entire tooth at this timestamp (can vary smoothly or randomly)
                base_force_on_tooth = np.random.uniform(0, 70) * (1 + 0.5 * np.sin(t + tooth_id)) # Add some variation over time
                
                for sensor_point_idx in range(1, num_sensor_points_per_tooth + 1):
                    # Simulate gradient: some points higher, some lower, can depend on sensor_point_idx
                    # Example: one point (e.g., center) might consistently read higher/lower
                    # or a simple random variation around the base_force
                    variation = np.random.uniform(-15, 30) # Wider variation for distinct points
                    if sensor_point_idx == 1: # Make one point often higher
                        variation += np.random.uniform(5,15)
                    
                    force = base_force_on_tooth + variation
                    force = max(0, min(100, force))  # Clamp force between 0 and 100
                    contact_time = np.random.uniform(0.01, 0.05)
                    
                    data_list.append({
                        'timestamp': t,
                        'tooth_id': tooth_id,
                        'sensor_point_id': sensor_point_idx, # Store sensor point identifier
                        'force': force,
                        'contact_time': contact_time
                    })
        sim_data = pd.DataFrame(data_list)
        # Concatenate if self.data exists and has the same columns, otherwise assign
        if not self.data.empty and set(self.data.columns) == set(sim_data.columns):
            self.data = pd.concat([self.data, sim_data], ignore_index=True)
        else:
            if not self.data.empty and set(self.data.columns) != set(sim_data.columns):
                logging.warning("Column mismatch between existing data and new sim_data. Replacing self.data.")
            self.data = sim_data
        logging.info(f"Generated simulated data: {len(sim_data)} rows, {num_teeth} teeth, {num_sensor_points_per_tooth} sensor points/tooth.")
        return self.data

    def save_data(self, filename='sensor_data.csv'):
        if not self.data.empty:
            self.data.to_csv(filename, index=False)
            logging.info(f"Data saved to {filename}")

    def close(self):
        if self.serial and self.is_connected:
            self.serial.close()
            self.is_connected = False
            logging.info("Sensor connection closed")

if __name__ == '__main__':
    reader = SensorDataReader(port='COM4', baudrate=115200)
    # reader.connect() # Not connecting for simulation
    data = reader.simulate_data(duration=1, num_teeth=2, num_sensor_points_per_tooth=3) # Test simulation
    print(data.head(10))
    reader.save_data('simulated_intra_tooth_sensor_data.csv')
    # reader.close()
# --- END OF FILE data_acquisition.py ---