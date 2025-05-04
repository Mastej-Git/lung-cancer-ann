import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import serial 
import csv
import sys
import argparse

class CubeBalanceController:
    def __init__(self, model_path, serial_port=None, baud_rate=115200):
        """
        Initialize the cube balancing controller.   
        Args:
            model_path: Path to the .keras model file
            serial_port: Optional serial port to communicate with the robot hardware
            baud_rate: Baud rate for serial communication
        """
    
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully")
    
        self.model.summary()
    
        self.serial_conn = None
        if serial_port:
            try:
                self.serial_conn = serial.Serial(serial_port, baud_rate)
                print(f"Connected to {serial_port} at {baud_rate} baud")
            except serial.SerialException as e:
                print(f"Error connecting to serial port: {e}")
    
        self.latest_data = None
    
        self.column_names = [
            't', 
            'ox', 'oy', 'oz', 
            'sx', 'sy', 'sz', 
            'wx', 'wy', 'wz', 
            'cx', 'cy', 'cz'  
        ]
    
        self.feature_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
        
    def preprocess_data(self, raw_data):
        """
        Preprocess the raw data to match the model's expected input format.
        
        Args:
            raw_data: A list or array of data values
            
        Returns:
            Preprocessed data suitable for model input
        """
    
        features = np.array([float(raw_data[i]) for i in self.feature_indices])
    
        features = features.reshape(1, -1)
        
        return features
    
    def parse_data_line(self, line):
        """
        Parse a line of data from the sensor or data source.
        
        Args:
            line: A string containing sensor data values separated by tabs or spaces
            
        Returns:
            A list of parsed values
        """
    
        values = line.strip().split()
        
        parsed_values = []
        for val in values:
            try:
                parsed_values.append(float(val))
            except ValueError:
                parsed_values.append(val)    

        return parsed_values
    
    def predict_control(self, input_data):
        """
        Use the model to predict control outputs.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Control output for the balancing motor
        """
    
        prediction = self.model.predict(input_data, verbose=0)
    
        control_output = prediction[0][-1] if prediction.ndim > 1 else prediction[0]
        
        return control_output
    
    def send_control_to_motor(self, control_value):
        """
        Send the control value to the motor.
        
        Args:
            control_value: The calculated control value to send
        """
        if self.serial_conn:
            command = f"CONTROL:{control_value:.4f}\n"
            self.serial_conn.write(command.encode())
            print(f"Sent control value: {control_value:.4f}")
        else:
            print(f"Control value (no hardware connected): {control_value:.4f}")
    
    def process_realtime_data(self, data_line):
        """
        Process a line of real-time data.
        
        Args:
            data_line: A string containing a line of sensor data
            
        Returns:
            The calculated control value
        """
    
        parsed_data = self.parse_data_line(data_line)
        self.latest_data = parsed_data
    
        processed_input = self.preprocess_data(parsed_data)
    
        control_value = self.predict_control(processed_input)
        
        return control_value
    
    def run_from_stdin(self):
        """
        Run the controller by reading data from standard input.
        This is useful for piping data from another process.
        """
        print("Reading data from stdin. Press Ctrl+C to exit.")
        try:
            for line in sys.stdin:
                if line.strip():
                    control_value = self.process_realtime_data(line)
                    self.send_control_to_motor(control_value)
        except KeyboardInterrupt:
            print("Controller stopped by user")
        finally:
            if self.serial_conn:
                self.serial_conn.close()
    
    def run_from_file(self, file_path, delay=0.1):
        """
        Run the controller by reading data from a file, 
        simulating real-time data with the specified delay.
        
        Args:
            file_path: Path to the data file
            delay: Delay between processing lines (in seconds)
        """
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
            
                if file_path.endswith('.csv'):
                    next(reader, None)
                
                print(f"Reading data from {file_path}. Press Ctrl+C to exit.")
                for row in reader:
                    line = '\t'.join(row)
                    control_value = self.process_realtime_data(line)
                    self.send_control_to_motor(control_value)
                    time.sleep(delay)
        except KeyboardInterrupt:
            print("Controller stopped by user")
        finally:
            if self.serial_conn:
                self.serial_conn.close()

def main():
    parser = argparse.ArgumentParser(description='Cube Balance Controller')
    parser.add_argument('--model', type=str, required=True, help='Path to the .keras model file')
    parser.add_argument('--port', type=str, help='Serial port for hardware communication')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate for serial communication')
    parser.add_argument('--file', type=str, help='Input data file (if not using stdin)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between readings when using file input')
    
    args = parser.parse_args()
    
    controller = CubeBalanceController(args.model, args.port, args.baud)
    
    if args.file:
        controller.run_from_file(args.file, args.delay)
    else:
        controller.run_from_stdin()

if __name__ == "__main__":
    main()