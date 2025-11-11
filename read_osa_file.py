 #!/usr/bin/env python3
"""
Script to read and analyze an ORB_SLAM3 .osa map atlas file

Usage:
    python read_osa_file.py <path_to_osa_file>

Example:
    python read_osa_file.py /home/yinzi/universal_manipulation_interface/example_demo_session/demos/mapping/map_atlas.osa
"""

import sys
import os
import struct
import argparse
import binascii
import numpy as np
from collections import defaultdict


def print_hex_view(data, offset=0, width=16):
    """
    Print a hexdump-like view of binary data
    """
    for i in range(0, len(data), width):
        row_data = data[i:i+width]
        hex_view = ' '.join(f'{b:02x}' for b in row_data)
        ascii_view = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in row_data)
        
        print(f"{offset+i:08x}  {hex_view:<{width*3}}  |{ascii_view}|")


def analyze_osa_file(file_path):
    """
    Analyze the structure of an ORB_SLAM3 .osa file
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"File size: {len(data)} bytes")
        
        # Check header
        if b'serialization::archive' in data[:50]:
            print("File appears to be a serialized archive")
            header_end = data.find(b'\0', data.find(b'serialization::archive'))
            header = data[:header_end].decode('utf-8', errors='replace')
            print(f"Header: {header}")
        
        # Print first 200 bytes for analysis
        print("\nFirst 200 bytes:")
        print_hex_view(data[:200])
        
        # Try to identify strings in the file
        strings = []
        current_string = ""
        for i, byte in enumerate(data):
            if 32 <= byte <= 126:  # printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) > 3:  # Only keep strings with reasonable length
                    strings.append((i - len(current_string), current_string))
                current_string = ""
        
        if len(current_string) > 3:
            strings.append((len(data) - len(current_string), current_string))
        
        print("\nIdentified strings:")
        for offset, string in strings[:50]:  # Show only first 50 strings
            print(f"Offset: {offset:08x}, String: '{string}'")
        
        if len(strings) > 50:
            print(f"...and {len(strings) - 50} more strings")
        
        # Try to find float arrays or other structured data
        print("\nPotential data structures:")
        # Look for sequences of float-like values
        for i in range(0, len(data) - 16, 4):
            # Check if we have a sequence of valid-looking floats
            try:
                floats = [struct.unpack('<f', data[j:j+4])[0] for j in range(i, i+16, 4)]
                # Check if these look like reasonable coordinate values
                if all(-100 < f < 100 for f in floats) and any(abs(f) > 0.01 for f in floats):
                    print(f"Potential float array at offset {i:08x}: {floats}")
            except:
                pass
        
        # Statistics on data
        byte_freq = defaultdict(int)
        for byte in data:
            byte_freq[byte] += 1
        
        print("\nByte frequency statistics:")
        most_common = sorted(byte_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for byte, count in most_common:
            percentage = (count / len(data)) * 100
            print(f"Byte 0x{byte:02x}: {count} occurrences ({percentage:.2f}%)")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Analyze ORB_SLAM3 .osa map atlas files')
    parser.add_argument('file_path', help='Path to the .osa file')
    parser.add_argument('--full', action='store_true', help='Perform full analysis (may be slow for large files)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        return 1
    
    print(f"Analyzing file: {args.file_path}")
    analyze_osa_file(args.file_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())