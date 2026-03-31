import depthai as dai
import time

with dai.Device() as device:
    device.startIMUFirmwareUpdate()
    print("Updating IMU firmware, do NOT disconnect...")
    
    while True:
        isDone, pct = device.getIMUFirmwareUpdateStatus()
        print(f"IMU firmware update: {pct:.0f}%", end="\r")
        if isDone:
            print("\nUpdate complete!")
            break
        time.sleep(0.5)