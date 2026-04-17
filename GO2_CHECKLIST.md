# Go2 Session Startup Checklist

Run through this every time you work with the robot. Don't skip steps.

## Pre-session
- [ ] Go2 battery charged
- [ ] Phone: Unitree mobile app fully closed (force-quit)
- [ ] Laptop charged or plugged in
- [ ] Tape measure + notebook handy (if recording)

## Connect
1. Power on Go2 — wait 45s for full boot, watch LEDs (should not blink red)
2. Laptop: connect to "Unitree_Go..." Wi-Fi (password on robot sticker)
3. Verify connection with: `ping -c 4 192.168.12.1` — expect 0% packet loss

## Launch SDK
Terminal A (leave running entire session):

    export ROBOT_IP=192.168.12.1
    export CONN_TYPE=webrtc
    ros2 launch go2_robot_sdk robot.launch.py

Wait ~30s for WebRTC handshake.

## Verify LiDAR
Terminal B:

    ros2 topic hz /point_cloud2

Expect ~7 Hz. Ctrl+C after confirmed.

## If recording rosbags
Terminal B:

    cd ~/autopass/bags
    ros2 bag record -o run_NN /point_cloud2 /imu /odom

Replace NN with run number (01, 02, 03...). Ctrl+C when done.
Verify with: `ros2 bag info run_NN`

## Shutdown
- [ ] Ctrl+C all terminals
- [ ] Power off Go2 (hold power button)
- [ ] Upload any new bags to Google Drive IMMEDIATELY
- [ ] git status and commit any code changes

## Known issues
- **Red LED fault state:** Power cycle. Document in case a pattern emerges.
- **ros2 topic hz shows nothing:** Mobile app still connected, or WebRTC didn't handshake. Relaunch Terminal A.
- **rviz2 shows topic but no points:** Wrong Fixed Frame. Try base_link → odom → utlidar_lidar.
