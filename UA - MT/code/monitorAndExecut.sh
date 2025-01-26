#!/bin/bash

# Specify the PID of the process to monitor
pid_to_monitor=3183686

# Monitor the process status continuously
while true; do
    # Check if the process with the specified PID is still running
    if ! ps -p $pid_to_monitor > /dev/null; then
        echo "Process with PID $pid_to_monitor has finished"
        
        # Execute the command you want when the process finishes
        echo "nohup python3 train_LA_meanteacher_certainty.py --exp UAMT-MT-BAM-LABEL > uamtbamlabel.out &"
        # Replace the command below with the command you want to execute
        # For example: echo "Process finished" | mail -s "Notification" user@example.com
        break  # Exit the loop once the process finishes
    fi
    sleep 60  # Wait for 1 second before checking again
done

