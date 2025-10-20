import csv
import io
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def generate_lpips_plot(csv_data_string):
    """
    Parses CSV data, filters for relevant video and packet loss conditions,
    calculates average LPIPS, and generates a grouped bar chart.

    Args:
        csv_data_string (str): A string containing the CSV data.
                               The first line should be the header.
    """

    # Data structure to hold LPIPS values before averaging:
    # { "video-0.mp4": {"0%_loss": [values], "20%_loss": [values]}, ... } # Updated key
    parsed_data = defaultdict(lambda: {"0%_loss": [], "20%_loss": []}) # Updated key

    # Target videos (video-0.mp4 to video-8.mp4)
    target_video_names = [
      "video-0.mp4", "video-1.mp4", "video-2.mp4", 
      "video-4.mp4", "video-5.mp4", "video-6.mp4", 
      "video-7.mp4", "video-9.mp4"
    ]

    # Use csv.reader to handle CSV parsing
    # Convert the string data to a file-like object
    csvfile = io.StringIO(csv_data_string)
    reader = csv.reader(csvfile)

    try:
        header = next(reader) # Get the header row
    except StopIteration:
        print("Error: CSV data is empty.")
        return

    # Determine column indices from header (more robust)
    try:
        lpips_col_idx = header.index("lpips")
        avg_loss_col_idx = header.index("avg_packet_loss")
        video_col_idx = header.index("video")
    except ValueError:
        print("Error: CSV header does not contain required columns: 'lpips', 'avg_packet_loss', 'video'.")
        print(f"Found header: {header}")
        # As a fallback, assuming fixed indices if header check fails (less robust)
        lpips_col_idx = 6
        avg_loss_col_idx = 8
        video_col_idx = 13
        print(f"Warning: Using fallback column indices: lpips={lpips_col_idx}, avg_packet_loss={avg_loss_col_idx}, video={video_col_idx}")


    # Process each row in the CSV data
    for row_number, row in enumerate(reader, 1): # Start row count from 1 for data rows
        if not row: # Skip empty rows
            continue
        
        try:
            # Ensure video_name is stripped of potential extra quotes or spaces
            video_name = row[video_col_idx].strip().replace('"', '')


            if video_name in target_video_names:
                lpips = float(row[lpips_col_idx])
                # Ensure avg_packet_loss is stripped of potential extra quotes or spaces before conversion
                avg_loss_str = row[avg_loss_col_idx].strip().replace('"', '')
                avg_loss = float(avg_loss_str)

                # Categorize by packet loss
                if avg_loss == 0.0:
                    parsed_data[video_name]["0%_loss"].append(lpips)
                elif 0.15 <= avg_loss <= 0.25: # Changed: 20% +/- 5%
                    parsed_data[video_name]["20%_loss"].append(lpips) # Updated key

        except IndexError:
            print(f"Warning: Row {row_number+1} has an unexpected number of columns. Skipping: {row}")
        except ValueError as e:
            print(f"Warning: Could not parse data in row {row_number+1}. Error: {e}. Skipping: {row}")
        except Exception as e:
            print(f"An unexpected error occurred processing row {row_number+1}: {e}. Skipping: {row}")


    # Prepare data for plotting (calculate averages)
    plot_data = {
        "video_labels": [],
        "lpips_0_loss_avg": [],
        "lpips_20_loss_avg": [] # Updated key
    }

    # Ensure videos are processed in the order of target_video_names for consistent plotting
    for video_name in target_video_names:
        # Generate user-friendly labels like "Video 0", "Video 1", ...
        video_number = video_name.replace(".mp4", "").split('-')[-1]  # Extract number from e.g. video-0.mp4
        plot_data["video_labels"].append(f"Video{video_number}")



        data_0_loss = parsed_data[video_name]["0%_loss"]
        if data_0_loss:
            plot_data["lpips_0_loss_avg"].append(np.mean(data_0_loss))
        else:
            plot_data["lpips_0_loss_avg"].append(0) # Use 0 if no data (or np.nan to show gaps)
            print(f"Info: No data for {video_name} with 0% packet loss.")

        data_20_loss = parsed_data[video_name]["20%_loss"] # Updated key
        if data_20_loss:
            plot_data["lpips_20_loss_avg"].append(np.mean(data_20_loss)) # Updated key
        else:
            plot_data["lpips_20_loss_avg"].append(0) # Use 0 if no data # Updated key
            print(f"Info: No data for {video_name} with ~20% packet loss.")
            
    if not any(plot_data["lpips_0_loss_avg"]) and not any(plot_data["lpips_20_loss_avg"]): # Updated key
        print("Error: No data found for the specified videos and packet loss conditions. Cannot generate plot.")
        return

    # --- Plotting ---
    num_videos = len(plot_data["video_labels"])
    x_indices = np.arange(num_videos)  # the label locations
    bar_width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8)) # Increased figure size

    # Bars for 0% packet loss
    rects1 = ax.bar(x_indices - bar_width/2, plot_data["lpips_0_loss_avg"], bar_width, 
                    label='0% Packet Loss', color='royalblue')
    # Bars for ~20% packet loss
    rects2 = ax.bar(x_indices + bar_width/2, plot_data["lpips_20_loss_avg"], bar_width, # Updated key
                    label='20% Packet Loss', color='sandybrown') # Updated label

    # Add labels, title, and legend
    ax.set_xlabel('Video Sequence', fontsize=20, labelpad=10)
    ax.set_ylabel('LPIPS', fontsize=20, labelpad=10)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(plot_data["video_labels"], rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis='y', labelsize=16)

    ax.legend(fontsize=20)

    # Add value labels on top of each bar
    ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=11) # Format to 4 decimal places
    ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=11)

    # Improve aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Add horizontal grid lines
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f')) # Format y-axis ticks

    fig.tight_layout() # Adjust layout to make room for rotated x-axis labels and title
    output_base = "lpips_vs_video_packet_loss_20pct"
    # To save the figure:
    try:
        

        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_base}.pdf", dpi=300, bbox_inches='tight')

        print(f"\nPlot generated and saved as '{output_base}.png' and '{output_base}.pdf'")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    
    # To display the figure (e.g., in a Jupyter notebook or if running script locally):
    plt.show()


# --- Main part of the script ---
if __name__ == "__main__":
    # Define the path to the CSV file
    csv_file_path = "/Users/tron/RealTron/UIUC/25Spring/CS538/Project/pretty-plots/video_loss/GenStream.csv"
    
    try:
        # Attempt to open and read the CSV file
        with open(csv_file_path, "r", encoding='utf-8') as f: # Added encoding for wider compatibility
            csv_data_string_from_file = f.read()
        
        # Check if data was actually read
        if not csv_data_string_from_file.strip():
            print(f"Error: The file {csv_file_path} is empty or contains only whitespace.")
        else:
            # Call the function with the data read from the file
            generate_lpips_plot(csv_data_string_from_file)
            
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found. Please ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred while reading or processing the file: {e}")

