import json
from datetime import datetime, timedelta
import os
import shutil
import matplotlib.transforms
import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from moviepy.editor import ImageSequenceClip

def offset_date(start_date: str, offset_days: float) -> str:
    """Return a date offset by a number of days from a given date.

    Keyword arguments:
    start_date -- string in format 'YYYY-MM-DD'
    offset_days -- number of days to offset start_date by

    Returns:
    String in format 'DD/MM/YY'
    """
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    offset_datetime = start_datetime + timedelta(days=offset_days)
    return offset_datetime.strftime('%d/%m/%y')

def generate_plots(all_artists, proportions, bin_size, frames_per_bin, 
        start_date, max_data_points_on_plot, shown_artists):
    """Generate plots for each bin and save them to files.

    Keyword arguments:
    all_artists -- list of all artists in the data
    proportions -- list of dictionaries, one for each bin, containing
        the proportion of listening time for each artist
    bin_size -- number of days in each bin
    frames_per_bin -- number of frames to generate for each bin
    start_date -- string in format 'YYYY-MM-DD'
    max_data_points_on_plot -- maximum number of data points to show on the plot
    shown_artists -- number of artists to show on the plot
    """
    # Create directory for temporary files. If it already exists, delete it and create a new one.
    if os.path.exists('spotify_graph_bin'):
        shutil.rmtree('spotify_graph_bin')
    os.mkdir('spotify_graph_bin')

    for i, current_values in enumerate(tqdm(proportions)):
        with plt.style.context('cyberpunk'):
            # Create figure and axis
            fig, ax = plt.subplots()
            # Only show last m data points
            n = frames_per_bin
            m = max_data_points_on_plot
            if i > m:
                start = (i + 2)/n - m/n
                stop = (i + 2)/n
                start *= bin_size / 7
                stop *= bin_size / 7
                x = np.linspace(start, stop, m)
            else:
                x = np.linspace(1/n, (i + 2)/n, i+1)
            # Only plot top n artists for each bin
            lower_bound = sorted(current_values, reverse=True)[shown_artists]
            artists_to_plot = [a for i,a in enumerate(all_artists) if current_values[i] > lower_bound]

            for j, artist in enumerate(all_artists):
                if i > m:
                    y = [max(p[j],0) for p in proportions[i-m+1:i+1]]
                else:
                    y = [max(p[j],0) for p in proportions[:i+1]]

                if max(y) > 0:
                    colours = [
                        "#02d7f2", "#f2e900", "#007aff", "#ff5500", "#980BBF", "#FF7700",
                        "#2BF0FB", "#F2E307", "#F2CC0F", "#E93CAC", "#1E22AA", "#59CBE8",
                        "#00BCE1", "#ffd319", "#ff901f", "#ff2975", "#f222ff", "#8c1eff",
                        "#ff9a00", "#01ff1f", "#e3ff00", "#F2E50B", "#21B20C", "#FF7F50",
                        "#FFFFFF", "#990000", "#FF7F50", "#00FF00", "#00FFFF", "#0000FF",
                        "#FF00FF", "#FFD700", "#FFA500", "#FF0000", "#00FF00", "#00FFFF",
                        "#FF0066"
                    ]
                    colour = colours[hash(artist) % len(colours)]
                    label = artist[:30] if artist in artists_to_plot else '_nolegend_'
                    ax.plot(x, y, label=label, color=colour)
            ax.set_xlabel('Week')
            ax.set_ylabel('Proportion of Listening Time')
            ax.set_title(f'Top {shown_artists} Artists by Proportion of Listening Time ({offset_date(start_date, i*bin_size/n)})')
            # Clear legend
            ax.legend().remove()
            # Sort legend by proportion
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = zip(*sorted(zip(handles, labels),
                key=lambda x: current_values[list([art[:30] for art in all_artists]).index(x[1])], reverse=True))
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5, 0.5, 0.5), loc='upper left')

            # Save to file, use leading zeros in filename to ensure correct order
            filename = f'spotify_graph_bin/graph_{i:05d}.png'
            plt.savefig(filename, bbox_inches=matplotlib.transforms.Bbox([[0,0],[8.5,5]]))
            plt.close()

def group_into_bins(data, bin_size):
    """Group data into bins of size bin_size.

    Keyword arguments:
    data -- list of dictionaries, each containing the data for one bin
    bin_size -- number of days in each bin

    Returns:
    Dictionary containing the total listening time for each artist in each bin
    """
    bin_totals = {}

    for entry in data:
        # Calculate the start of the bin
        end_time = datetime.strptime(entry['endTime'], '%Y-%m-%d %H:%M')
        total_days = (end_time - datetime(1970,1,1)).days
        start_of_bin = end_time - timedelta(days=total_days % bin_size)

        # Format the start of the bin as a string
        bin_str = start_of_bin.strftime('%Y-%m-%d')

        # Initialize a dictionary for this bin if it doesn't exist yet
        if bin_str not in bin_totals:
            bin_totals[bin_str] = {}

        # Get the artist name and duration of this entry
        artist_name = entry['artistName']
        duration_ms = entry['msPlayed']
        duration_min = duration_ms / 1000 / 60

        # Add the duration to the total for this artist in this bin
        if artist_name in bin_totals[bin_str]:
            bin_totals[bin_str][artist_name] += duration_min
        else:
            bin_totals[bin_str][artist_name] = duration_min

    # Return a list of dictionaries, one for each bin
    return [bin_totals[bin_str] for bin_str in sorted(bin_totals)]


def load_data():
    """Load the data from the StreamingHistory files.
    
    Returns:
    A list of dictionaries, each containing the data for one bin.
    """
    # Get a list of all the StreamingHistory files in the current directory
    files = []
    for file in os.listdir():
        if file.startswith("StreamingHistory") and file.endswith(".json") and file.split('.')[0][-1].isdigit():
            files.append(file)

    if len(files) == 0:
        print("No StreamingHistory files found in the working directory.")
        exit()

    # Sort the files by the number at the end
    files.sort(key=lambda x: int(x.split('.')[0][-1]))

    # Load the data from each file
    data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            # Process the text to remove non-ASCII characters
            text = ''.join(c if ord(c) < 128 else ' ' for c in f.read())
            # Parse the JSON from the processed text
            file_data = json.loads(text)
            # Add the data to the list
            data += file_data

    return data

def filter_top_n_values(bin_totals, n):
    """Filter the top n values from each bin.

    Keyword arguments:
    bin_totals -- list of dictionaries, each containing the data for one bin
    n -- number of top values to keep

    Returns:
    A list of dictionaries, each containing the data for one bin
    """
    top_n_values = []

    # Iterate through each bin's totals
    for bin in bin_totals:
        # Sort the artists by total time listened
        sorted_artists = sorted(bin, key=bin.get, reverse=True)
        # Get the top n artists
        bin_top_n = sorted_artists[:n]
        # Create a new dictionary with only the top n artists
        top_n_dict = {artist: bin[artist] for artist in bin_top_n}
        # Append the new dictionary to the list of top n values
        top_n_values.append(top_n_dict)

    return top_n_values

def calculate_proportions(unfiltered_bins, bins, all_artists):
    """Calculate the proportions of each artist in each bin.

    Keyword arguments:
    unfiltered_bins -- list of dictionaries, each containing the data for one bin
    bins -- list of dictionaries, each containing the data for one bin
    all_artists -- list of all artists

    Returns:
    A list of lists, where each inner list contains the proportions of each artist
    in the corresponding bin.
    """
    proportions = []
    for i in range(len(unfiltered_bins)):
        bin_total = sum(unfiltered_bins[i].values())
        bin_proportions = []
        for artist in all_artists:
            if artist in bins[i]:
                bin_proportions.append(bins[i][artist] / bin_total)
            else:
                bin_proportions.append(0)
        proportions.append(bin_proportions)

    return proportions

def interpolate_proportions(raw_proportions, all_artists, frames_per_bin):
    """Interpolate the proportions of each artist in each bin.
    
    Keyword arguments:
    raw_proportions -- list of lists, where each inner list contains the proportions
                       of each artist in the corresponding bin.
    all_artists -- list of all artists
    frames_per_bin -- number of frames per bin

    Returns:
    A list of lists, where each inner list contains the interpolated proportions of
    each artist.
    """
    # Calculate the number of frames in the video
    # Remove any interpolated frames that would be outside the range of the data
    num_frames = len(raw_proportions)*frames_per_bin - frames_per_bin + 1
    proportions = [[0 for j in range(len(all_artists))] for i in range(num_frames)]
    for j in range(len(all_artists)):
        x_points = [*range(len(raw_proportions))]
        # Extract the jth artist's proportion data
        y_points = [raw_proportions[k][j] for k in range(len(raw_proportions))]
        tck = interpolate.splrep(x_points, y_points)
        # Generate the interpolated data for this artist
        artist_proportions = [interpolate.splev(i/frames_per_bin, tck) for i in range(num_frames)]

        # Add the interpolated data to the list of proportions
        for i in range(num_frames):
            proportions[i][j] = artist_proportions[i]

    return proportions

def create_video(fps, video_name):
    """Create the video from the images in the 'spotify_graph_bin' directory.

    Keyword arguments:
    fps -- frames per second
    video_name -- name of the video file
    """
    clip = ImageSequenceClip('spotify_graph_bin/', fps=fps)
    video_name = video_name if video_name.endswith('.mp4') else video_name + '.mp4'
    clip.write_videofile(video_name, fps=fps, codec='libx264', preset='medium')

def clean_up():
    """Delete the 'spotify_graph_bin' directory."""
    shutil.rmtree('spotify_graph_bin')


if __name__ == '__main__':
    # Load bin_size and shown_artists from standard input, with default values
    bin_size = int(input('Bin size (days) [Default=7]: ') or 7)
    shown_artists = int(input('Number of artists to show [Default=10]: ') or 10)
    frames_per_bin = int(input('Frames per bin [Default=7]: ') or 7)
    max_data_points_on_plot = int(input('Maximum number of data points on plot [Default=40]: ') or 40)
    video_fps = int(input('Video FPS [Default=7]: ') or 7)
    video_name = input('Video name [Default=output.mp4]: ') or 'output.mp4'

    # Load data
    data = load_data()

    # Group data by bin size and filter to top N artists for each bin
    unfiltered_bins = group_into_bins(data, bin_size)
    bins = filter_top_n_values(unfiltered_bins, shown_artists)

    # Create dictionary of all artists that appear in any bin
    all_artists = set()
    for b in bins:
        all_artists.update(set(b.keys()))

    # Calculate proportion of listening time for each artist in each bin
    raw_proportions = calculate_proportions(unfiltered_bins, bins, all_artists)

    # Augment the proportions with interpolated data
    proportions = interpolate_proportions(raw_proportions, all_artists, frames_per_bin)

    # Generate frames for the video
    print("Generating frames...")
    first_day = data[0]["endTime"][:10]
    generate_plots(all_artists, proportions, bin_size, frames_per_bin, 
        first_day, max_data_points_on_plot, shown_artists)

    # Create the video
    create_video(video_fps, video_name)
    clean_up()
