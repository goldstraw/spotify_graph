import json
import sys
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
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    offset_datetime = start_datetime + timedelta(days=offset_days)
    return offset_datetime.strftime('%d/%m/%y')

def generate_plots(all_artists, proportions, bin_size, frames_per_bin, start_date, max_data_points_on_plot, shown_artists):
    # Create directory for temporary files. If it already exists, delete it and create a new one.
    if os.path.exists('spotify_graph_bin'):
        shutil.rmtree('spotify_graph_bin')
    os.mkdir('spotify_graph_bin')

    for i, week_props in enumerate(tqdm(proportions)):
        with plt.style.context('cyberpunk'):
            # Create figure and axis
            fig, ax = plt.subplots()
            # Only show last M data points
            N = frames_per_bin
            M = max_data_points_on_plot
            if i > M:
                start = (i + 2)/N - M/N
                stop = (i + 2)/N
                start *= bin_size / 7
                stop *= bin_size / 7
                x = np.linspace(start, stop, M)
            else:
                x = np.linspace(1/N, (i + 2)/N, i+1)
            # Only plot top n artists for each week
            artists_to_plot = [a for i,a in enumerate(all_artists) if week_props[i] > sorted(week_props, reverse=True)[shown_artists]]
            
            for j, artist in enumerate(all_artists):
                if i > M:
                    y = [max(p[j],0) for p in proportions[i-M+1:i+1]]
                else:
                    y = [max(p[j],0) for p in proportions[:i+1]]

                if max(y) > 0:
                    colours = [
                        "#02d7f2", "#f2e900", "#007aff", "#ff1111", "#980BBF", "#024059", "#2BF0FB", "#F2E307", "#F2CC0F",
                        "#E93CAC", "#1E22AA", "#59CBE8", "#00BCE1", "#ffd319", "#ff901f", "#ff2975", "#f222ff", "#8c1eff",
                        "#ff9a00", "#01ff1f", "#e3ff00", "#F2E50B", "#21B20C", "#FF7F50"
                    ]
                    colour = colours[hash(artist) % len(colours)]
                    ax.plot(x, y, label=artist[:30] if artist in artists_to_plot else '_nolegend_', color=colour)
            ax.set_xlabel('Week')
            ax.set_ylabel('Proportion of Listening Time')
            ax.set_title(f'Top 10 Artists by Proportion of Listening Time ({offset_date(start_date, i*7/N)})')
            # Clear legend
            ax.legend().remove()
            # Sort legend by proportion
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: week_props[list([art[:30] for art in all_artists]).index(x[1])], reverse=True))
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 0.5, 0.5, 0.5), loc='upper left')

            # Save to file, use leading zeros in filename to ensure correct order
            filename = f'spotify_graph_bin/graph_{i:05d}.png'
            plt.savefig(filename, bbox_inches=matplotlib.transforms.Bbox([[0,0],[8.5,5]]))
            plt.close()

def group_into_bins(data, bin_size):
    bin_totals = {}

    for entry in data:
        # Calculate the start of the bin
        end_time = datetime.strptime(entry['endTime'], '%Y-%m-%d %H:%M')
        total_days = (end_time - datetime(1970,1,1)).days
        start_of_bin = end_time - timedelta(days=total_days % bin_size)

        # Format the start of the bin as a string
        bin_str = start_of_bin.strftime('%Y-%m-%d')

        # Initialize a dictionary for this week if it doesn't exist yet
        if bin_str not in bin_totals:
            bin_totals[bin_str] = {}

        # Get the artist name and duration of this entry
        artist_name = entry['artistName']
        duration_ms = entry['msPlayed']
        duration_min = duration_ms / 1000 / 60

        # Add the duration to the total for this artist in this week
        if artist_name in bin_totals[bin_str]:
            bin_totals[bin_str][artist_name] += duration_min
        else:
            bin_totals[bin_str][artist_name] = duration_min

    # Return a list of dictionaries, one for each week
    return [bin_totals[week_str] for week_str in sorted(bin_totals)]


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Process the text to remove non-ASCII characters
        text = ''.join(c if ord(c) < 128 else ' ' for c in f.read())
        # Parse the JSON from the processed text
        data = json.loads(text)
    return data

def filter_top_n_values(bin_totals, n):
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
    clip = ImageSequenceClip('spotify_graph_bin/', fps=fps)
    video_name = video_name if video_name.endswith('.mp4') else video_name + '.mp4'
    clip.write_videofile(video_name, fps=fps, codec='libx264', preset='medium')

def clean_up():
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
    data = load_data('StreamingHistory0.json')

    # Group data by bin size and filter to top N artists for each bin
    unfiltered_bins = group_into_bins(data, bin_size)
    bins = filter_top_n_values(unfiltered_bins, shown_artists)

    # Create dictionary of all artists that appear in any week
    all_artists = set()
    for bin in bins:
        all_artists.update(set(bin.keys()))

    # Calculate proportion of listening time for each artist in each bin
    raw_proportions = calculate_proportions(unfiltered_bins, bins, all_artists)

    # Augment the proportions with interpolated data
    proportions = interpolate_proportions(raw_proportions, all_artists, frames_per_bin)

    # Generate frames for the video
    print("Generating frames...")
    first_day = data[0]["endTime"][:10]
    generate_plots(all_artists, proportions, bin_size, frames_per_bin, first_day, max_data_points_on_plot, shown_artists)

    # Create the video
    create_video(video_fps, video_name)
    clean_up()