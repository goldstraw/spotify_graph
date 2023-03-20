# Spotify Graph Generator
Spotify Graph Generator is a Python project that converts raw Spotify listening data into an animated graph. The graph shows the proportion of total listening time occupied by each artist over time. The program is designed to provide users with an intuitive and interactive way to visualize their listening history over time.
## Example
https://user-images.githubusercontent.com/65353874/226466808-b2c493f4-3031-49f1-a2f6-d10e57a31a94.mp4

## Installation
To install Spotify Graph Generator, you'll need to have pipenv installed. You can do this by running:
```bash
pip install pipenv
```

Then, navigate to the project directory and enter the virtual environment:
```bash
cd /path/to/project
pipenv shell
```

Finally, install the required dependencies:
```bash
pipenv sync
```

## Usage
Before running the program, ensure that you have downloaded your Spotify listening history from the [Privacy Settings](https://www.spotify.com/us/account/privacy/). Once Spotify sends your listening history, place all your StreamingHistoryX.json files into the same directory as spotify_graph.py

Once the previous steps are complete, you can run the program using:
```bash
python spotify_graph.py
```
The program will prompt you to enter the settings for your output video. You can define the following attributes:
- **Bin Size**: To smooth the graph, the data is clustered and interpolated. The bin size is the number of days to cluster into a single data point.
- **Number of artists to show**: The program will only show your top N artists.
- **Frames per bin**: The number of frames to generate per data cluster (the number of interpolated points).
- **Maximum number of data points on plot**: If the number of interpolated points exceeds this value, the graph will scroll sideways.
- **Video FPS**: The number of frames to show in a second of the video.
- **Video name**: The name of the output file.

After entering the desired settings, the program will begin generating the graph. This may take several minutes, depending on the amount of data being processed. Once the video has been generated, it will be saved into the same directory as the program with the specified name.

## Contributing
If you would like to contribute to Spotify Graph Generator, please follow these steps:

- Fork the repository
- Create a new branch
- Commit your changes
- Push to the branch
- Create a new pull request

# License
Spotify Graph Generator is licensed under the GNU Affero General Public License. See [LICENSE](LICENSE) for more information.
