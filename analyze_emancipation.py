import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Replace with your Client ID & Secret
client_id = '82bd1718fd854e39b94fc6eca3471ccf'
client_secret = '03a814e4f4d9495da11a01c306b607c2'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# Search for the album
album_name = "The Emancipation Procrastination"
artist_name = "Christian Scott"
result = sp.search(q=f"album:{album_name} artist:{artist_name}", type='album', limit=1)
album = result['albums']['items'][0]

# Get album ID & metadata
album_id = album['id']
print(f"Album: {album['name']}")
print(f"Release Date: {album['release_date']}")
print(f"Total Tracks: {album['total_tracks']}")

# Get all tracks
tracks = sp.album_tracks(album_id)['items']
for t in tracks:
    track_info = sp.track(t['id'])
    print(f"\nTrack: {t['name']}")
    print(f"  Popularity: {track_info['popularity']}")
    print(f"  Duration: {round(t['duration_ms']/60000, 2)} min")
