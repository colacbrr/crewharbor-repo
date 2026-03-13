# Media library layout

This folder defines the on-disk structure for media that will be indexed and
served by search. Keep file paths in manifests relative to `mir_project/` so
the backend can resolve them consistently across machines.

## Structure

```
media/
  manifests/
    videos.json
    audio.json
  videos/
    raw/          # original uploads
    processed/    # transcoded clips (uniform codec/resolution)
    thumbs/       # poster frames used in UI
    frames/       # optional keyframes for embedding and search
  audio/
    raw/          # original uploads
    processed/    # normalized audio (wav/flac)
    spectrograms/ # optional derived images
```

## Manifest format

Each manifest is a JSON array. Use the same schema as `metadata_sample.json`
so the search pipeline can re-use metadata across image/video/audio.

Minimum fields:
- `id`: unique string
- `type`: `video` or `audio`
- `path`: relative path to the main file under `mir_project/`
- `caption`: short description for search
- `tags`: list of keywords
- `source`: where the media came from
- `timestamp`: `{ "start_sec": 0.0, "end_sec": 0.0 }` or `null`
- `extra`: width/height/duration/fps

Optional fields you can add later:
- `preview_image`: poster frame for video cards
- `segments`: array of clip segments with their own captions

## Notes
- Put raw uploads in `videos/raw` or `audio/raw`.
- Store derived assets (transcoded files, poster frames, spectrograms) in the
  matching `processed` or `thumbs` folders.
- Add entries to the manifest when new media is added. This is what will power
  search in the next indexing step.
