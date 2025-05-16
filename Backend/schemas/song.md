# Song Schema

Each user has a `songs` array. Each item (song) in that array follows this structure:

```json
{
  "id": "uuid-string",
  "title": "Song Title",
  "artist": "Optional Artist",
  "notes": [ ... ],
  "tags": ["tag1", "tag2"]
}
