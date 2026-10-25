# 2D Transform Editor
Weekend exercise project. Simple app to draw and transform polygons (quads).
Transform modes available:
- translation
- rigid (rotation)
- similarity (scaled rotation)
- affine
- projective

<img width="1249" height="1242" alt="image" src="https://github.com/user-attachments/assets/a525df14-e695-4769-be06-66bdd71cde50" />

## Running

Install dependencies:
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run:
```
python -m transform_editor.app --window_size 1500 1500
```
